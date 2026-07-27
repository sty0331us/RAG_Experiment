"""
Main experiment runner.

Usage:
    python experiments/run_experiments.py
    python experiments/run_experiments.py --frameworks llamaindex --methods cosine bm25
    python experiments/run_experiments.py --llm-provider ollama --llm-model llama3.2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.data_loader import chunk_documents, load_pdfs
from src.embeddings import EmbeddingModel
from src.evaluator import evaluate_single, summarise_results
from src.similarity_search import SimilaritySearcher
from src.visualizer import generate_all_plots


def parse_args():
    p = argparse.ArgumentParser(description="RAG Experiment Runner")
    p.add_argument("--frameworks", nargs="+", default=None,
                   choices=["llamaindex", "langchain", "haystack"],
                   help="Frameworks to test (default: all three)")
    p.add_argument("--methods", nargs="+", default=None,
                   choices=["cosine", "euclidean", "dot_product", "manhattan", "bm25", "hybrid"],
                   help="Similarity methods to test (default: all)")
    p.add_argument("--queries", nargs="+", default=None,
                   help="Custom query strings (default: sample_queries.py)")
    p.add_argument("--llm-provider", default=None, choices=["openai", "ollama"])
    p.add_argument("--llm-model", default=None)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--chunk-overlap", type=int, default=64)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--similarity-only", action="store_true",
                   help="Only run standalone similarity search (no LLM calls)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
    )
    if args.frameworks:
        cfg.frameworks = args.frameworks
    if args.methods:
        cfg.similarity_methods = args.methods
    if args.llm_provider:
        cfg.llm_provider = args.llm_provider
    if args.llm_model:
        cfg.llm_model = args.llm_model

    logger.info("=" * 60)
    logger.info("RAG Experiment – LlamaIndex vs LangChain vs Haystack")
    logger.info(f"Frameworks : {cfg.frameworks}")
    logger.info(f"Methods    : {cfg.similarity_methods}")
    logger.info(f"LLM        : {cfg.llm_provider}/{cfg.llm_model}")
    logger.info("=" * 60)

    # ── 1. Load documents ─────────────────────────────────────────────
    documents = load_pdfs(cfg.data_dir)
    if not documents:
        logger.error(f"No PDFs found in {cfg.data_dir}. Add PDF files and retry.")
        sys.exit(1)

    chunks = chunk_documents(documents, cfg.chunk_size, cfg.chunk_overlap)

    # ── 2. Build embeddings ───────────────────────────────────────────
    embed_model = EmbeddingModel(cfg.embedding_model, base_url=cfg.ollama_base_url)
    cfg.embedding_dimension = embed_model.dimension

    logger.info("Encoding all chunks …")
    t_embed = time.perf_counter()
    vectors = embed_model.embed([c["text"] for c in chunks], normalize=False)
    logger.info(f"Embedding done in {time.perf_counter() - t_embed:.1f}s – shape {vectors.shape}")

    def embed_fn(texts):
        return embed_model.embed(texts, normalize=False)

    # ── 3. Load queries ───────────────────────────────────────────────
    if args.queries:
        queries = args.queries
    else:
        from experiments.sample_queries import QUERIES
        queries = QUERIES

    logger.info(f"Running {len(queries)} queries")

    # ── 4. Standalone similarity search comparison ────────────────────
    logger.info("\n--- Standalone Similarity Search Benchmark ---")
    searcher = SimilaritySearcher(chunks, vectors)
    similarity_results = []

    for q in tqdm(queries, desc="Similarity search"):
        q_vec = embed_model.embed_query(q, normalize=False)
        for method in cfg.similarity_methods:
            results, elapsed = searcher.search(q, q_vec, method, cfg.top_k)
            similarity_results.append({
                "query": q,
                "method": method,
                "elapsed_s": elapsed,
                "top1_text": results[0]["text"][:120] if results else "",
                "top1_score": results[0]["score"] if results else None,
                "num_retrieved": len(results),
            })

    _save_json(similarity_results, cfg.results_dir / "raw" / "similarity_only.json")

    if args.similarity_only:
        logger.info("--similarity-only flag set, skipping RAG pipeline runs.")
        _print_similarity_summary(similarity_results, cfg.similarity_methods)
        return

    # ── 5. Full RAG experiments ───────────────────────────────────────
    all_rag_results = []

    for framework in cfg.frameworks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Framework: {framework.upper()}")
        logger.info(f"{'='*50}")

        for method in cfg.similarity_methods:
            logger.info(f"\n  Method: {method}")
            try:
                rag = _build_rag(framework, chunks, vectors, cfg, method)
            except Exception as exc:
                logger.error(f"Failed to build {framework}/{method}: {exc}")
                continue

            for q in tqdm(queries, desc=f"  {framework}/{method}", leave=False):
                q_vec = embed_model.embed_query(q, normalize=False)
                try:
                    result = rag.query(q, query_vector=q_vec)
                    result = evaluate_single(result, q_vec, embed_fn)
                    all_rag_results.append(result)
                    logger.debug(
                        f"    Q: {q[:50]}… | time={result['total_time_s']:.2f}s "
                        f"| ctx_rel={result['avg_ctx_relevance']:.3f}"
                    )
                except Exception as exc:
                    logger.error(f"Query failed ({framework}/{method}): {exc}")

    # ── 6. Save raw results ───────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = cfg.results_dir / "raw" / f"rag_results_{ts}.json"
    _save_json(all_rag_results, raw_path)
    logger.info(f"\nRaw results saved: {raw_path}")

    # ── 7. Summarise & visualise ──────────────────────────────────────
    summary = summarise_results(all_rag_results)
    generate_all_plots(summary, cfg.results_dir / "plots")

    logger.info("\n✓ Experiment complete. Check results/plots/ for visualisations.")


# ── helpers ──────────────────────────────────────────────────────────────


def _build_rag(framework: str, chunks, vectors, cfg, method):
    if framework == "llamaindex":
        from src.llamaindex_rag import LlamaIndexRAG
        return LlamaIndexRAG(chunks, vectors, cfg, method)
    elif framework == "haystack":
        from src.haystack_rag import HaystackRAG
        return HaystackRAG(chunks, vectors, cfg, method)
    else:
        from src.langchain_rag import LangChainRAG
        return LangChainRAG(chunks, vectors, cfg, method)


def _save_json(data, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def _print_similarity_summary(results, methods):
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby("method")[["elapsed_s", "top1_score"]].mean().round(4)
    print("\n── Similarity Search Speed & Score Summary ──")
    print(summary.to_string())


if __name__ == "__main__":
    main()
