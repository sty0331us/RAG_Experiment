"""
Interactive single-query RAG CLI.

Usage:
    python experiments/ask.py "How do I clean the air conditioner filter?"
    python experiments/ask.py --framework langchain --method hybrid
    python experiments/ask.py   # prompts for a question interactively
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.data_loader import chunk_documents, load_pdfs
from src.embeddings import EmbeddingModel
from src.factory import build_rag


def parse_args():
    p = argparse.ArgumentParser(description="Ask a single question against the PDF corpus")
    p.add_argument("question", nargs="?", default=None, help="Question to ask")
    p.add_argument("--framework", default="langchain",
                   choices=["llamaindex", "langchain", "haystack"])
    p.add_argument("--method", default="hybrid",
                   choices=["cosine", "euclidean", "dot_product", "manhattan", "bm25", "hybrid"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-embed-cache", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    question = args.question or input("Question: ").strip()
    if not question:
        logger.error("No question provided.")
        sys.exit(1)

    cfg = ExperimentConfig(top_k=args.top_k)
    if args.no_embed_cache:
        cfg.use_embedding_cache = False

    documents = load_pdfs(cfg.data_dir)
    if not documents:
        logger.error(f"No PDFs found in {cfg.data_dir}")
        sys.exit(1)

    chunks = chunk_documents(documents, cfg.chunk_size, cfg.chunk_overlap)
    embed_model = EmbeddingModel(
        cfg.embedding_model,
        base_url=cfg.ollama_base_url,
        cache_dir=cfg.cache_dir if cfg.use_embedding_cache else None,
    )
    cfg.embedding_dimension = embed_model.dimension
    vectors = embed_model.embed([c["text"] for c in chunks], normalize=False)

    rag = build_rag(args.framework, chunks, vectors, cfg, args.method)
    q_vec = embed_model.embed_query(question, normalize=False)
    result = rag.query(question, query_vector=q_vec)

    print("\n" + "=" * 60)
    print(f"Framework : {result['framework']} / {result['method']}")
    print(f"Latency   : {result['total_time_s']:.2f}s (gen {result['gen_time_s']:.2f}s)")
    print("=" * 60)
    print("\nAnswer:\n")
    print(result["answer"].strip())
    print("\nRetrieved chunks:")
    for chunk in result.get("retrieved_chunks", []):
        score = chunk.get("score")
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        preview = chunk["text"].replace("\n", " ")[:140]
        print(f"  [{chunk.get('rank', '?')}] score={score_s}  {preview}…")


if __name__ == "__main__":
    main()
