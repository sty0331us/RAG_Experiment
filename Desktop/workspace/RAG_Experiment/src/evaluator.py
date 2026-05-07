"""
Evaluation metrics for RAG experiment results.

Metrics computed:
  - retrieval_time_s  : time spent retrieving (total - gen)
  - gen_time_s        : LLM generation time
  - total_time_s      : end-to-end latency
  - avg_ctx_relevance : mean cosine similarity between query embedding and retrieved chunks
  - answer_length     : character count of the generated answer
  - num_retrieved     : actual number of chunks retrieved
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


def compute_context_relevance(
    query_vector: np.ndarray,
    retrieved_chunks: List[Dict[str, Any]],
    embed_fn,
) -> float:
    """Average cosine similarity between the query and each retrieved chunk."""
    if not retrieved_chunks:
        return 0.0
    texts = [c["text"] for c in retrieved_chunks]
    chunk_vecs = embed_fn(texts)
    sims = cosine_similarity(query_vector.reshape(1, -1), chunk_vecs)[0]
    return float(np.mean(sims))


def evaluate_single(result: Dict[str, Any], query_vector: np.ndarray, embed_fn) -> Dict[str, Any]:
    """Add evaluation metrics to a single RAG result dict (in-place, also returns it)."""
    retrieved = result.get("retrieved_chunks", [])
    total = result.get("total_time_s", 0.0)
    gen = result.get("gen_time_s", total)

    result["retrieval_time_s"] = max(0.0, total - gen)
    result["num_retrieved"] = len(retrieved)
    result["answer_length"] = len(result.get("answer", ""))
    result["avg_ctx_relevance"] = compute_context_relevance(query_vector, retrieved, embed_fn)
    return result


def summarise_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Group results by (framework, method) and compute mean metrics.
    Returns a dict keyed by "<framework>/<method>".
    """
    from collections import defaultdict
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_results:
        key = f"{r['framework']}/{r['method']}"
        groups[key].append(r)

    summary = {}
    metric_keys = ["total_time_s", "retrieval_time_s", "gen_time_s", "avg_ctx_relevance", "answer_length", "num_retrieved"]
    for key, results in groups.items():
        summary[key] = {
            m: float(np.mean([r.get(m, 0) for r in results]))
            for m in metric_keys
        }
        summary[key]["n_queries"] = len(results)
    return summary
