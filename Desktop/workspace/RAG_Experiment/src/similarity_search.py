"""
Standalone similarity search implementations.

Supported methods:
  cosine      – FAISS IndexFlatIP with L2-normalised vectors
  euclidean   – FAISS IndexFlatL2
  dot_product – FAISS IndexFlatIP (raw, no normalisation)
  manhattan   – Brute-force L1 via sklearn
  bm25        – Sparse BM25Okapi (rank_bm25)
  hybrid      – Reciprocal Rank Fusion of BM25 + cosine
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.metrics import pairwise_distances

from .rrf import reciprocal_rank_fusion


def _tokenize(text: str) -> List[str]:
    """Minimal whitespace tokeniser (works for CJK + Latin)."""
    return text.lower().split()


class SimilaritySearcher:
    """
    Build once, query repeatedly with any of the supported methods.

    Parameters
    ----------
    chunks   : list of {"text": str, "metadata": dict}
    vectors  : (N, D) float32 – raw (non-normalised) embeddings
    """

    def __init__(self, chunks: List[Dict[str, Any]], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self._dim = vectors.shape[1]

        # Pre-normalise for cosine / dot-product variants
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        self._norm_vectors = (self.vectors / norms).astype(np.float32)

        # FAISS indices (built lazily)
        self._idx_flat_l2: faiss.IndexFlatL2 | None = None
        self._idx_flat_ip: faiss.IndexFlatIP | None = None
        self._idx_flat_ip_raw: faiss.IndexFlatIP | None = None

        # BM25 index (built lazily)
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: List[List[str]] | None = None

        logger.debug(f"SimilaritySearcher ready – {len(chunks)} chunks, dim={self._dim}")

    # ── private builders ────────────────────────────────────────────────

    def _build_l2(self) -> faiss.IndexFlatL2:
        if self._idx_flat_l2 is None:
            idx = faiss.IndexFlatL2(self._dim)
            idx.add(self.vectors)
            self._idx_flat_l2 = idx
        return self._idx_flat_l2

    def _build_ip_norm(self) -> faiss.IndexFlatIP:
        """Inner product on normalised vectors ≡ cosine similarity."""
        if self._idx_flat_ip is None:
            idx = faiss.IndexFlatIP(self._dim)
            idx.add(self._norm_vectors)
            self._idx_flat_ip = idx
        return self._idx_flat_ip

    def _build_ip_raw(self) -> faiss.IndexFlatIP:
        """Raw inner product (dot product) without normalisation."""
        if self._idx_flat_ip_raw is None:
            idx = faiss.IndexFlatIP(self._dim)
            idx.add(self.vectors)
            self._idx_flat_ip_raw = idx
        return self._idx_flat_ip_raw

    def _build_bm25(self) -> BM25Okapi:
        if self._bm25 is None:
            self._corpus_tokens = [_tokenize(c["text"]) for c in self.chunks]
            self._bm25 = BM25Okapi(self._corpus_tokens)
        return self._bm25

    # ── public search API ────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        query_vector: np.ndarray,
        method: str,
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Returns (results, elapsed_seconds).
        Each result: {"rank", "score", "text", "metadata"}
        """
        t0 = time.perf_counter()
        results = self._dispatch(query_text, query_vector, method, top_k)
        elapsed = time.perf_counter() - t0
        return results, elapsed

    def _dispatch(self, query_text, query_vector, method, top_k):
        q = query_vector.astype(np.float32)
        if method == "cosine":
            return self._search_cosine(q, top_k)
        elif method == "euclidean":
            return self._search_euclidean(q, top_k)
        elif method == "dot_product":
            return self._search_dot_product(q, top_k)
        elif method == "manhattan":
            return self._search_manhattan(q, top_k)
        elif method == "bm25":
            return self._search_bm25(query_text, top_k)
        elif method == "hybrid":
            return self._search_hybrid(query_text, q, top_k)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _search_cosine(self, q: np.ndarray, top_k: int) -> List[Dict]:
        norm = np.linalg.norm(q) + 1e-10
        q_norm = (q / norm).reshape(1, -1)
        idx = self._build_ip_norm()
        scores, indices = idx.search(q_norm, top_k)
        return self._format(indices[0], scores[0])

    def _search_euclidean(self, q: np.ndarray, top_k: int) -> List[Dict]:
        idx = self._build_l2()
        dists, indices = idx.search(q.reshape(1, -1), top_k)
        # Convert L2 distance to similarity score (lower = better → negate)
        scores = -dists[0]
        return self._format(indices[0], scores)

    def _search_dot_product(self, q: np.ndarray, top_k: int) -> List[Dict]:
        idx = self._build_ip_raw()
        scores, indices = idx.search(q.reshape(1, -1), top_k)
        return self._format(indices[0], scores[0])

    def _search_manhattan(self, q: np.ndarray, top_k: int) -> List[Dict]:
        q_2d = q.reshape(1, -1)
        dists = pairwise_distances(q_2d, self.vectors, metric="manhattan")[0]
        top_indices = np.argsort(dists)[:top_k]
        scores = -dists[top_indices]   # negate so higher = better
        return self._format(top_indices, scores)

    def _search_bm25(self, query_text: str, top_k: int) -> List[Dict]:
        bm25 = self._build_bm25()
        tokens = _tokenize(query_text)
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return self._format(top_indices, scores[top_indices])

    def _search_hybrid(self, query_text: str, q: np.ndarray, top_k: int, k_rrf: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion of BM25 and cosine rankings."""
        bm25 = self._build_bm25()
        bm25_scores = bm25.get_scores(_tokenize(query_text))
        bm25_ranks = [int(i) for i in np.argsort(bm25_scores)[::-1]]

        norm = np.linalg.norm(q) + 1e-10
        q_norm = (q / norm).reshape(1, -1)
        _, cosine_indices = self._build_ip_norm().search(q_norm, len(self.chunks))
        cosine_ranks = [int(i) for i in cosine_indices[0]]

        fused = reciprocal_rank_fusion(
            [bm25_ranks, cosine_ranks],
            top_k=top_k,
            k_rrf=k_rrf,
        )
        sorted_ids = np.array([doc_id for doc_id, _ in fused])
        scores = np.array([score for _, score in fused])
        return self._format(sorted_ids, scores)

    def _format(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if 0 <= idx < len(self.chunks):
                results.append({
                    "rank": rank + 1,
                    "score": float(score),
                    "text": self.chunks[idx]["text"],
                    "metadata": self.chunks[idx]["metadata"],
                })
        return results
