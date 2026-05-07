"""LlamaIndex RAG implementations for each similarity method."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from loguru import logger

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from .config import ExperimentConfig
from .similarity_search import SimilaritySearcher


def _build_llm(cfg: ExperimentConfig):
    from llama_index.llms.ollama import Ollama
    return Ollama(model=cfg.llm_model, base_url=cfg.ollama_base_url, temperature=cfg.temperature)


def _build_faiss_index(dim: int, method: str) -> faiss.Index:
    if method in ("cosine", "dot_product"):
        return faiss.IndexFlatIP(dim)
    else:  # euclidean (manhattan falls back to L2 here; raw manhattan is in SimilaritySearcher)
        return faiss.IndexFlatL2(dim)


class LlamaIndexRAG:
    """
    Wraps a LlamaIndex VectorStoreIndex (or BM25Retriever) for a single similarity method.
    For hybrid: combines BM25 retriever + cosine VectorStoreIndex with RRF.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
        cfg: ExperimentConfig,
        method: str,
    ):
        self.cfg = cfg
        self.method = method
        self.chunks = chunks

        logger.info(f"[LlamaIndex] Building index – method={method}")

        embed_model = OllamaEmbedding(model_name=cfg.embedding_model, base_url=cfg.ollama_base_url)
        llm = _build_llm(cfg)

        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = cfg.chunk_size
        Settings.chunk_overlap = cfg.chunk_overlap

        # Convert chunks → LlamaIndex Documents
        documents = [
            Document(text=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

        if method == "bm25":
            # Build a VectorStoreIndex first (needed for BM25Retriever)
            self._index = VectorStoreIndex.from_documents(documents)
            self._retriever = BM25Retriever.from_defaults(
                index=self._index, similarity_top_k=cfg.top_k
            )
            self._query_engine = None  # will use retriever directly

        elif method == "hybrid":
            # Cosine VectorStoreIndex + BM25Retriever, fused with RRF
            faiss_idx = _build_faiss_index(cfg.embedding_dimension, "cosine")
            vector_store = FaissVectorStore(faiss_index=faiss_idx)
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
            self._vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
            self._vector_retriever = self._vector_index.as_retriever(similarity_top_k=cfg.top_k * 2)
            self._bm25_retriever = BM25Retriever.from_defaults(
                index=self._vector_index, similarity_top_k=cfg.top_k * 2
            )
            self._query_engine = None

        else:
            # Dense similarity via FAISS (cosine, euclidean, dot_product)
            faiss_idx = _build_faiss_index(cfg.embedding_dimension, method)
            vector_store = FaissVectorStore(faiss_index=faiss_idx)
            storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
            self._query_engine = self._index.as_query_engine(similarity_top_k=cfg.top_k)

        # Manhattan uses cosine index but we re-rank via SimilaritySearcher post-retrieval
        if method == "manhattan":
            self._searcher = SimilaritySearcher(chunks, vectors)

        logger.info(f"[LlamaIndex] Index ready – method={method}")

    def query(self, question: str, query_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if self.method == "bm25":
            nodes = self._retriever.retrieve(question)
            context = "\n\n".join(n.text for n in nodes)
            answer, gen_time = self._generate_answer(question, context)
            retrieved = [{"rank": i + 1, "score": n.score, "text": n.text} for i, n in enumerate(nodes)]

        elif self.method == "hybrid":
            vec_nodes = self._vector_retriever.retrieve(question)
            bm25_nodes = self._bm25_retriever.retrieve(question)
            # RRF fusion
            scores: Dict[str, float] = {}
            texts: Dict[str, str] = {}
            k = 60
            for rank, n in enumerate(vec_nodes):
                key = n.text[:80]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                texts[key] = n.text
            for rank, n in enumerate(bm25_nodes):
                key = n.text[:80]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                texts[key] = n.text
            top_keys = sorted(scores, key=scores.get, reverse=True)[: self.cfg.top_k]
            context = "\n\n".join(texts[k] for k in top_keys)
            retrieved = [{"rank": i + 1, "score": scores[k], "text": texts[k]} for i, k in enumerate(top_keys)]
            answer, gen_time = self._generate_answer(question, context)

        elif self.method == "manhattan" and query_vector is not None:
            # Retrieve 2× candidates via L2, then re-rank by L1
            nodes = self._index.as_retriever(similarity_top_k=self.cfg.top_k * 3).retrieve(question)
            cand_chunks = [{"text": n.text, "metadata": {}} for n in nodes]
            cand_vecs_raw = np.array([self.cfg._embed_fn(n.text) for n in nodes], dtype=np.float32) if hasattr(self.cfg, "_embed_fn") else None
            # Fall back to SimilaritySearcher for proper manhattan
            results, _ = self._searcher.search(question, query_vector, "manhattan", self.cfg.top_k)
            context = "\n\n".join(r["text"] for r in results)
            retrieved = results
            answer, gen_time = self._generate_answer(question, context)

        else:
            response = self._query_engine.query(question)
            answer = str(response)
            retrieved = [
                {"rank": i + 1, "score": n.score, "text": n.text}
                for i, n in enumerate(response.source_nodes)
            ]
            gen_time = time.perf_counter() - t0

        total_time = time.perf_counter() - t0
        return {
            "framework": "llamaindex",
            "method": self.method,
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "total_time_s": total_time,
            "gen_time_s": gen_time,
        }

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        prompt = (
            f"다음 문서 내용을 참고하여 질문에 답변해주세요.\n\n"
            f"문서 내용:\n{context}\n\n"
            f"질문: {question}\n\n"
            f"답변:"
        )
        response = Settings.llm.complete(prompt)
        return str(response), time.perf_counter() - t0
