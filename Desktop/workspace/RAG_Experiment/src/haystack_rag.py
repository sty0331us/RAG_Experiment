"""Haystack RAG implementations for each similarity method."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from haystack import Document
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from loguru import logger

from .config import ExperimentConfig
from .prompts import format_answer_prompt
from .rrf import fuse_text_rankings
from .similarity_search import SimilaritySearcher


class HaystackRAG:
    """
    Haystack 2.x RAG for a single similarity method.

    Dense cosine / dot_product use InMemoryDocumentStore + InMemoryEmbeddingRetriever.
    BM25 uses InMemoryBM25Retriever. Hybrid fuses BM25 + cosine with RRF.
    Euclidean / manhattan use the shared SimilaritySearcher, then Haystack generation.
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
        self.vectors = vectors.astype(np.float32)

        logger.info(f"[Haystack] Building pipeline – method={method}")

        self._llm = OllamaChatGenerator(
            model=cfg.llm_model,
            url=cfg.ollama_base_url,
            generation_kwargs={"temperature": cfg.temperature},
        )
        self._text_embedder = OllamaTextEmbedder(
            model=cfg.embedding_model,
            url=cfg.ollama_base_url,
        )

        docs = [
            Document(
                content=c["text"],
                meta={**(c.get("metadata") or {}), "chunk_idx": i},
                embedding=vectors[i].astype(np.float32).tolist(),
            )
            for i, c in enumerate(chunks)
        ]

        if method == "bm25":
            store = InMemoryDocumentStore()
            store.write_documents(docs)
            self._bm25_retriever = InMemoryBM25Retriever(document_store=store, top_k=cfg.top_k)

        elif method in ("cosine", "dot_product"):
            sim_fn = "cosine" if method == "cosine" else "dot_product"
            store = InMemoryDocumentStore(embedding_similarity_function=sim_fn)
            store.write_documents(docs)
            self._embed_retriever = InMemoryEmbeddingRetriever(
                document_store=store, top_k=cfg.top_k
            )

        elif method == "hybrid":
            dense_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
            dense_store.write_documents(docs)
            self._embed_retriever = InMemoryEmbeddingRetriever(
                document_store=dense_store, top_k=cfg.top_k * 2
            )
            bm25_store = InMemoryDocumentStore()
            bm25_store.write_documents(
                [
                    Document(
                        content=c["text"],
                        meta={**(c.get("metadata") or {}), "chunk_idx": i},
                    )
                    for i, c in enumerate(chunks)
                ]
            )
            self._bm25_retriever = InMemoryBM25Retriever(
                document_store=bm25_store, top_k=cfg.top_k * 2
            )

        else:
            # euclidean / manhattan — shared FAISS / L1 backends, Haystack for generation
            self._searcher = SimilaritySearcher(chunks, vectors)

        logger.info(f"[Haystack] Pipeline ready – method={method}")

    def query(self, question: str, query_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if self.method == "bm25":
            docs = self._bm25_retriever.run(query=question)["documents"]
            context = "\n\n".join(d.content for d in docs)
            retrieved = [
                {"rank": i + 1, "score": d.score, "text": d.content}
                for i, d in enumerate(docs)
            ]
            answer, gen_time = self._generate_answer(question, context)

        elif self.method in ("cosine", "dot_product"):
            embed_out = self._text_embedder.run(text=question)
            docs = self._embed_retriever.run(
                query_embedding=embed_out["embedding"]
            )["documents"]
            context = "\n\n".join(d.content for d in docs)
            retrieved = [
                {"rank": i + 1, "score": d.score, "text": d.content}
                for i, d in enumerate(docs)
            ]
            answer, gen_time = self._generate_answer(question, context)

        elif self.method == "hybrid":
            embed_out = self._text_embedder.run(text=question)
            vec_docs = self._embed_retriever.run(
                query_embedding=embed_out["embedding"]
            )["documents"]
            bm25_docs = self._bm25_retriever.run(query=question)["documents"]
            retrieved = fuse_text_rankings(
                [
                    [d.content for d in vec_docs],
                    [d.content for d in bm25_docs],
                ],
                top_k=self.cfg.top_k,
            )
            context = "\n\n".join(r["text"] for r in retrieved)
            answer, gen_time = self._generate_answer(question, context)

        else:
            # euclidean / manhattan
            results, _ = self._searcher.search(
                question, query_vector, self.method, self.cfg.top_k
            )
            context = "\n\n".join(r["text"] for r in results)
            retrieved = results
            answer, gen_time = self._generate_answer(question, context)

        return {
            "framework": "haystack",
            "method": self.method,
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "total_time_s": time.perf_counter() - t0,
            "gen_time_s": gen_time,
        }

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        prompt = format_answer_prompt(question, context)
        result = self._llm.run(messages=prompt)
        replies = result.get("replies") or []
        if not replies:
            answer = ""
        else:
            reply = replies[0]
            answer = reply.text if hasattr(reply, "text") else str(reply)
        return answer, time.perf_counter() - t0
