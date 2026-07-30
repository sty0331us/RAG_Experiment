"""LangChain RAG implementations for each similarity method."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_ollama import OllamaEmbeddings
from loguru import logger

from .config import ExperimentConfig
from .prompts import format_answer_prompt
from .rrf import fuse_text_rankings
from .similarity_search import SimilaritySearcher

_DISTANCE_MAP = {
    "cosine": DistanceStrategy.COSINE,
    "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
    "dot_product": DistanceStrategy.MAX_INNER_PRODUCT,
}


def _build_llm(cfg: ExperimentConfig):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=cfg.llm_model, base_url=cfg.ollama_base_url, temperature=cfg.temperature)


class LangChainRAG:
    """
    Wraps a LangChain FAISS vectorstore (or BM25Retriever) for a single similarity method.
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

        logger.info(f"[LangChain] Building index – method={method}")

        self._embed_model = OllamaEmbeddings(
            model=cfg.embedding_model,
            base_url=cfg.ollama_base_url,
        )
        self._llm = _build_llm(cfg)

        docs = [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

        if method == "bm25":
            self._retriever = BM25Retriever.from_documents(docs, k=cfg.top_k)
            self._qa_chain = None

        elif method == "manhattan":
            # FAISS L2 for candidate retrieval; re-rank by L1 via SimilaritySearcher
            self._vectorstore = FAISS.from_documents(
                docs, self._embed_model,
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
            self._searcher = SimilaritySearcher(chunks, vectors)
            self._retriever = None
            self._qa_chain = None

        elif method == "hybrid":
            # BM25 + cosine FAISS, fused with RRF
            self._vectorstore = FAISS.from_documents(
                docs, self._embed_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            self._bm25_retriever = BM25Retriever.from_documents(docs, k=cfg.top_k * 2)
            self._vector_retriever = self._vectorstore.as_retriever(search_kwargs={"k": cfg.top_k * 2})
            self._retriever = None
            self._qa_chain = None

        else:
            strategy = _DISTANCE_MAP[method]
            self._vectorstore = FAISS.from_documents(docs, self._embed_model, distance_strategy=strategy)
            self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": cfg.top_k})
            self._qa_chain = None

        logger.info(f"[LangChain] Index ready – method={method}")

    def query(self, question: str, query_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if self.method == "bm25":
            docs = self._retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer, gen_time = self._generate_answer(question, context)
            retrieved = [{"rank": i + 1, "score": None, "text": d.page_content} for i, d in enumerate(docs)]

        elif self.method == "manhattan":
            results, _ = self._searcher.search(question, query_vector, "manhattan", self.cfg.top_k)
            context = "\n\n".join(r["text"] for r in results)
            answer, gen_time = self._generate_answer(question, context)
            retrieved = results

        elif self.method == "hybrid":
            vec_docs = self._vector_retriever.invoke(question)
            bm25_docs = self._bm25_retriever.invoke(question)
            retrieved = fuse_text_rankings(
                [
                    [d.page_content for d in vec_docs],
                    [d.page_content for d in bm25_docs],
                ],
                top_k=self.cfg.top_k,
            )
            context = "\n\n".join(r["text"] for r in retrieved)
            answer, gen_time = self._generate_answer(question, context)

        else:
            docs = self._retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            answer, gen_time = self._generate_answer(question, context)
            retrieved = [
                {"rank": i + 1, "score": None, "text": d.page_content}
                for i, d in enumerate(docs)
            ]

        total_time = time.perf_counter() - t0
        return {
            "framework": "langchain",
            "method": self.method,
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved,
            "total_time_s": total_time,
            "gen_time_s": gen_time,
        }

    def _generate_answer(self, question: str, context: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        prompt = format_answer_prompt(question, context)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, time.perf_counter() - t0
