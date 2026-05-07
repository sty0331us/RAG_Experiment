"""LangGraph RAG implementation — retrieve → generate state graph."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
from loguru import logger

from .config import ExperimentConfig
from .similarity_search import SimilaritySearcher

_DISTANCE_MAP = {
    "cosine": DistanceStrategy.COSINE,
    "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
    "dot_product": DistanceStrategy.MAX_INNER_PRODUCT,
}

_ANSWER_PROMPT = (
    "다음 문서 내용을 참고하여 질문에 답변해주세요.\n\n"
    "문서 내용:\n{context}\n\n"
    "질문: {question}\n\n"
    "답변:"
)


class RAGState(TypedDict):
    question: str
    query_vector: Optional[List[float]]
    context: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    retrieval_time_s: float
    gen_time_s: float


class LangGraphRAG:
    """
    Two-node StateGraph: retrieve → generate.
    Uses the same FAISS / BM25 backends as LangChainRAG for a fair comparison.
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

        logger.info(f"[LangGraph] Building graph – method={method}")

        self._embed_model = OllamaEmbeddings(
            model=cfg.embedding_model,
            base_url=cfg.ollama_base_url,
        )
        self._llm = ChatOllama(
            model=cfg.llm_model,
            base_url=cfg.ollama_base_url,
            temperature=cfg.temperature,
        )

        docs = [
            Document(page_content=c["text"], metadata=c["metadata"])
            for c in chunks
        ]

        if method == "bm25":
            self._retriever = BM25Retriever.from_documents(docs, k=cfg.top_k)

        elif method == "manhattan":
            self._vectorstore = FAISS.from_documents(
                docs, self._embed_model,
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
            self._searcher = SimilaritySearcher(chunks, vectors)
            self._retriever = None

        elif method == "hybrid":
            self._vectorstore = FAISS.from_documents(
                docs, self._embed_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            self._bm25_retriever = BM25Retriever.from_documents(docs, k=cfg.top_k * 2)
            self._vector_retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": cfg.top_k * 2}
            )
            self._retriever = None

        else:
            strategy = _DISTANCE_MAP[method]
            self._vectorstore = FAISS.from_documents(
                docs, self._embed_model, distance_strategy=strategy
            )
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": cfg.top_k}
            )

        self._app = self._build_graph()
        logger.info(f"[LangGraph] Graph ready – method={method}")

    # ── graph construction ────────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("retrieve", self._retrieve_node)
        g.add_node("generate", self._generate_node)
        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ── nodes ─────────────────────────────────────────────────────────────

    def _retrieve_node(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        question = state["question"]
        qv_list = state.get("query_vector")
        query_vector = np.array(qv_list, dtype=np.float32) if qv_list else None

        if self.method == "bm25":
            docs = self._retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            retrieved = [
                {"rank": i + 1, "score": None, "text": d.page_content}
                for i, d in enumerate(docs)
            ]

        elif self.method == "manhattan" and query_vector is not None:
            results, _ = self._searcher.search(question, query_vector, "manhattan", self.cfg.top_k)
            context = "\n\n".join(r["text"] for r in results)
            retrieved = results

        elif self.method == "hybrid":
            vec_docs = self._vector_retriever.invoke(question)
            bm25_docs = self._bm25_retriever.invoke(question)
            k = 60
            scores: Dict[str, float] = {}
            texts: Dict[str, str] = {}
            for rank, d in enumerate(vec_docs):
                key = d.page_content[:80]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                texts[key] = d.page_content
            for rank, d in enumerate(bm25_docs):
                key = d.page_content[:80]
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
                texts[key] = d.page_content
            top_keys = sorted(scores, key=scores.get, reverse=True)[: self.cfg.top_k]
            context = "\n\n".join(texts[k] for k in top_keys)
            retrieved = [
                {"rank": i + 1, "score": scores[k], "text": texts[k]}
                for i, k in enumerate(top_keys)
            ]

        else:
            docs = self._retriever.invoke(question)
            context = "\n\n".join(d.page_content for d in docs)
            retrieved = [
                {"rank": i + 1, "score": None, "text": d.page_content}
                for i, d in enumerate(docs)
            ]

        return {
            **state,
            "context": context,
            "retrieved_chunks": retrieved,
            "retrieval_time_s": time.perf_counter() - t0,
        }

    def _generate_node(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        prompt = _ANSWER_PROMPT.format(context=state["context"], question=state["question"])
        response = self._llm.invoke([HumanMessage(content=prompt)])
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            **state,
            "answer": answer,
            "gen_time_s": time.perf_counter() - t0,
        }

    # ── public interface ──────────────────────────────────────────────────

    def query(self, question: str, query_vector: Optional[np.ndarray] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()

        initial_state: RAGState = {
            "question": question,
            "query_vector": query_vector.tolist() if query_vector is not None else None,
            "context": "",
            "retrieved_chunks": [],
            "answer": "",
            "retrieval_time_s": 0.0,
            "gen_time_s": 0.0,
        }

        result = self._app.invoke(initial_state)

        return {
            "framework": "langgraph",
            "method": self.method,
            "question": question,
            "answer": result["answer"],
            "retrieved_chunks": result["retrieved_chunks"],
            "total_time_s": time.perf_counter() - t0,
            "gen_time_s": result["gen_time_s"],
        }
