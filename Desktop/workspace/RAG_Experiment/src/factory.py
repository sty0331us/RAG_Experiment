"""Factory helpers for constructing RAG pipelines by framework name."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .config import ExperimentConfig


def build_rag(
    framework: str,
    chunks: List[Dict[str, Any]],
    vectors: np.ndarray,
    cfg: ExperimentConfig,
    method: str,
):
    """Return a framework-specific RAG instance for *method*."""
    name = framework.lower().strip()
    if name == "llamaindex":
        from .llamaindex_rag import LlamaIndexRAG
        return LlamaIndexRAG(chunks, vectors, cfg, method)
    if name == "haystack":
        from .haystack_rag import HaystackRAG
        return HaystackRAG(chunks, vectors, cfg, method)
    if name == "langchain":
        from .langchain_rag import LangChainRAG
        return LangChainRAG(chunks, vectors, cfg, method)
    raise ValueError(
        f"Unknown framework '{framework}'. Expected one of: llamaindex, langchain, haystack"
    )
