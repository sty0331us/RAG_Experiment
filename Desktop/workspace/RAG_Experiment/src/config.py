from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


@dataclass
class ExperimentConfig:
    # ── LLM (Ollama only – no API key needed) ────────────────────────────
    llm_provider: str = "ollama"
    llm_model: str = "gemma4:26b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    # ── Embedding ────────────────────────────────────────────────────────
    # Ollama-hosted embedding model (runs fully local, no API key needed)
    # Alternatives via Ollama: "mxbai-embed-large" (1024-dim), "all-minilm" (384-dim)
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ── Retrieval ────────────────────────────────────────────────────────
    top_k: int = 5

    # ── Similarity methods to benchmark ──────────────────────────────────
    similarity_methods: List[str] = field(default_factory=lambda: [
        "cosine",        # FAISS IndexFlatIP + L2-normalised vectors
        "euclidean",     # FAISS IndexFlatL2
        "dot_product",   # FAISS IndexFlatIP (raw, no normalisation)
        "manhattan",     # sklearn pairwise L1
        "bm25",          # sparse BM25Okapi
        "hybrid",        # RRF fusion: BM25 + cosine
    ])

    # ── Frameworks to benchmark ───────────────────────────────────────────
    frameworks: List[str] = field(default_factory=lambda: ["llamaindex", "langchain"])

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: Path = BASE_DIR / "data" / "pdfs"
    results_dir: Path = BASE_DIR / "results"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        (self.results_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(parents=True, exist_ok=True)

        # Allow env overrides
        if os.getenv("OLLAMA_BASE_URL"):
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if os.getenv("LLM_MODEL"):
            self.llm_model = os.getenv("LLM_MODEL")
