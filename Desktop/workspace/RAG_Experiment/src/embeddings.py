from __future__ import annotations

from typing import List

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm


class EmbeddingModel:
    """Ollama-hosted embedding model (nomic-embed-text or any ollama pull'd model)."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        test_vec = self._call_api(["test"])
        self.dimension = len(test_vec[0])
        logger.info(f"Embedding model: {model_name} via Ollama, dimension: {self.dimension}")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def embed(self, texts: List[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        all_vecs: List[List[float]] = []
        batches = range(0, len(texts), batch_size)
        for i in tqdm(batches, desc="Embedding", disable=len(texts) <= batch_size):
            all_vecs.extend(self._call_api(texts[i : i + batch_size]))
        arr = np.array(all_vecs, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)
        return arr

    def embed_query(self, text: str, normalize: bool = False) -> np.ndarray:
        return self.embed([text], normalize=normalize)[0]
