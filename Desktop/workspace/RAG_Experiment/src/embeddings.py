from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from loguru import logger
from tqdm import tqdm


class EmbeddingModel:
    """Ollama-hosted embedding model with optional on-disk cache."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        cache_dir: Optional[Path] = None,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        test_vec = self._call_api(["test"])
        self.dimension = len(test_vec[0])
        logger.info(f"Embedding model: {model_name} via Ollama, dimension: {self.dimension}")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model_name, "input": texts},
                    timeout=120,
                )
                resp.raise_for_status()
                return resp.json()["embeddings"]
            except Exception as exc:
                last_exc = exc
                wait = min(2 ** attempt, 8)
                logger.warning(
                    f"Embedding request failed (attempt {attempt}/{self.max_retries}): {exc}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _cache_key(self, texts: List[str], normalize: bool) -> str:
        payload = {
            "model": self.model_name,
            "normalize": normalize,
            "texts": texts,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest

    def _cache_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        return self.cache_dir / f"{self.model_name.replace(':', '_')}__{key}.npy"

    def embed(self, texts: List[str], batch_size: int = 32, normalize: bool = False) -> np.ndarray:
        if self.cache_dir and texts:
            key = self._cache_key(texts, normalize)
            path = self._cache_path(key)
            if path.exists():
                arr = np.load(path)
                logger.info(f"Loaded cached embeddings from {path.name} – shape {arr.shape}")
                return arr

        all_vecs: List[List[float]] = []
        batches = range(0, len(texts), batch_size)
        for i in tqdm(batches, desc="Embedding", disable=len(texts) <= batch_size):
            all_vecs.extend(self._call_api(texts[i : i + batch_size]))
        arr = np.array(all_vecs, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)

        if self.cache_dir and texts:
            path = self._cache_path(self._cache_key(texts, normalize))
            np.save(path, arr)
            logger.info(f"Cached embeddings → {path.name}")

        return arr

    def embed_query(self, text: str, normalize: bool = False) -> np.ndarray:
        # Queries are short and unique; skip disk cache to avoid clutter.
        return self.embed([text], normalize=normalize)[0]
