from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class CrisisRAG:
    """FAISS-backed retrieval memory for crisis events."""

    def __init__(self, dim: int = 768, top_k: int = 30):
        self.dim = dim
        self.top_k = top_k
        self.index = faiss.IndexFlatIP(dim)
        self.meta: list[dict] = []

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-8, None)

    def add(self, embedding, metadata: dict) -> None:
        vec = self._normalize(embedding)
        self.index.add(vec.astype(np.float32))
        self.meta.append(metadata)

    def query(self, embedding, k: int | None = None):
        k = min(k or self.top_k, len(self.meta))
        if k <= 0:
            return [], []
        sims, idxs = self.index.search(self._normalize(embedding).astype(np.float32), k)
        sims = sims[0]
        idxs = idxs[0]
        results = [self.meta[i] for i in idxs if i >= 0]
        return results, sims[: len(results)]

    def rag_score(self, embedding) -> float:
        results, sims = self.query(embedding)
        if not results:
            return 50.0

        scores = np.array([float(r.get("gpi", r.get("score", 50.0))) for r in results], dtype=np.float32)
        weights = np.clip(np.asarray(sims, dtype=np.float32), 0.0, None)
        if np.allclose(weights.sum(), 0.0):
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        return float(np.dot(scores, weights))

    def save(self, path: str = "rag_index") -> None:
        base = Path(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{base}.faiss")
        with open(f"{base}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, ensure_ascii=False, indent=2)

    def load(self, path: str = "rag_index") -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.meta.json", "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
