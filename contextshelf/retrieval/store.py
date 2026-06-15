from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

SOURCES_REGISTRY = "sources_registry.json"
DEFAULT_INDEX_PATH = ".contextshelf/index"


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    indexed_at: float = field(default_factory=time.time)
    embedding: list[float] | None = None

    def to_storage_dict(self) -> dict[str, Any]:
        d = asdict(self)
        del d["embedding"]
        return d


class VectorStore:
    """Cosine similarity search over numpy-backed embeddings."""

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self._emb_path = os.path.join(index_dir, "embeddings.npy")
        self._meta_path = os.path.join(index_dir, "chunks_meta.json")
        self.embeddings: np.ndarray = np.zeros((0, 384), dtype=np.float32)
        self.chunks: list[dict[str, Any]] = []

    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new_emb = np.array([c.embedding or [] for c in chunks], dtype=np.float32)
        if new_emb.ndim != 2 or new_emb.shape[1] != 384:
            raise ValueError("Each chunk needs a 384-dim embedding")
        rows = np.linalg.norm(new_emb, axis=1, keepdims=True)
        rows = np.clip(rows, 1e-12, None)
        new_emb = new_emb / rows
        for c in chunks:
            self.chunks.append(c.to_storage_dict())
        self.embeddings = np.vstack([self.embeddings, new_emb]) if len(self.embeddings) else new_emb

    def search(
        self, query_embedding: list[float], top_k: int, threshold: float
    ) -> list[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        q = q / np.clip(np.linalg.norm(q), 1e-12, None)
        sims = (self.embeddings @ q.T).flatten()
        idx = np.argsort(-sims)
        out: list[tuple[Chunk, float]] = []
        for i in idx[:top_k]:
            score = float(sims[i])
            if score < threshold:
                continue
            meta = self.chunks[int(i)]
            ch = Chunk(
                id=meta["id"],
                content=meta["content"],
                source=meta["source"],
                metadata=meta.get("metadata", {}),
                indexed_at=meta.get("indexed_at", 0.0),
            )
            out.append((ch, score))
        return out

    def save(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(self._emb_path, self.embeddings)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f)

    def load(self) -> None:
        if not os.path.isfile(self._emb_path) or not os.path.isfile(self._meta_path):
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.chunks = []
            return
        self.embeddings = np.load(self._emb_path).astype(np.float32)
        with open(self._meta_path, encoding="utf-8") as f:
            self.chunks = json.load(f)
        if len(self.embeddings) != len(self.chunks):
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
            self.chunks = []

    def remove_source_prefix(self, source_prefix: str) -> int:
        """Remove chunks whose source equals or starts with prefix."""
        prefix = source_prefix.rstrip("/")
        keep_idx: list[int] = []
        removed = 0
        for i, ch in enumerate(self.chunks):
            src = ch.get("source", "")
            if src == prefix or src.startswith(prefix + "/"):
                removed += 1
                continue
            keep_idx.append(i)
        if removed == 0:
            return 0
        self.chunks = [self.chunks[i] for i in keep_idx]
        if keep_idx:
            self.embeddings = self.embeddings[np.array(keep_idx, dtype=np.int64)]
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)
        return removed


def chunk_id(content: str, source: str) -> str:
    return hashlib.sha256(f"{source}:{content}".encode()).hexdigest()[:16]
