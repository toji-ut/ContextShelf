from __future__ import annotations

import tempfile

import numpy as np
import pytest

from contextshelf.retrieval.store import Chunk, VectorStore, chunk_id


def _fake_chunk(cid: str, content: str, source: str = "test") -> Chunk:
    return Chunk(id=cid, content=content, source=source)


def _random_embedding(dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed=42)
    v = rng.random(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _chunk_with_embedding(cid: str, content: str, embedding: list[float] | None = None) -> Chunk:
    ch = _fake_chunk(cid, content)
    ch.embedding = embedding or _random_embedding()
    return ch


class TestChunkId:
    def test_deterministic(self):
        assert chunk_id("hello", "src") == chunk_id("hello", "src")

    def test_different_content(self):
        assert chunk_id("hello", "src") != chunk_id("world", "src")

    def test_different_source(self):
        assert chunk_id("hello", "src1") != chunk_id("hello", "src2")

    def test_length_16(self):
        assert len(chunk_id("x", "y")) == 16


class TestVectorStore:
    def test_empty_store_size_zero(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            assert store.size() == 0

    def test_add_increases_size(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            chunks = [_chunk_with_embedding(f"c{i}", f"content {i}") for i in range(5)]
            store.add(chunks)
            assert store.size() == 5

    def test_search_returns_relevant(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            target_emb = np.zeros(384, dtype=np.float32)
            target_emb[0] = 1.0
            target_emb = (target_emb / np.linalg.norm(target_emb)).tolist()

            similar = _chunk_with_embedding("similar", "auth JWT", target_emb)
            noise = _chunk_with_embedding("noise", "unrelated css", _random_embedding())
            store.add([similar, noise])

            results = store.search(target_emb, top_k=1, threshold=0.0)
            assert results[0][0].id == "similar"

    def test_search_threshold_filters(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            emb = _random_embedding()
            chunk = _chunk_with_embedding("c1", "content", emb)
            store.add([chunk])
            results = store.search(emb, top_k=5, threshold=0.9999)
            assert len(results) == 1

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            chunks = [_chunk_with_embedding(f"c{i}", f"text {i}") for i in range(3)]
            store.add(chunks)
            store.save()

            store2 = VectorStore(d)
            store2.load()
            assert store2.size() == 3

    def test_load_missing_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            store.load()
            assert store.size() == 0

    def test_remove_source_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            keep = _chunk_with_embedding("keep", "keep me", _random_embedding())
            keep.source = "/projects/other"
            drop = _chunk_with_embedding("drop", "drop me", _random_embedding())
            drop.source = "/projects/mine"
            store.add([keep, drop])
            removed = store.remove_source_prefix("/projects/mine")
            assert removed == 1
            assert store.size() == 1

    def test_add_requires_384_dim(self):
        with tempfile.TemporaryDirectory() as d:
            store = VectorStore(d)
            bad = _fake_chunk("bad", "content")
            bad.embedding = [0.1, 0.2]
            with pytest.raises(ValueError):
                store.add([bad])
