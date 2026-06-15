from __future__ import annotations

import tempfile
from unittest.mock import patch

import numpy as np

from contextshelf.retrieval.retriever import retrieve
from contextshelf.retrieval.store import Chunk, VectorStore


def _emb(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(384).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


@patch("contextshelf.retrieval.retriever.encode_query")
def test_retrieve_returns_ranked_chunks(mock_encode):
    mock_encode.return_value = _emb(1)
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(d)
        a = Chunk(id="a", content="JWT auth", source="https://docs/a")
        a.embedding = _emb(1)
        b = Chunk(id="b", content="unrelated css", source="https://docs/b")
        b.embedding = _emb(99)
        store.add([a, b])

        hits = retrieve(store, "authentication", top_k=2, threshold=0.0)
        assert len(hits) >= 1
        assert hits[0].id == "a"


@patch("contextshelf.retrieval.retriever.encode_query")
def test_retrieve_empty_store(mock_encode):
    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(d)
        assert retrieve(store, "query") == []
        mock_encode.assert_not_called()
