from __future__ import annotations

from unittest.mock import patch

import numpy as np

from contextshelf.retrieval.indexer import index_docs_into_store
from contextshelf.retrieval.indexer_helpers import (
    chunk_code,
    chunk_from_text,
    chunk_markdown,
    estimate_tokens,
    texts_to_chunks,
)
from contextshelf.retrieval.store import Chunk


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_chunk_code_splits_functions():
    src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
    chunks = chunk_code("auth.py", src, "py")
    symbols = {c["meta"].get("symbol") for c in chunks}
    assert "foo" in symbols
    assert "bar" in symbols


def test_chunk_markdown_splits_headings():
    md = "# Intro\n\nHello\n\n## Details\n\nMore text\n"
    chunks = chunk_markdown("readme.md", md)
    headings = {c["meta"].get("heading") for c in chunks}
    assert "Intro" in headings
    assert "Details" in headings


def test_chunk_from_text_picks_strategy():
    py_chunks = chunk_from_text("x.py", "def a(): pass", ".py")
    assert py_chunks[0]["meta"]["kind"] == "code"
    md_chunks = chunk_from_text("x.md", "# Hi\n", ".md")
    assert md_chunks[0]["meta"]["kind"] == "markdown"


def test_texts_to_chunks_builds_ids():
    chunks = texts_to_chunks(
        "/root",
        [{"content": "hello", "meta": {"kind": "text", "path": "a.txt"}}],
    )
    assert len(chunks) == 1
    assert isinstance(chunks[0], Chunk)
    assert chunks[0].source == "/root"


@patch("contextshelf.retrieval.indexer_helpers.encode_texts")
def test_index_docs_into_store_mocks_embed(mock_encode):
    mock_encode.return_value = np.ones((1, 384), dtype=np.float32)
    docs = [("/proj", "main.py", "def run():\n    pass\n")]
    out = index_docs_into_store(docs, batch_size=8)
    assert len(out) >= 1
    assert out[0].embedding is not None
    assert len(out[0].embedding) == 384
