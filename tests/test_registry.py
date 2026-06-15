from __future__ import annotations

import tempfile

from contextshelf.retrieval.registry import (
    add_source,
    add_source_unique,
    load_registry,
    remove_registry_target,
    save_registry,
)


def test_registry_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        add_source(d, "local", "/projects/foo")
        reg = load_registry(d)
        assert len(reg) == 1
        assert reg[0]["kind"] == "local"
        assert reg[0]["target"] == "/projects/foo"


def test_add_source_unique_dedupes():
    with tempfile.TemporaryDirectory() as d:
        add_source_unique(d, "agent_docs", "https://example.com/docs")
        add_source_unique(d, "agent_docs", "https://example.com/docs")
        assert len(load_registry(d)) == 1


def test_remove_registry_target():
    with tempfile.TemporaryDirectory() as d:
        save_registry(
            d,
            [
                {"kind": "local", "target": "/a", "meta": {}},
                {"kind": "local", "target": "/b", "meta": {}},
            ],
        )
        remove_registry_target(d, "/a")
        targets = [e["target"] for e in load_registry(d)]
        assert targets == ["/b"]
