from __future__ import annotations

import json
import os
from typing import Any

REGISTRY_FILE = "sources_registry.json"


def registry_path(index_dir: str) -> str:
    return os.path.join(index_dir, REGISTRY_FILE)


def load_registry(index_dir: str) -> list[dict[str, Any]]:
    path = registry_path(index_dir)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_registry(index_dir: str, entries: list[dict[str, Any]]) -> None:
    os.makedirs(index_dir, exist_ok=True)
    with open(registry_path(index_dir), "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def add_source(index_dir: str, kind: str, target: str, meta: dict[str, Any] | None = None) -> None:
    add_source_unique(index_dir, kind, target, meta)


def remove_registry_target(index_dir: str, target: str) -> None:
    entries = load_registry(index_dir)
    target_norm = target.rstrip("/")
    entries = [
        e
        for e in entries
        if e.get("target", "").rstrip("/") != target_norm
    ]
    save_registry(index_dir, entries)


def add_source_unique(index_dir: str, kind: str, target: str, meta: dict[str, Any] | None = None) -> None:
    entries = load_registry(index_dir)
    target_norm = target.rstrip("/")
    if any(e.get("target", "").rstrip("/") == target_norm for e in entries):
        return
    entries.append({"kind": kind, "target": target, "meta": meta or {}})
    save_registry(index_dir, entries)
