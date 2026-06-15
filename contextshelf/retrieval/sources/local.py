from __future__ import annotations

import os
from pathlib import Path

from contextshelf.retrieval.indexer_helpers import ALL_INDEX_EXT, SKIP_DIRS


def iter_local_files(
    root: str,
    extensions: frozenset[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return list of (source_uri, rel_path, content)."""
    root = os.path.abspath(root)
    source_uri = root
    exts = extensions or ALL_INDEX_EXT
    docs: list[tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            suf = Path(name).suffix.lower()
            if suf not in exts:
                continue
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError:
                continue
            rel = os.path.relpath(path, root)
            docs.append((source_uri, rel, content))
    return docs


def load_local_folder(root: str, extensions_csv: str | None = None) -> list[tuple[str, str, str]]:
    exts: frozenset[str] | None = None
    if extensions_csv:
        parts = [("." + p.strip().lstrip(".")) if not p.startswith(".") else p.strip() for p in extensions_csv.split(",")]
        exts = frozenset(parts)
    return iter_local_files(root, exts)
