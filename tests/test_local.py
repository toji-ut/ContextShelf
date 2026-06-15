from __future__ import annotations

import os
import tempfile

from contextshelf.retrieval.sources.local import load_local_folder


def test_load_local_folder_respects_extensions():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "keep.py"), "w", encoding="utf-8") as f:
            f.write("x = 1\n")
        with open(os.path.join(d, "skip.bin"), "wb") as f:
            f.write(b"\x00\x01")
        os.makedirs(os.path.join(d, "node_modules"), exist_ok=True)
        with open(os.path.join(d, "node_modules", "skip.js"), "w") as f:
            f.write("skip")

        docs = load_local_folder(d, extensions_csv=".py")
        rels = {rel for _uri, rel, _c in docs}
        assert "keep.py" in rels
        assert "skip.bin" not in rels
        assert "node_modules/skip.js" not in rels
