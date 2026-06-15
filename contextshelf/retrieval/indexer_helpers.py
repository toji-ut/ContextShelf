from __future__ import annotations

import re
from typing import Any

from contextshelf.retrieval.embed_model import encode_texts
from contextshelf.retrieval.store import Chunk, chunk_id

CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".yaml", ".yml", ".json", ".toml"}
DOC_EXT = {".md", ".mdx", ".rst"}
TEXT_EXT = {".txt"}
ALL_INDEX_EXT = CODE_EXT | DOC_EXT | TEXT_EXT | {".htm", ".html"}

SKIP_DIRS = {".git", "node_modules", "__pycache__", "dist", "build", ".venv", "venv", ".contextshelf"}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _split_tokens_budget(text: str, max_tokens: int, overlap: int = 0) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]
    approx_chars = max_tokens * 4
    overlap_chars = overlap * 4
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + approx_chars)
        chunk = text[start:end]
        parts.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap_chars if overlap_chars < end else end
    return [p for p in parts if p]


def chunk_code(source_path: str, content: str, lang: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"(?m)^(?:async\s+def|def|class)\s+\w+")
    boundaries = [m.start() for m in pattern.finditer(content)]
    if not boundaries:
        return [
            {
                "content": content.strip(),
                "meta": {"kind": "code", "path": source_path, "language": lang},
            }
        ]
    boundaries.append(len(content))
    chunks: list[dict[str, Any]] = []
    for i, start in enumerate(boundaries[:-1]):
        end = boundaries[i + 1]
        block = content[start:end].strip()
        if not block:
            continue
        header_m = re.match(r"^(?:(?:async\s+)?def|class)\s+(\w+)", block, re.M)
        name = header_m.group(1) if header_m else "block"
        if estimate_tokens(block) > 512:
            for piece in _split_tokens_budget(block, 512):
                chunks.append(
                    {
                        "content": piece,
                        "meta": {
                            "kind": "code",
                            "path": source_path,
                            "language": lang,
                            "symbol": name,
                        },
                    }
                )
        else:
            chunks.append(
                {
                    "content": block,
                    "meta": {"kind": "code", "path": source_path, "language": lang, "symbol": name},
                }
            )
    return chunks


def chunk_markdown(source_path: str, content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    sections: list[tuple[str, int, str]] = []
    current: list[str] = []
    heading_depth = 0
    current_heading = ""
    for line in lines:
        if re.match(r"^#{1,6}\s+", line):
            if current:
                sections.append((current_heading, heading_depth, "\n".join(current)))
            current_heading = line.lstrip("#").strip()
            heading_depth = len(line) - len(line.lstrip("#"))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append((current_heading, heading_depth, "\n".join(current)))

    if not sections:
        sections = [("", 0, content)]

    out: list[dict[str, Any]] = []
    for heading, depth, sec_text in sections:
        sec_text = sec_text.strip()
        if not sec_text:
            continue
        if estimate_tokens(sec_text) > 512:
            for piece in _split_tokens_budget(sec_text, 512):
                out.append(
                    {
                        "content": piece,
                        "meta": {
                            "kind": "markdown",
                            "path": source_path,
                            "heading": heading,
                            "depth": depth,
                        },
                    }
                )
        else:
            out.append(
                {
                    "content": sec_text,
                    "meta": {
                        "kind": "markdown",
                        "path": source_path,
                        "heading": heading,
                        "depth": depth,
                    },
                }
            )
    return out


def chunk_plain(source_path: str, content: str) -> list[dict[str, Any]]:
    paras = re.split(r"\n\s*\n+", content)
    out: list[dict[str, Any]] = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        cand = (buf + "\n\n" + p).strip() if buf else p
        if estimate_tokens(cand) <= 256:
            buf = cand
        else:
            if buf:
                out.append({"content": buf, "meta": {"kind": "text", "path": source_path}})
            if estimate_tokens(p) > 256:
                for piece in _split_tokens_budget(p, 256, overlap=50):
                    out.append({"content": piece, "meta": {"kind": "text", "path": source_path}})
                buf = ""
            else:
                buf = p
    if buf:
        out.append({"content": buf, "meta": {"kind": "text", "path": source_path}})
    return out


def chunk_from_text(source_path: str, content: str, ext: str) -> list[dict[str, Any]]:
    ext_l = ext.lower()
    if ext_l in CODE_EXT:
        lang = ext_l.lstrip(".")
        return chunk_code(source_path, content, lang)
    if ext_l in DOC_EXT:
        return chunk_markdown(source_path, content)
    return chunk_plain(source_path, content)


def texts_to_chunks(source_uri: str, chunk_dicts: list[dict[str, Any]]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in chunk_dicts:
        content = d["content"]
        meta = d["meta"]
        cid = chunk_id(content, f"{source_uri}:{meta}")
        out.append(Chunk(id=cid, content=content, source=source_uri, metadata=meta))
    return out


def embed_chunks(chunks: list[Chunk], batch_size: int = 32, progress_fn=None) -> None:
    if not chunks:
        return
    texts = [c.content for c in chunks]
    total = len(texts)
    import numpy as np

    all_batches = []
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        all_batches.append(encode_texts(batch, batch_size=len(batch), show_progress=False))
        if progress_fn:
            progress_fn(min(start + batch_size, total), total)
    merged = np.vstack(all_batches)
    for i, ch in enumerate(chunks):
        ch.embedding = merged[i].tolist()
