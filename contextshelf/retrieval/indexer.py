from __future__ import annotations

from contextshelf.retrieval.indexer_helpers import embed_chunks

__all__ = ["embed_chunks", "index_docs_into_store"]


def index_docs_into_store(
    docs: list[tuple[str, str, str]],
    *,
    batch_size: int = 32,
    progress_callback=None,
) -> list:
    """Chunk + embed documents; returns Chunk objects (caller adds to VectorStore)."""
    from contextshelf.retrieval.indexer_helpers import chunk_from_text, texts_to_chunks

    all_chunks = []
    for uri, path_suffix, content in docs:
        full_path = f"{uri}:{path_suffix}" if path_suffix else uri
        ext = ""
        if "." in path_suffix:
            ext = "." + path_suffix.rsplit(".", 1)[-1]
        cds = chunk_from_text(full_path, content, ext)
        all_chunks.extend(texts_to_chunks(uri, cds))

    def _prog(done: int, total: int) -> None:
        if progress_callback:
            progress_callback(done, total)

    embed_chunks(all_chunks, batch_size=batch_size, progress_fn=_prog if progress_callback else None)
    return all_chunks
