from __future__ import annotations

from contextshelf.retrieval.embed_model import encode_query
from contextshelf.retrieval.store import Chunk, VectorStore

DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.35


def retrieve(
    store: VectorStore,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Chunk]:
    """Embed query and return ranked chunks above the similarity threshold."""
    if store.size() == 0:
        return []
    q_emb = encode_query(query)
    hits = store.search(q_emb, top_k=top_k, threshold=threshold)
    return [ch for ch, _score in hits]
