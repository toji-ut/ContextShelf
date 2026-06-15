from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: Any = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s …", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def encode_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    model = get_embedding_model()
    kwargs: dict[str, Any] = {"batch_size": batch_size, "convert_to_numpy": True}
    if show_progress:
        kwargs["show_progress_bar"] = True
    emb = model.encode(texts, **kwargs)
    return np.asarray(emb, dtype=np.float32)


def encode_query(text: str) -> list[float]:
    arr = encode_texts([text], batch_size=1)
    return arr[0].tolist()
