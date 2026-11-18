"""Local text embeddings with bge-small-en-v1.5 (384 dims, free, runs on CPU/Apple GPU)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
# bge models expect this prefix on *queries* (not on documents) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    # normalize=True -> unit-length vectors, so cosine similarity == dot product.
    return _model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )


def embed_query(query: str) -> np.ndarray:
    return _model().encode(QUERY_PREFIX + query, normalize_embeddings=True)
