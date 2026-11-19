"""Two-stage retrieval: hybrid search for ~60 candidates, cross-encoder rerank to top-k.

A cross-encoder reads (question, chunk) together and outputs a relevance score. Far more
accurate than comparing separate embeddings, but too slow to run over every chunk,
so it only sees the candidates that cheap retrieval already found.
"""

from __future__ import annotations

import os
from functools import lru_cache

import psycopg
from sentence_transformers import CrossEncoder

from analyst.retrieval.hybrid import CANDIDATES, rrf_fuse
from analyst.retrieval.keyword import keyword_search
from analyst.retrieval.vector import Hit, vector_search

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=2)
def _model(name: str) -> CrossEncoder:
    # Inputs longer than max_length tokens (question + chunk) are truncated.
    return CrossEncoder(name, max_length=512)


def candidates(
    conn: psycopg.Connection, query: str, n: int = CANDIDATES, company=None, year=None
) -> list[Hit]:
    """Union of vector top-n and keyword top-n (deduplicated, RRF order)."""
    return rrf_fuse(
        [
            vector_search(conn, query, k=n, company=company, year=year),
            keyword_search(conn, query, k=n, company=company, year=year),
        ]
    )


def rerank(query: str, hits: list[Hit], k: int, model_name: str = RERANK_MODEL) -> list[Hit]:
    if not hits:
        return []
    scores = _model(model_name).predict([(query, h.text) for h in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [Hit(h.id, h.doc_name, h.page, h.text, float(s)) for h, s in ranked[:k]]


def rerank_search(
    conn: psycopg.Connection, query: str, k: int = 6, company=None, year=None
) -> list[Hit]:
    return rerank(query, candidates(conn, query, company=company, year=year), k)
