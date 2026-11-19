"""Hybrid search: vector + keyword, merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import psycopg

from analyst.retrieval.keyword import keyword_search
from analyst.retrieval.vector import Hit, vector_search

RRF_K = 60  # standard constant from the original RRF paper; dampens the top ranks
CANDIDATES = 30  # how many results to pull from each retriever before fusing


def rrf_fuse(ranked_lists: list[list[Hit]], k: int = RRF_K) -> list[Hit]:
    """Score each chunk by sum(1 / (k + rank)) across lists. Uses ranks only,
    so it doesn't matter that cosine similarity and ts_rank are on different scales."""
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranked in ranked_lists:
        for rank, hit in enumerate(ranked, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(hit.id, hit)
    fused = sorted(scores, key=scores.get, reverse=True)
    return [Hit(hits[i].id, hits[i].doc_name, hits[i].page, hits[i].text, scores[i]) for i in fused]


def hybrid_search(
    conn: psycopg.Connection, query: str, k: int = 6, company=None, year=None
) -> list[Hit]:
    vec = vector_search(conn, query, k=CANDIDATES, company=company, year=year)
    kw = keyword_search(conn, query, k=CANDIDATES, company=company, year=year)
    return rrf_fuse([vec, kw])[:k]
