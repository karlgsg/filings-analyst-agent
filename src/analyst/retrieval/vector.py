"""Vector similarity search over the chunks table."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from analyst.embeddings import embed_query


@dataclass
class Hit:
    id: int
    doc_name: str
    page: int
    text: str
    score: float  # cosine similarity, higher is better

    @property
    def label(self) -> str:
        return f"[{self.doc_name}, p.{self.page}]"


def vector_search(conn: psycopg.Connection, query: str, k: int = 6) -> list[Hit]:
    """Top-k chunks by cosine similarity, searched across *all* documents."""
    rows = conn.execute(
        """
        SELECT id, doc_name, page, text, 1 - (embedding <=> %s) AS score
        FROM chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (q := embed_query(query), q, k),
    ).fetchall()
    return [Hit(*row) for row in rows]
