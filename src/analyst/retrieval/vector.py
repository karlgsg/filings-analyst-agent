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
    score: float  # higher is better (meaning depends on the retriever)

    @property
    def label(self) -> str:
        return f"[{self.doc_name}, p.{self.page}]"


# Optional filters shared by all retrievers. NULL means "don't filter".
FILTER_SQL = """
    (%(company)s::text IS NULL OR company = %(company)s)
    AND (%(year)s::int IS NULL OR doc_year = %(year)s)
"""


def vector_search(
    conn: psycopg.Connection,
    query: str,
    k: int = 6,
    company: str | None = None,
    year: int | None = None,
) -> list[Hit]:
    """Top-k chunks by cosine similarity, optionally filtered by company / filing year."""
    rows = conn.execute(
        f"""
        SELECT id, doc_name, page, text, 1 - (embedding <=> %(q)s) AS score
        FROM chunks
        WHERE {FILTER_SQL}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": embed_query(query), "k": k, "company": company, "year": year},
    ).fetchall()
    return [Hit(*row) for row in rows]
