"""Keyword (full-text) search with Postgres tsvector."""

from __future__ import annotations

import re

import psycopg

from analyst.retrieval.vector import Hit

# Words that describe *how* to answer rather than *what* to find. In an OR query they
# match boilerplate everywhere and drown out the content words.
INSTRUCTION_WORDS = {
    "answer", "answers", "question", "questions", "round", "rounded", "decimal", "decimals",
    "place", "places", "unit", "units", "percent", "percents", "please", "assume",
    "analyst", "analysts", "equity", "research", "perspective", "according", "agreement",
    "information", "details", "detail", "outlined", "shown", "clearly", "using", "use",
    "based", "relying", "provided", "give", "response", "address", "adopting", "primarily",
    "calculate", "calculated", "what", "which", "how", "does", "did", "is", "are", "was",
    "were", "the", "of", "in", "for", "and", "to", "a", "an", "as", "if", "you", "your",
    "have", "has", "lost", "internet", "connection", "access", "financial", "metric",
    "providers", "statement", "statements", "within", "only", "can", "who", "would", "like",
}  # fmt: skip


def clean_query(question: str) -> str:
    """'What is the FY2018 capex for 3M? Round to one decimal place.' -> 'FY2018 2018 capex 3M one'.

    Filings write years as '2018' or 'fiscal 2018', almost never 'FY2018', so add the bare year.
    """
    text = re.sub(r"\bFY\s?'?(\d{2,4})\b", lambda m: f"FY{m.group(1)} {_full_year(m.group(1))}",
                  question, flags=re.IGNORECASE)  # fmt: skip
    words = re.findall(r"[A-Za-z0-9&.\-]+", text)
    return " ".join(w for w in words if w.lower().strip(".") not in INSTRUCTION_WORDS)


def _full_year(y: str) -> str:
    return y if len(y) == 4 else f"20{y}"


def keyword_search(conn: psycopg.Connection, query: str, k: int = 6) -> list[Hit]:
    """Top-k chunks by IDF-weighted keyword match.

    Each query word is weighted by idf = ln(total chunks / chunks containing it), so rare,
    specific words ("bestbuy", "2015") count far more than common ones ("net", "year").
    A chunk's score is the sum of the idf of the distinct query words it contains.
    (Like BM25 without the term-frequency and length parts; Postgres's built-in
    ts_rank has no idf at all, which made plain keyword search nearly useless here.)
    """
    rows = conn.execute(
        """
        WITH terms AS (
            SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %s))
        ),
        weighted AS (
            SELECT t.lexeme, ln((SELECT count(*) FROM chunks)::float / s.ndoc) AS idf
            FROM terms t JOIN lexeme_stats s ON s.word = t.lexeme
        )
        SELECT c.id, c.doc_name, c.page, c.text, sum(w.idf) AS score
        FROM weighted w
        JOIN chunks c ON c.tsv @@ quote_literal(w.lexeme)::tsquery
        GROUP BY c.id
        ORDER BY score DESC
        LIMIT %s
        """,
        (clean_query(query), k),
    ).fetchall()
    return [Hit(*row) for row in rows]
