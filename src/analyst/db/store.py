"""Database access: create schema, insert chunks, count rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from analyst.config import DATABASE_URL
from analyst.ingest.chunker import Chunk

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets us pass numpy arrays as vector values
    return conn


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA_FILE.read_text())
    conn.commit()


def insert_chunks(
    conn: psycopg.Connection,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    meta: dict[str, dict],
) -> None:
    """meta maps doc_name -> {company, doc_type, doc_year}. Re-inserts are ignored."""
    rows = [
        (
            c.doc_name,
            meta.get(c.doc_name, {}).get("company"),
            meta.get(c.doc_name, {}).get("doc_type"),
            meta.get(c.doc_name, {}).get("doc_year"),
            c.page,
            c.chunk_index,
            c.text,
            emb,
        )
        for c, emb in zip(chunks, embeddings, strict=True)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks
                (doc_name, company, doc_type, doc_year, page, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_name, page, chunk_index) DO NOTHING
            """,
            rows,
        )
    conn.commit()


def count_chunks(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT count(*) FROM chunks").fetchone()[0]


def indexed_docs(conn: psycopg.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT DISTINCT doc_name FROM chunks")}
