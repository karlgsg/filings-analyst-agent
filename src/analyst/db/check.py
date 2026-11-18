"""Smoke test: connect to Postgres and confirm pgvector is available.

Run: uv run python -m analyst.db.check
"""

import psycopg

from analyst.config import DATABASE_URL


def main() -> None:
    with psycopg.connect(DATABASE_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pg_version = conn.execute("SELECT version()").fetchone()[0]
        vec_version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]
    print(f"Connected: {pg_version.split(',')[0]}")
    print(f"pgvector:  {vec_version}")


if __name__ == "__main__":
    main()
