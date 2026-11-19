"""Chunk every downloaded PDF, embed the chunks, and store them in Postgres.

Run:  uv run python -m analyst.ingest.index
Safe to re-run: documents already in the database are skipped.
"""

from __future__ import annotations

import json

from analyst.db.store import connect, count_chunks, indexed_docs, init_schema, insert_chunks
from analyst.embeddings import embed_documents
from analyst.ingest.financebench import DOCS_FILE, PDF_DIR
from analyst.ingest.pdf_parser import chunk_pdf


def load_doc_meta() -> dict[str, dict]:
    meta = {}
    with open(DOCS_FILE) as f:
        for line in f:
            d = json.loads(line)
            meta[d["doc_name"]] = {
                "company": d["company"],
                "doc_type": d["doc_type"],
                "doc_year": d["doc_period"],
            }
    return meta


def main() -> None:
    meta = load_doc_meta()
    with connect() as conn:
        init_schema(conn)
        done = indexed_docs(conn)
        pdfs = [p for p in sorted(PDF_DIR.glob("*.pdf")) if p.stem not in done]
        print(f"{len(done)} docs already indexed, {len(pdfs)} to go")

        expected = 0
        for i, pdf_path in enumerate(pdfs, 1):
            chunks = chunk_pdf(pdf_path)
            print(f"[{i}/{len(pdfs)}] {pdf_path.stem}: {len(chunks)} chunks")
            embeddings = embed_documents([c.text for c in chunks])
            insert_chunks(conn, chunks, embeddings, meta)
            expected += len(chunks)

        # Keyword search weights words by rarity; recount after adding documents.
        conn.execute("REFRESH MATERIALIZED VIEW lexeme_stats")
        conn.commit()

        print(f"\nInserted this run: {expected} chunks")
        print(f"Rows in chunks table: {count_chunks(conn)}")


if __name__ == "__main__":
    main()
