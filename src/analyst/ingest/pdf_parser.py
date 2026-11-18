"""PDF -> page texts -> chunks, keeping doc name + page number on every chunk.

Run:  uv run python -m analyst.ingest.pdf_parser            # stats for all PDFs
      uv run python -m analyst.ingest.pdf_parser 3M_2018_10K --page 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf

from analyst.ingest.chunker import Chunk, chunk_pages
from analyst.ingest.financebench import PDF_DIR


def read_pages(pdf_path: Path) -> list[str]:
    """Return one string per page. Note: tables come out as loose lines of text."""
    with pymupdf.open(pdf_path) as doc:
        # Some PDFs contain NUL bytes, which Postgres text columns reject.
        return [page.get_text().replace("\x00", "") for page in doc]


def chunk_pdf(pdf_path: Path) -> list[Chunk]:
    return chunk_pages(pdf_path.stem, read_pages(pdf_path))


def chunk_all(pdf_dir: Path = PDF_DIR) -> list[Chunk]:
    chunks = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf_path))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("doc", nargs="?", help="doc name, e.g. 3M_2018_10K")
    parser.add_argument("--page", type=int, help="print chunks from this page (1-indexed)")
    args = parser.parse_args()

    if args.doc:
        chunks = chunk_pdf(PDF_DIR / f"{args.doc}.pdf")
        shown = [c for c in chunks if args.page is None or c.page == args.page]
        for c in shown[:3]:
            print(f"--- [{c.doc_name}, p.{c.page}, chunk {c.chunk_index}] ---")
            print(c.text[:600], "...\n")
        print(f"{len(chunks)} chunks in {args.doc}")
        return

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    total = 0
    for pdf_path in pdfs:
        n = len(chunk_pdf(pdf_path))
        total += n
        print(f"{pdf_path.stem:45s} {n:5d} chunks")
    print(f"\nTotal: {total} chunks from {len(pdfs)} PDFs")


if __name__ == "__main__":
    main()
