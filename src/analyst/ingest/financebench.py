"""Download the FinanceBench open-source set (questions + source PDFs).

We only fetch PDFs for the top-N companies by question count, to keep the
download small while we build. Bump N later for the full 150-question run.

Run:  uv run python -m analyst.ingest.financebench            # download
      uv run python -m analyst.ingest.financebench --show 3    # print samples
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "financebench"
QUESTIONS_FILE = DATA_DIR / "financebench_open_source.jsonl"
DOCS_FILE = DATA_DIR / "financebench_document_information.jsonl"
PDF_DIR = DATA_DIR / "pdfs"
DEFAULT_TOP_N = 10


def _download(url: str, dest: Path) -> None:
    """Download url to dest, skipping files we already have."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)  # only "complete" once fully written


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        return [json.loads(line) for line in f]


def top_companies(questions: list[dict], n: int) -> list[str]:
    counts = Counter(q["company"] for q in questions)
    return [company for company, _ in counts.most_common(n)]


def selected_questions(n: int = DEFAULT_TOP_N) -> list[dict]:
    """Questions whose company is in the top-n set (the ones we have PDFs for)."""
    questions = load_questions()
    companies = set(top_companies(questions, n))
    return [q for q in questions if q["company"] in companies]


def download(n: int = DEFAULT_TOP_N) -> None:
    _download(f"{REPO_RAW}/data/financebench_open_source.jsonl", QUESTIONS_FILE)
    _download(f"{REPO_RAW}/data/financebench_document_information.jsonl", DOCS_FILE)

    questions = selected_questions(n)
    doc_names = sorted({q["doc_name"] for q in questions})
    print(f"{len(questions)} questions across {len(doc_names)} documents")

    for i, doc in enumerate(doc_names, 1):
        dest = PDF_DIR / f"{doc}.pdf"
        status = "cached" if dest.exists() else "downloading"
        print(f"[{i}/{len(doc_names)}] {doc} ({status})")
        if status == "downloading":
            _download(f"{REPO_RAW}/pdfs/{doc}.pdf", dest)
            time.sleep(0.2)  # be polite to GitHub


def show(k: int, n: int = DEFAULT_TOP_N) -> None:
    for q in selected_questions(n)[:k]:
        pages = [e["evidence_page_num"] for e in q["evidence"]]
        print(f"\nQ: {q['question']}")
        print(f"A: {q['answer']}")
        print(f"Source: {q['doc_name']}  evidence pages: {pages}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help="number of companies")
    parser.add_argument("--show", type=int, help="print N sample questions instead")
    args = parser.parse_args()
    if args.show:
        show(args.show, args.top)
    else:
        download(args.top)
