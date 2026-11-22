"""Fetch a company's latest 10-K from SEC EDGAR and index it.

Run:  uv run python -m analyst.ingest.edgar NVDA

EDGAR is free but asks every client to send a User-Agent with a name and contact email
(set SEC_USER_AGENT in .env) and to stay under 10 requests/second. One ingest makes
3 requests: ticker -> CIK, the company's filing list, the 10-K document itself.

Filings are HTML, not PDF. Filers mark printed page breaks with CSS
(page-break-before/after), so we split on those to keep page-level citations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx

import analyst.config  # noqa: F401  (loads .env)
from analyst.db.store import connect, indexed_docs, init_schema, insert_chunks
from analyst.embeddings import embed_documents
from analyst.ingest.chunker import chunk_pages, split_words

EDGAR_DIR = Path(__file__).resolve().parents[3] / "data" / "edgar"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
FALLBACK_PAGE_WORDS = 600


class EdgarError(Exception):
    pass


# ---- HTTP ----


def _user_agent() -> str:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        raise EdgarError('Set SEC_USER_AGENT in .env, e.g. SEC_USER_AGENT="Jane Doe jane@x.com"')
    return ua


def _get(url: str) -> httpx.Response:
    time.sleep(0.15)  # stay well under SEC's 10 requests/second
    r = httpx.get(url, headers={"User-Agent": _user_agent()}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r


# ---- finding the filing ----


def lookup_cik(ticker: str) -> int:
    for entry in _get(TICKERS_URL).json().values():
        if entry["ticker"].upper() == ticker.upper():
            return int(entry["cik_str"])
    raise EdgarError(f"Unknown ticker: {ticker}")


def pick_latest_10k(submissions: dict) -> dict:
    """The newest 10-K in EDGAR's 'recent filings' table (stored as parallel arrays)."""
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i],
                "report_date": recent["reportDate"][i],  # end of the fiscal year covered
                "document": recent["primaryDocument"][i],
            }
    raise EdgarError(f"No 10-K found for {submissions.get('name')}")


def clean_company_name(name: str) -> str:
    """'NVIDIA CORP' -> 'Nvidia', 'APPLE INC.' -> 'Apple', '3M CO' -> '3M'."""
    name = re.sub(
        r"[,.]?\s+(CORP(ORATION)?|INC|CO|LTD|PLC|HOLDINGS?|COMPANY)\.?$",
        "",
        name.strip(),
        flags=re.IGNORECASE,
    )
    return name.title() if name.isupper() and not name[0].isdigit() else name


# ---- HTML -> pages ----

BLOCK_TAGS = {"p", "div", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"}
VOID_TAGS = {"br", "hr", "img", "meta", "link", "input"}
SKIP_TAGS = {"ix:header", "script", "style", "head"}  # ix:header = hidden XBRL data


class _PageSplitter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: list[list[str]] = [[]]
        self.stack: list[tuple[str, bool]] = []  # (tag, page break after it closes)
        self.skip = 0

    def _new_page(self) -> None:
        if "".join(self.pages[-1]).strip():  # don't create empty pages
            self.pages.append([])

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").lower().replace(" ", "")
        if "page-break-before:always" in style or "break-before:page" in style:
            self._new_page()
        after = "page-break-after:always" in style or "break-after:page" in style
        if tag in VOID_TAGS:
            if tag == "br":
                self.pages[-1].append("\n")
            if after:
                self._new_page()
            return
        if tag in SKIP_TAGS:
            self.skip += 1
        if tag in BLOCK_TAGS:
            self.pages[-1].append("\n")
        elif tag in ("td", "th"):
            self.pages[-1].append(" ")
        self.stack.append((tag, after))

    def handle_endtag(self, tag):
        # Pop back to the matching open tag (tolerates unclosed tags in between).
        if not any(t == tag for t, _ in self.stack):
            return
        while self.stack:
            t, after = self.stack.pop()
            if t in SKIP_TAGS:
                self.skip -= 1
            if t == tag:
                if t in BLOCK_TAGS:
                    self.pages[-1].append("\n")
                if after:
                    self._new_page()
                return

    def handle_data(self, data):
        if not self.skip:
            self.pages[-1].append(data)


def _clean(text: str) -> str:
    text = re.sub("[ \t\xa0\u200b]+", " ", text.replace("\x00", ""))
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def html_pages(html: str) -> list[str]:
    """One string per printed page of the filing."""
    parser = _PageSplitter()
    parser.feed(html)
    parser.close()
    pages = [_clean("".join(p)) for p in parser.pages]
    pages = [p for p in pages if p]
    if len(pages) <= 1:  # the filer marked no page breaks: fall back to fixed-size pages
        text = "\n".join(pages)
        pages = split_words(text, size=FALLBACK_PAGE_WORDS, overlap=0)
    return pages


# ---- ingest ----


def ingest(ticker: str, conn=None) -> dict:
    """Download + index the latest 10-K for `ticker`. Safe to re-run (skips if indexed)."""
    ticker = ticker.strip().upper()
    cik = lookup_cik(ticker)
    submissions = _get(SUBMISSIONS_URL.format(cik=cik)).json()
    filing = pick_latest_10k(submissions)
    company = clean_company_name(submissions["name"])
    year = int(filing["report_date"][:4])
    doc_name = f"{ticker}_{year}_10K"
    summary = {"ticker": ticker, "company": company, "year": year, "doc_name": doc_name}

    own_conn = conn is None
    conn = conn or connect()
    try:
        init_schema(conn)
        if doc_name in indexed_docs(conn):
            return {**summary, "status": "already indexed"}

        path = EDGAR_DIR / f"{doc_name}.htm"
        if not path.exists():
            url = ARCHIVE_URL.format(
                cik=cik, accession=filing["accession"].replace("-", ""), document=filing["document"]
            )
            EDGAR_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_get(url).content)

        pages = html_pages(path.read_text(encoding="utf-8", errors="replace"))
        chunks = chunk_pages(doc_name, pages)
        embeddings = embed_documents([c.text for c in chunks])
        meta = {doc_name: {"company": company, "doc_type": "10k", "doc_year": year}}
        insert_chunks(conn, chunks, embeddings, meta)
        conn.execute("REFRESH MATERIALIZED VIEW lexeme_stats")  # keyword-search word rarity
        conn.commit()
        return {**summary, "status": "indexed", "pages": len(pages), "chunks": len(chunks)}
    finally:
        if own_conn:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="e.g. NVDA")
    args = parser.parse_args()
    print(json.dumps(ingest(args.ticker), indent=2))


if __name__ == "__main__":
    main()
