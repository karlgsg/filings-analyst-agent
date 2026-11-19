"""Baseline RAG: vector search -> stuff top-k chunks into the prompt -> cited answer.

Deliberately simple. This is the number Phase 2 has to beat.

Run:  uv run python -m analyst.agent.baseline "What was 3M's FY2018 capex?"
      uv run python -m analyst.agent.baseline --samples 5
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import anthropic

from analyst.agent.prompts import BASELINE_SYSTEM, baseline_user_message
from analyst.config import ANSWER_MODEL
from analyst.db.store import connect
from analyst.retrieval.hybrid import hybrid_search
from analyst.retrieval.rerank import rerank_search
from analyst.retrieval.vector import Hit, vector_search

TOP_K = 6
RETRIEVERS = {"vector": vector_search, "hybrid": hybrid_search, "rerank": rerank_search}
CITATION_RE = re.compile(r"\[([^\[\],]+), p\.(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    citations: list[tuple[str, int]]  # (doc_name, page) actually cited in the answer
    retrieved: list[Hit]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    extra: dict = field(default_factory=dict)


def parse_citations(text: str) -> list[tuple[str, int]]:
    seen = []
    for doc, page in CITATION_RE.findall(text):
        cite = (doc.strip(), int(page))
        if cite not in seen:
            seen.append(cite)
    return seen


def answer(
    question: str,
    conn=None,
    client: anthropic.Anthropic | None = None,
    retriever: str = "vector",
    top_k: int = TOP_K,
) -> Answer:
    client = client or anthropic.Anthropic()
    own_conn = conn is None
    conn = conn or connect()
    try:
        hits = RETRIEVERS[retriever](conn, question, k=top_k)
    finally:
        if own_conn:
            conn.close()

    response = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=4000,
        output_config={"effort": "medium"},
        system=BASELINE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": baseline_user_message(question, [(h.label, h.text) for h in hits]),
            }
        ],
    )

    if response.stop_reason == "refusal":
        text = "[model declined to answer]"
    else:
        text = "".join(b.text for b in response.content if b.type == "text").strip()

    return Answer(
        question=question,
        text=text,
        citations=parse_citations(text),
        retrieved=hits,
        model=ANSWER_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        extra={"stop_reason": response.stop_reason},
    )


def _print(a: Answer, gold: dict | None = None) -> None:
    print("=" * 80)
    print(f"Q: {a.question}")
    if gold:
        gold_pages = [e["evidence_page_num"] + 1 for e in gold["evidence"]]
        print(f"GOLD: {gold['answer']}   [{gold['doc_name']}, p.{gold_pages}]")
    print(f"\n{a.text}\n")
    print("Retrieved:", ", ".join(f"{h.label} {h.score:.2f}" for h in a.retrieved))
    print(f"Tokens: {a.input_tokens} in / {a.output_tokens} out  ({a.model})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--samples", type=int, help="run N hand-picked FinanceBench questions")
    args = parser.parse_args()

    if args.question:
        _print(answer(args.question))
        return

    from analyst.ingest.financebench import selected_questions

    qs = selected_questions()
    # Spread across companies rather than taking the first N (which are all 3M).
    picks, seen = [], set()
    for q in qs:
        if q["company"] not in seen:
            picks.append(q)
            seen.add(q["company"])
    client = anthropic.Anthropic()
    with connect() as conn:
        for q in picks[: args.samples or 5]:
            _print(answer(q["question"], conn=conn, client=client), gold=q)


if __name__ == "__main__":
    main()
