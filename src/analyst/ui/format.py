"""Pure text helpers for the UI, kept separate so they're easy to unit-test."""

from __future__ import annotations

import json


def md_safe(text: str) -> str:
    """Streamlit renders $...$ as LaTeX math, which garbles '$1,577 million ... $4,000'."""
    return text.replace("$", r"\$")


def strip_sources(answer: str) -> str:
    """The agent appends 'Sources: [doc, p.N], ...'; the UI shows citations separately."""
    return answer.split("\n\nSources: ")[0].strip()


def describe_step(step: dict) -> str:
    """One line per agent step, e.g. "search_filings: query='capex', company='3M' -> 6 pages"."""
    args = ", ".join(f"{k}={v!r}" for k, v in (step.get("input") or {}).items())
    if step["tool"] == "final_answer":
        return "final_answer"
    out = step.get("output")
    if isinstance(out, list):
        result = f"{len(out)} pages: " + ", ".join(out)
    else:
        result = json.dumps(out) if not isinstance(out, str) else out
    return f"{step['tool']}({args}) → {result}"
