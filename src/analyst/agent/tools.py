"""Tool definitions (JSON schemas the model sees) and their Python implementations.

The model never runs anything itself: it returns a `tool_use` block naming a tool and
its arguments, and `run_tool` below decides what actually executes.
"""

from __future__ import annotations

import psycopg

from analyst.agent.calculator import CalcError, calculate
from analyst.retrieval.rerank import rerank_search
from analyst.retrieval.vector import Hit

SEARCH_K = 6


def list_companies(conn: psycopg.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT company FROM chunks ORDER BY 1")]


def tool_definitions(companies: list[str]) -> list[dict]:
    """Built from the database so the company list stays in sync with what's indexed.
    Keep this deterministic (sorted): it's part of the cached prompt prefix."""
    return [
        {
            "name": "search_filings",
            "description": (
                "Search the indexed SEC filings (10-K, 10-Q, 8-K, earnings releases) and return "
                f"the {SEARCH_K} most relevant excerpts, each labeled [doc_name, p.N]. "
                "Use short, specific queries naming one figure or topic, e.g. "
                "'cash flow statement purchases of property plant and equipment'. "
                "Call it several times to gather different figures."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "company": {
                        "type": "string",
                        "enum": companies,
                        "description": "Only search this company's filings. Strongly recommended.",
                    },
                    "year": {
                        "type": "integer",
                        "description": (
                            "Only search filings for this fiscal year. A 10-K also reports the "
                            "prior 1-2 years, so FY2015 figures may be in the FY2016 or FY2017 "
                            "10-K. Omit if unsure."
                        ),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "calculate",
            "description": (
                "Evaluate an arithmetic expression exactly. Use this for ALL math; never compute "
                "in your head. Supports + - * / ** %, parentheses, and functions: abs, round, "
                "min, max, sum, avg, sqrt, pct_change(new, old), margin(part, whole) (percent), "
                "cagr(start, end, years) (percent). Write plain numbers: 1577, not 1,577 or $1,577."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. margin(1228, 39403)"}
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
        {
            "name": "final_answer",
            "description": "Submit the final answer. Call exactly once, when you are done.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": (
                            "Start with the direct answer (a number with units, or a short phrase) "
                            "in the first sentence, then a brief explanation of how it was derived."
                        ),
                    },
                    "citations": {
                        "type": "array",
                        "description": "Every excerpt the answer relies on.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doc_name": {"type": "string"},
                                "page": {"type": "integer"},
                            },
                            "required": ["doc_name", "page"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["answer", "citations"],
                "additionalProperties": False,
            },
        },
    ]


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return "No matching excerpts. Try different words, or drop the year filter."
    return "\n\n".join(f"{h.label}\n{h.text}" for h in hits)


def run_tool(name: str, args: dict, conn: psycopg.Connection) -> tuple[str, list[Hit], bool]:
    """Execute one tool call. Returns (result text, hits retrieved, is_error)."""
    if name == "search_filings":
        hits = rerank_search(
            conn,
            args["query"],
            k=SEARCH_K,
            company=args.get("company"),
            year=args.get("year"),
        )
        return format_hits(hits), hits, False
    if name == "calculate":
        try:
            value = calculate(args["expression"])
        except (CalcError, ArithmeticError, TypeError) as e:
            return f"Error: {e}", [], True
        return f"{value:.6g}" if abs(value) < 1e15 else str(value), [], False
    return f"Error: unknown tool {name!r}", [], True
