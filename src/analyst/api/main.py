"""HTTP API around the agent.

Run:   uv run uvicorn analyst.api.main:app --reload
Docs:  http://127.0.0.1:8000/docs   (interactive, generated from the models below)
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated

import anthropic
import psycopg
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from analyst import tracing
from analyst.agent.loop import run_agent
from analyst.config import cost_usd
from analyst.db.store import connect

# ---- request / response shapes (FastAPI validates JSON against these) ----


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["What was 3M's FY2018 capex?"])
    company: str | None = Field(None, description="Only search this company's filings")
    year: int | None = Field(None, description="Only search filings for this fiscal year")


class Company(BaseModel):
    company: str
    years: list[int]


class Source(BaseModel):
    doc_name: str
    page: int
    text: str | None = None  # the retrieved passage from that page, if we have it


class Step(BaseModel):
    step: int
    tool: str
    input: dict
    output: str | list | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Source]
    steps: list[Step]  # what the agent did: searches, calculations, final answer
    model: str
    cost_usd: float
    latency_s: float
    trace_id: str | None = None
    trace_url: str | None = None


# ---- app setup ----


def warm_up() -> None:
    """Load the local embedding + reranker models now, not on the first request (~5 s)."""
    from analyst.embeddings import embed_query
    from analyst.retrieval.rerank import RERANK_MODEL, _model

    embed_query("warm up")
    _model(RERANK_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    yield
    tracing.flush()  # send any buffered traces before the server exits


app = FastAPI(title="Filings Analyst Agent", lifespan=lifespan)
_client = anthropic.Anthropic()  # thread-safe, shared by all requests


def get_conn() -> Iterator:
    """One DB connection per request: a psycopg connection must not be shared across
    threads, and FastAPI runs each sync request in its own worker thread."""
    with connect() as conn:
        yield conn


def get_client() -> anthropic.Anthropic:
    return _client


# "Dependencies": FastAPI calls these per request and passes the result in. Tests swap
# them out with app.dependency_overrides.
Conn = Annotated[psycopg.Connection, Depends(get_conn)]
Client = Annotated[anthropic.Anthropic, Depends(get_client)]


# ---- endpoints ----


@app.get("/companies")
def companies(conn: Conn) -> list[Company]:
    """What's indexed: each company and the fiscal years of its filings (for UI filters)."""
    rows = conn.execute(
        "SELECT company, array_agg(DISTINCT doc_year ORDER BY doc_year) FROM chunks "
        "GROUP BY company ORDER BY company"
    ).fetchall()
    return [Company(company=c, years=y) for c, y in rows]


@app.get("/health")
def health(conn: Conn) -> dict:
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": chunks}


# A plain `def` (not `async def`): the agent blocks on Claude and Postgres, so FastAPI
# runs it in a thread pool instead of freezing the server while one request waits.
@app.post("/ask")
def ask(req: AskRequest, conn: Conn, client: Client) -> AskResponse:
    start = time.perf_counter()
    try:
        with tracing.trace_attributes(trace_name="api ask", tags=["api"]):
            a = run_agent(
                req.question, conn=conn, client=client, company=req.company, year=req.year
            )
    except anthropic.RateLimitError:
        raise HTTPException(503, "The model is busy (rate limited). Try again shortly.")
    except anthropic.APIError as e:
        raise HTTPException(502, f"Model API error: {e.__class__.__name__}")

    # Attach the retrieved passage to each citation so a UI can show the source text.
    passages = {(h.doc_name, h.page): h.text for h in a.retrieved}
    citations = [Source(doc_name=d, page=p, text=passages.get((d, p))) for d, p in a.citations]
    trace_id = a.extra.get("trace_id")
    return AskResponse(
        answer=a.text,
        citations=citations,
        steps=[Step(**{k: t.get(k) for k in Step.model_fields}) for t in a.extra["trace"]],
        model=a.model,
        cost_usd=cost_usd(
            a.model, a.input_tokens, a.output_tokens, a.cache_write_tokens, a.cache_read_tokens
        ),
        latency_s=round(time.perf_counter() - start, 2),
        trace_id=trace_id,
        trace_url=tracing.trace_url(trace_id),
    )
