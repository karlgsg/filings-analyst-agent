# Filings Analyst Agent

An AI analyst that answers questions about public-company filings (10-Ks, 10-Qs, 8-Ks, earnings releases) with **page-level citations**, and is **measured** against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark rather than judged by eye.

```
Q: What is Amcor's year end FY2020 net AR (in USD millions)?

A: $1,615.9 million (trade receivables, net, as of June 30, 2020).
   Amcor's FY2020 balance sheet reports "Trade receivables, net" of $1,615.9 million,
   down from $1,864.3 million a year earlier [AMCOR_2020_10K, p.50].
```

> **Status:** Phase 1 complete — working baseline + evaluation harness. Phase 2 (hybrid search, reranking, tool-calling agent) in progress. See [PLAN.md](PLAN.md).

## Results

Accuracy on a fixed 30-question FinanceBench subset (same questions for every row):

| Config | Accuracy | Retrieval hit rate | Cost / question | p50 latency |
|---|---|---|---|---|
| Baseline: vector search → Claude Haiku 5.5 | **33.3%** | 33.3% | $0.0009 | 3.5 s |
| + hybrid search (vector + IDF keyword, RRF) | 33.3% | 30.0% | $0.0010 | 3.6 s |
| + cross-encoder reranker | | | | |
| + tool-calling agent with calculator | | | | |

*Retrieval hit rate* = how often the page containing the gold evidence was among the retrieved chunks.

**Hybrid search: same score, different questions.** It gained 3 questions that hinge on exact years and figures (e.g. Best Buy's FY2015–17 margin) and lost 3 others, net zero. Plain Postgres full-text ranking has no notion of word rarity, so keyword search alone found the gold page only 3% of the time; adding IDF weighting raised that to 27%. The useful finding: vector and keyword search find *different* gold pages — their combined top-30 candidates contain the gold page for **63%** of questions vs 53% for vector alone — but RRF can't pick the right 6 from that pool. That's the reranker's job (next).

**What the baseline tells us:** half of the misses are the model correctly saying *"I couldn't find this in the provided filings"* — the right page was never retrieved. Retrieval, not the LLM, is the bottleneck, which sets the order of Phase 2. Full per-question results: [evals/results/baseline.json](evals/results/baseline.json).

## How it works

```
                 ┌──────────── ingest (once) ─────────────┐
FinanceBench PDFs → page text (PyMuPDF) → ~450-word chunks → bge-small embeddings → Postgres + pgvector
                                          (never cross pages,                         (HNSW vector index +
                                           keep doc + page)                            full-text tsvector)

question → embed → top-6 nearest chunks → Claude (answer only from excerpts, cite [doc, p.N]) → answer + citations
```

**Evaluation** ([evals/](evals/)): each answer is scored against FinanceBench's gold answer. Numeric answers are checked in code (±2%, with million/billion scaling); everything else goes to an LLM judge with structured output. Judge verdicts were spot-checked by hand.

## Tech stack

| | |
|---|---|
| LLM | Claude via the official `anthropic` SDK (no LangChain) — Haiku 5.5 by default, model is a config switch |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally (free) |
| Database | Postgres 17 + pgvector (Docker) — vectors and full-text search in one place |
| Tooling | Python 3.12, `uv`, `pytest`, `ruff` |

## Run it locally

Requires Docker (or OrbStack), [uv](https://docs.astral.sh/uv/), and an [Anthropic API key](https://platform.claude.com).

```bash
cp .env.example .env                              # add your ANTHROPIC_API_KEY
docker compose up -d                              # Postgres + pgvector
uv sync                                           # Python deps

uv run python -m analyst.ingest.financebench      # download questions + PDFs (~50 MB)
uv run python -m analyst.ingest.index             # chunk, embed, store (~7k chunks)

uv run python -m analyst.agent.baseline "What was Boeing's revenue in FY2022?"
uv run python evals/run_eval.py --config baseline # run the 30-question eval (~$0.03)
uv run pytest
```

Switch models per run: `ANSWER_MODEL=claude-sonnet-5-5 uv run python evals/run_eval.py --config baseline-sonnet`

## Design decisions

- **Chunks never cross page boundaries**, so every citation points at exactly one page. Trade-off: a table split across two pages becomes two chunks.
- **Search across all filings, no company filter.** Harder than searching only the right document, but closer to how a real user asks questions — and the number we report is the honest one.
- **Fixed, committed eval set** (seed 42) so every configuration is compared on identical questions.
- **Free checks first:** numeric answers and "not found" responses are scored in code; only the rest pay for an LLM judge.
- **Filing text is treated as untrusted input** in the prompt (groundwork for prompt-injection testing).

## Project layout

```
src/analyst/
  ingest/      download FinanceBench, PDF → chunks, index into Postgres
  db/          schema + database access
  retrieval/   vector, keyword (IDF-weighted), and hybrid (RRF) search
  agent/       prompts + baseline answerer
evals/         eval set, scorer, runner, committed results
tests/         unit tests
```

## Data & credits

Questions, gold answers, and filings come from **FinanceBench** by Patronus AI ([paper](https://arxiv.org/abs/2311.11944), [repo](https://github.com/patronus-ai/financebench)), licensed CC BY-NC 4.0. This project uses the open-source 150-question set; currently 10 companies (81 questions, 33 filings) are indexed.
