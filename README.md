# Filings Analyst Agent

An AI analyst that answers questions about public-company filings (10-Ks, 10-Qs, 8-Ks, earnings releases) with **page-level citations**, and is **measured** against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark rather than judged by eye.

```
Q: What is Amcor's year end FY2020 net AR (in USD millions)?

A: $1,615.9 million (trade receivables, net, as of June 30, 2020).
   Amcor's FY2020 balance sheet reports "Trade receivables, net" of $1,615.9 million,
   down from $1,864.3 million a year earlier [AMCOR_2020_10K, p.50].
```

> **Status:** Phase 2 done — **82.7% on all 150 FinanceBench questions with Claude Haiku 5.5 at $0.002/question.** Next: API + UI (Phase 3). See [PLAN.md](PLAN.md).

## Results

### Full benchmark: all 150 FinanceBench questions

The final agent over all 84 filings (32 companies, 20,474 chunks), with three answering models. The judge is Haiku 5.5 in every run, so grading is identical.

| Answering model | Accuracy | Retrieval hit rate | Cost / question | p50 latency | Avg steps |
|---|---|---|---|---|---|
| **Claude Haiku 5.5** | **82.7%** | 88.0% | **$0.0020** | 5.0 s | 2.7 |
| Claude Sonnet 5.5 | 83.3% | 85.3% | $0.0258 | 5.3 s | 2.5 |
| Claude Opus 5.5 | 86.7% | 90.0% | $0.0634 | 8.0 s | 2.9 |

**A bigger model buys little here.** Sonnet costs 13× more than Haiku for one extra question. Opus costs 31× more for six, and that gap isn't statistically solid: Opus fixed 11 of Haiku's misses but missed 5 that Haiku got (McNemar p ≈ 0.21). The default stays Haiku. Full per-question results: [evals/results/](evals/results/).

| FinanceBench question type (50 each) | Haiku | Sonnet | Opus |
|---|---|---|---|
| metrics-generated (compute a ratio from the statements) | 98% | 96% | 96% |
| domain-relevant (analyst judgement, e.g. "is it capital-intensive?") | 80% | 76% | 86% |
| novel-generated (free-form questions about one filing) | 70% | 78% | 78% |

**What's left is mostly not the model.** The calculation questions are essentially solved. 13 questions were missed by *all three* models, and they usually agree with each other against the answer key:
- **6 are definitional.** The key uses a specific formula (e.g. quick ratio 0.96 vs 0.85 from cash + securities + receivables, or "operating" working capital $831M vs reported $2,278M), or it makes a judgement call the models read the other way.
- **2 don't name a company** ("Were there any board nominees with many votes against?"). FinanceBench assumes you're handed one filing; this agent searches all 84 at once (see *Design decisions*).
- **3 have a debatable answer key.** Boeing's FY2021 tax rate has the opposite sign, and the Upjohn spin-off was already complete by the Q2 2023 filing.
- **About 2 are genuine misses**, e.g. PepsiCo restructuring costs live in the notes, not as an income-statement line.

Of Haiku's 26 misses, the gold page *was* retrieved for 21, so retrieval is no longer the bottleneck it was in the baseline.

**A crash, and recovering from it.** The Sonnet and Opus runs stopped partway when API credits ran out. The eval only saved results at the end, so those answers would have been lost, but every one was already in Langfuse. I rebuilt the 119 finished rows from the traces (all of them matched the run logs exactly) and resumed each run in its original session. The eval now checkpoints each row as it's scored (`--resume`), so a crash costs nothing.

### Development: what each component added

Built on a fixed 30-question subset (top-10 companies; same questions for every row):

| Config | Accuracy | Retrieval hit rate | Cost / question | p50 latency |
|---|---|---|---|---|
| Baseline: vector search → Claude Haiku 5.5 | **33.3%** | 33.3% | $0.0009 | 3.5 s |
| + hybrid search (vector + IDF keyword, RRF) | 33.3% | 30.0% | $0.0010 | 3.6 s |
| + cross-encoder reranker (hybrid top-~55 → MiniLM → top 8) | 33.3% | 40.0% | $0.0011 | 3.8 s |
| **+ tool-calling agent** (multi-search with filters, calculator) | **76.7%**¹ | 90.0% | $0.0020 | 4.9 s |

*Retrieval hit rate* = how often the page containing the gold evidence was among the retrieved chunks.
¹ Two runs of the same agent scored 83.3% and 76.7%. The agent's answers were essentially identical; the difference is two borderline judge calls (see *Grading noise* below). The table reports the stricter run.

**Hybrid search: same score, different questions.** It gained 3 questions that hinge on exact years and figures (e.g. Best Buy's FY2015–17 margin) and lost 3 others, net zero. Plain Postgres full-text ranking has no notion of word rarity, so keyword search alone found the gold page only 3% of the time; adding IDF weighting raised that to 27%. The useful finding: vector and keyword search find *different* gold pages — their combined top-30 candidates contain the gold page for **63%** of questions vs 53% for vector alone — but RRF can't pick the right 6 from that pool. That's the reranker's job (next).

**Tracing (Langfuse).** Every question is a trace: each Claude call is a generation (tokens, cost, latency), each search/calculation a timed span, and the eval's verdict is attached as a score, so the dashboard can be filtered straight to wrong answers. Third-party SDK spans are filtered out to avoid double-counting cost. Where the time goes, per question: **Claude calls 4.8 s, search 0.9 s** (incl. the local reranker), calculator ~0 s — so speeding the agent up means fewer or cheaper model steps, not faster retrieval.

**Grading noise.** Re-running the agent unchanged moved accuracy from 83.3% to 73.3%. Comparing the runs question by question, *none* of the flips came from the agent: two were the LLM judge ruling differently on near-identical answers, and one was a scorer bug (a correct answer that mentioned "I could not find any FY2021 acquisitions" in its last sentence was auto-failed as "not found"). The bug is fixed and both affected saved results were re-scored (rerank unchanged, agent → 76.7%). With 30 questions, one question is 3.3 points — differences under ~10 points are within noise, which is why the headline number comes from the full 150-question run above. The numeric check also now skips years and dates ("In FY2022, …" used to be read as the number 2022), so a third of answers are graded in code instead of by the judge, which is free and deterministic.

**Agent: 33% → 77–83%.** Replacing "retrieve once, then answer" with a tool-calling loop was the big win. The model searches as many times as it needs (avg 2.6 steps, max 6) with company/year filters, does all arithmetic in a safe `calculate` tool, and submits a structured `final_answer` with citations. "Couldn't find it" answers dropped from 12 to 0, and the multi-statement questions (EBITDA, multi-year averages) are now answered correctly. Cost doubled to ~$0.002/question; prompt caching keeps it there (most of each step's input is re-sent conversation, billed at 10%). All 30 judge verdicts were hand-reviewed: 28 agreed, one lenient and one harsh call, net zero.

**Reranker: retrieval up, accuracy flat.** Reranking the hybrid candidates with a local cross-encoder lifted the gold-page hit rate from 33% to 40% (the small MiniLM model matched the larger `bge-reranker-base` at ~6× the speed). Accuracy stayed at 33.3%. Of the 4 questions that retrieved a gold page but were still wrong, 2 need figures from **two different statements** (e.g. EBITDA = operating income from the income statement + D&A from the cash flow statement) — one retrieval pass finds one page, never both — and 2 are judgment calls where the model read the question differently. Both point to the next step: an agent that can search multiple times and calculate.

**What the baseline tells us:** half of the misses are the model correctly saying *"I couldn't find this in the provided filings"* — the right page was never retrieved. Retrieval, not the LLM, is the bottleneck, which sets the order of Phase 2. Full per-question results: [evals/results/baseline.json](evals/results/baseline.json).

## How it works

```
                 ┌──────────── ingest (once) ─────────────┐
FinanceBench PDFs → page text (PyMuPDF) → ~450-word chunks → bge-small embeddings → Postgres + pgvector
                                          (never cross pages,                         (HNSW vector index +
                                           keep doc + page)                            full-text tsvector)

Baseline:  question → top-6 nearest chunks → Claude → answer + citations

Agent:     question → Claude ─┬─ search_filings(query, company, year)   hybrid search → cross-encoder rerank → top 6
                              ├─ calculate("11512 + 2763 - 5207")        safe AST evaluator, no eval()
                              ├─ ... (up to 6 steps)
                              └─ final_answer(answer, citations)
```

**Evaluation** ([evals/](evals/)): each answer is scored against FinanceBench's gold answer. Numeric answers are checked in code (±2%, with million/billion scaling); everything else goes to an LLM judge with structured output. Judge verdicts were spot-checked by hand.

## Tech stack

| | |
|---|---|
| LLM | Claude via the official `anthropic` SDK (no LangChain) — Haiku 5.5 by default, model is a config switch |
| Embeddings | `BAAI/bge-small-en-v1.5`, run locally (free) |
| Database | Postgres 17 + pgvector (Docker) — vectors and full-text search in one place |
| API | FastAPI (`POST /ask`, `GET /companies`, `GET /health`; auto-generated docs at `/docs`) |
| UI | Streamlit chat: expandable citations with the source passage, company/year filter, reasoning-steps toggle |
| Tracing | Langfuse (optional — disabled when keys aren't set) |
| Tooling | Python 3.12, `uv`, `pytest`, `ruff` |

## Run it locally

Requires Docker (or OrbStack), [uv](https://docs.astral.sh/uv/), and an [Anthropic API key](https://platform.claude.com).

```bash
cp .env.example .env                              # add your ANTHROPIC_API_KEY
docker compose up -d                              # Postgres + pgvector
uv sync                                           # Python deps

uv run python -m analyst.ingest.financebench      # questions + PDFs for the top-10 companies (~50 MB)
uv run python -m analyst.ingest.financebench --top 32   # ...or all 84 filings (~160 MB)
uv run python -m analyst.ingest.index             # chunk, embed, store (~7k / ~20k chunks)

uv run python -m analyst.agent.loop "What was Boeing's revenue in FY2022?"   # the agent, from the CLI
uv run uvicorn analyst.api.main:app --reload     # the API: docs at http://127.0.0.1:8000/docs
uv run streamlit run src/analyst/ui/app.py       # the chat UI (in a second terminal; needs the API)
uv run python evals/run_eval.py --config baseline # run the 30-question eval (~$0.03)
uv run python evals/run_eval.py --config agent --retriever agent --full   # all 150 (~$0.30 on Haiku)
uv run pytest
```

Ask the API directly:

```bash
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "What was Walmart'"'"'s FY2020 total revenue?"}'
# -> {"answer": "...$523,964 million...", "citations": [{"doc_name": "WALMART_2020_10K", "page": 51,
#     "text": "..."}], "steps": [...], "cost_usd": 0.0011, "latency_s": 3.0, "trace_id": "..."}
```

Switch models per run: `ANSWER_MODEL=claude-sonnet-5-5 uv run python evals/run_eval.py --config baseline-sonnet`. If a run is interrupted, re-run the same command with `--resume`.

## Design decisions

- **Chunks never cross page boundaries**, so every citation points at exactly one page. Trade-off: a table split across two pages becomes two chunks.
- **Search across all filings, no company filter.** Harder than searching only the right document, but closer to how a real user asks questions — and the number we report is the honest one.
- **Fixed, committed eval set** (seed 42) so every configuration is compared on identical questions.
- **Free checks first:** numeric answers and "not found" responses are scored in code; only the rest pay for an LLM judge.
- **Haiku as the default model.** On the full benchmark it's within 4 points of Opus at 1/31 of the cost; the model is one env var (`ANSWER_MODEL`) if a use case needs the extra accuracy.
- **UI → API → agent.** The Streamlit app only calls the HTTP API, so it deploys separately and anything else can use the same API. Trade-off: two processes to run locally.
- **UI filters are enforced, not suggested.** Picking a company/year overrides the filter on every search the agent makes (and tells the model the scope), rather than hoping the model follows an instruction.
- **Filing text is treated as untrusted input** in the prompt (groundwork for prompt-injection testing).

## Project layout

```
src/analyst/
  ingest/      download FinanceBench, PDF → chunks, index into Postgres
  db/          schema + database access
  retrieval/   vector, keyword (IDF-weighted), hybrid (RRF), cross-encoder rerank
  agent/       prompts, baseline answerer, tool-calling loop, tools, safe calculator
  api/         FastAPI app: POST /ask, GET /companies, GET /health
  ui/          Streamlit chat UI (talks to the API over HTTP)
evals/         eval set, scorer, runner, committed results
tests/         unit tests
```

## Data & credits

Questions, gold answers, and filings come from **FinanceBench** by Patronus AI ([paper](https://arxiv.org/abs/2311.11944), [repo](https://github.com/patronus-ai/financebench)), licensed CC BY-NC 4.0. This project uses the open-source 150-question set; currently 10 companies (81 questions, 33 filings) are indexed.
