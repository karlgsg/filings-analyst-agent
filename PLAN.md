# Filings Analyst Agent — Build Plan

An AI agent that answers questions about public-company filings (10-Ks, 10-Qs, earnings reports) with **cited sources**, does financial math with **real tools** instead of guessing, and is **measured** against a public benchmark (FinanceBench).

Repo name suggestion: `filings-analyst-agent` (or pick your own brand name).

---

## The resume line we're building toward

> Built an AI financial analyst agent that answers questions over SEC filings with page-level citations; improved accuracy on the FinanceBench benchmark from **X%** (baseline RAG) to **Y%** using hybrid retrieval, reranking, and tool-calling, with automated evals in CI and a deployed demo.

Every phase below exists to make that sentence true and to fill in X and Y with real numbers.

---

## Tech stack (and why — you'll be asked)

| Piece | Choice | Why this over alternatives |
|---|---|---|
| Language | Python 3.11+, managed with `uv` | Standard for AI work; `uv` is the modern, fast package manager |
| LLM | Claude or OpenAI via the **official SDK** (no LangChain) | You learn how tool calling actually works, and you can explain every line |
| Database | **Postgres + pgvector** (in Docker) | One database for vectors *and* keyword search *and* SQL. Very relevant to data/analyst roles too |
| Search | **Hybrid**: vector similarity + Postgres full-text, merged with Reciprocal Rank Fusion | Plain vector search misses exact terms like "FY2022" or "EBITDA" |
| Reranker | Local cross-encoder (`sentence-transformers`, e.g. `ms-marco-MiniLM`) | Free, runs on your laptop, and gives a big accuracy jump |
| Embeddings | `text-embedding-3-small` (OpenAI) or local `bge-small` | Cheap or free; swappable |
| API | FastAPI | Industry standard, auto-generated docs |
| UI | Streamlit (fast); optional React upgrade later | Get a demo running in hours, not days |
| Evals | FinanceBench open set (150 Qs) + your own scorer | Gives you the real numbers |
| Tracing | Langfuse (free tier) | Shows each step, latency, and cost per question |
| CI | GitHub Actions | Runs tests and a small eval on every push |
| Deploy | Render/Fly.io (API) + Neon or Supabase (Postgres with pgvector) + Streamlit Cloud | Free or cheap tiers, gives you a live link |

FinanceBench open set: 150 questions with gold answers, evidence and source PDFs, on GitHub (`patronus-ai/financebench`) and Hugging Face (`PatronusAI/financebench`). License is CC-BY-NC, which is fine for a portfolio project.

---

## Folder structure

```
filings-analyst-agent/
├── README.md
├── CLAUDE.md                 # instructions for Claude in VS Code (see bottom)
├── pyproject.toml
├── docker-compose.yml        # Postgres + pgvector
├── .env.example              # API keys go in .env (never commit .env)
├── .github/workflows/ci.yml
├── data/
│   ├── financebench/         # questions + PDFs (gitignored if large)
│   └── raw/                  # downloaded filings
├── src/analyst/
│   ├── config.py
│   ├── ingest/
│   │   ├── pdf_parser.py     # PDF -> pages -> chunks (keep page numbers!)
│   │   ├── chunker.py
│   │   └── edgar.py          # (Phase 3) fetch filings by ticker from SEC EDGAR
│   ├── db/
│   │   ├── schema.sql
│   │   └── store.py          # insert + search
│   ├── retrieval/
│   │   ├── vector.py
│   │   ├── keyword.py
│   │   ├── hybrid.py         # RRF fusion
│   │   └── rerank.py
│   ├── agent/
│   │   ├── tools.py          # search_filings, calculate, get_table...
│   │   ├── loop.py           # the tool-calling loop
│   │   └── prompts.py
│   ├── api/main.py           # FastAPI
│   └── ui/app.py             # Streamlit
├── evals/
│   ├── run_eval.py           # run N questions, save results
│   ├── scorer.py             # numeric tolerance + LLM judge
│   └── results/              # JSON results per config (commit these!)
└── tests/
```

---

## Phase 1 — Working MVP (Week 1)

Goal: ask a question about a filing, get an answer with page citations. **Push to GitHub by end of week.**

Each task = roughly one Claude-in-VS-Code session = **one commit**.

1. **Project setup.** `uv init`, folder structure, `.gitignore`, `.env.example`, `docker-compose.yml` with `pgvector/pgvector` image. *Done when:* `docker compose up` starts Postgres and you can connect.
   *Learn:* Docker basics, environment variables, why secrets never go in git.

2. **Load FinanceBench.** Script to download the open-source JSONL + PDFs; pick ~10 companies to start. *Done when:* you can print a question, its gold answer, and its source doc.

3. **PDF → chunks.** Parse PDFs page by page (`pypdf` or `pymupdf`), chunk ~500–800 tokens with overlap, **keep doc name + page number on every chunk**. *Done when:* chunks print with correct page numbers.
   *Learn:* why chunk size and overlap matter. Tables are the hard part; note it for later.

4. **Schema + embeddings.** `chunks` table with `embedding vector(...)` and a `tsvector` column for full-text search. Embed and insert all chunks. *Done when:* row count matches chunk count.
   *Learn:* what an embedding is, vector indexes (HNSW), SQL full-text search.

5. **Vector search + answer (the baseline).** Top-k vector search → stuff chunks into the prompt → answer with `[doc, page]` citations. *Done when:* 5 hand-picked questions return sensible, cited answers.
   **This is your baseline. Don't make it smart yet; you need it to compare against.**

6. **Minimal eval.** `run_eval.py` runs the baseline on 30 FinanceBench questions; `scorer.py` marks each one correct or incorrect (numbers within ~1–2% tolerance, otherwise an LLM judge compares to the gold answer). Save results to `evals/results/baseline.json`. *Done when:* you have **X%**.
   *Learn:* why "it looks right" isn't evaluation.

7. **README v1 + push.** What it does, how to run it, baseline score. **Add the GitHub link to your resume now.**

---

## Phase 2 — Make it smart, prove it with numbers (Week 2)

Change **one thing at a time** and re-run the eval after each. Log every result in a table. That table is your interview story.

8. **Keyword search + hybrid fusion (RRF).** *Eval →* record score.
9. **Reranker.** Retrieve top 30 with hybrid search, rerank, keep top 5–8. *Eval →* record score.
10. **Agent loop with tools.** Replace "stuff and answer" with a tool-calling loop:
    - `search_filings(query, company, year)`: the agent can search several times
    - `calculate(expression)`: **safe** math only, with a whitelist of functions (ratio, growth %, margin, CAGR). **Never use Python `eval()`.**
    - `final_answer(answer, citations)`: structured output
    Cap it at ~6 steps. *Eval →* record score.
    *Learn:* how tool calling actually works under the hood, and why the model shouldn't do arithmetic in its head.
11. **Tracing.** Add Langfuse: per-question steps, tokens, latency, cost. Add **avg cost per question** and **p50 latency** to your results table.
12. **Full 150-question run** on the best config → **Y%**. Write up what helped, what didn't, and why. *Failures are good content.*

Example results table for the README (fill in your own numbers):

| Config | Accuracy | Avg cost/Q | p50 latency |
|---|---|---|---|
| Baseline vector RAG | ?% | ? | ? |
| + hybrid search | ?% | ? | ? |
| + reranker | ?% | ? | ? |
| + agent + calculator | ?% | ? | ? |

---

## Phase 3 — Ship it (Week 3)

13. **FastAPI endpoint** `/ask` returning answer, citations and trace ID.
14. **Streamlit UI:** question box, answer, clickable citations showing the source text, company/year filter, and a "show reasoning steps" toggle.
15. **EDGAR ingestion.** Enter any ticker → fetch its latest 10-K from SEC EDGAR → index it. (EDGAR requires a `User-Agent` header with your name/email and has rate limits; respect them.) This turns it from a benchmark into a real product.
16. **CI.** GitHub Actions: lint (`ruff`) + unit tests on every push; a **20-question smoke eval** that fails the build if accuracy drops more than 5 points. *(Use a GitHub secret for the API key, and keep it small to control cost.)*
17. **Deploy.** Live link in the README. Add a daily request cap so a stranger can't burn your API budget.
18. **README v2:** demo GIF, architecture diagram, results table, design decisions, limitations.

---

## Phase 4 — Optional upgrades (only if time allows)

- **Security hardening (ties to your SE 4472 course):** treat filing text as untrusted input. Test prompt injection by planting "ignore previous instructions" in a document and confirm the agent doesn't follow it. Keep tools allowlisted. *Very few student projects do this, and it's a great interview talking point.*
- **Table extraction:** many FinanceBench misses come from tables; try a table-aware parser.
- **Analyst report mode:** "Compare Apple vs Microsoft margins over 3 years" → a one-page memo with a chart.
- **MCP server:** expose your tools so Claude Desktop and other clients can use your agent.
- **Real users:** get 10–20 Ivey classmates to use it for a case or stock pitch; collect feedback.

**Cut order if you're short on time:** MCP → React UI → table extraction → EDGAR. **Never cut evals or deployment.**

---

## Interview prep: be able to answer these

- Why hybrid search instead of only vectors? What does RRF do?
- What does a reranker do that embeddings don't?
- How does your agent decide when to stop? What happens if it loops?
- Why a calculator tool instead of letting the LLM do math?
- How did you score answers? What are the weaknesses of an LLM judge?
- What was your biggest accuracy gain, and what *didn't* work?
- How much does one question cost, and how would you cut that in half?
- How do you defend against prompt injection inside documents?
- Why no LangChain?

---

## CLAUDE.md (copy into the repo root)

```markdown
# Project: Filings Analyst Agent
See PLAN.md for the full plan. We work one numbered task at a time.

## How to work with me
- I'm learning. Before writing code for a task, explain the approach in a few sentences and any new concept involved.
- Keep changes small and focused on the current task only.
- After each task: run tests, show me how to verify it works, and suggest a clear commit message. I make the commit.
- Prefer simple, readable code over clever code. No LangChain; use the official LLM SDK directly.
- Never put secrets in code; use .env. Never use Python eval() on model output.
- When something is a design decision, tell me the tradeoff so I can explain it in interviews.
```

---

## Daily rhythm

1. Open the next task in VS Code: "Let's do task N from PLAN.md."
2. Read Claude's explanation **before** the code.
3. Run it yourself and verify the "Done when" check.
4. Commit with a clear message, then push.
5. Write one line in a `NOTES.md` about what you learned. That's your interview material.
