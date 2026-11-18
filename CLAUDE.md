# Project: Filings Analyst Agent
See PLAN.md for the full plan. We work one numbered task at a time.

## How to work with me
- I'm learning. Before writing code for a task, explain the approach in a few sentences and any new concept involved.
- Keep changes small and focused on the current task only.
- After each task: run tests, show me how to verify it works, and then commit with a clear message and push to origin/main.
- Prefer simple, readable code over clever code. No LangChain; use the official LLM SDK directly.
- Never put secrets in code; use .env. Never use Python eval() on model output.
- When something is a design decision, tell me the tradeoff so I can explain it in interviews.

## Decisions
- Keep it cheap: user wants low API spend.
- LLM: Claude via the `anthropic` SDK. Default answering model `claude-haiku-5-5` (cheapest) for all dev work and evals; one final comparison run with `claude-sonnet-5-5` in Phase 2 for the results table. No Opus unless explicitly asked.
- Scoring: numeric answers checked in code (free); LLM judge (`claude-haiku-5-5`) only for non-numeric answers. Same model as answerer, so hand-check ~20 judge verdicts.
- Use prompt caching for the stable system prompt + tool definitions.
- Embeddings: local `bge-small-en-v1.5` (384 dims) — free, no second API key.
- Data: FinanceBench top-10 companies by question count (81 questions, 33 PDFs) via `python -m analyst.ingest.financebench`.
