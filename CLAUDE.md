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
- LLM: Claude via the `anthropic` SDK. Answering model `claude-opus-5-5`; LLM judge `claude-haiku-5-5` (different model to reduce self-grading bias). Try `claude-sonnet-5-5` in Phase 2 as a cost experiment.
- Embeddings: local `bge-small-en-v1.5` (384 dims) — free, no second API key.
- Data: FinanceBench top-10 companies by question count (81 questions, 33 PDFs) via `python -m analyst.ingest.financebench`.
