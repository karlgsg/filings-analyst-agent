# Filings Analyst Agent

AI agent that answers questions over SEC filings with page-level citations. See [PLAN.md](PLAN.md).

## Setup

```bash
cp .env.example .env               # then fill in API keys
docker compose up -d               # start Postgres + pgvector
uv sync                            # install Python deps
uv run python -m analyst.db.check  # verify DB connection
```
