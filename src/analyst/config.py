"""Central settings, loaded from environment variables (.env)."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://analyst:analyst@localhost:5432/filings")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Models are swappable via .env (or per-run, e.g. `ANSWER_MODEL=claude-opus-5-5 uv run ...`).
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-haiku-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-5-5")
