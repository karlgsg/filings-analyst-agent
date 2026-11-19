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

# USD per 1M tokens (input, output). Update if pricing changes.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """input_tokens is the uncached part; cache writes cost 1.25x input, reads 0.1x."""
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    total = (
        input_tokens * price_in
        + cache_write_tokens * price_in * 1.25
        + cache_read_tokens * price_in * 0.10
        + output_tokens * price_out
    )
    return round(total / 1_000_000, 6)
