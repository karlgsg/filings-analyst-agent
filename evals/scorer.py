"""Score an answer against FinanceBench's gold answer.

1. Numeric check (free): if the gold answer is a number, compare it to the first
   number in the model's direct answer within a relative tolerance.
2. LLM judge (cheap): everything else, including numeric near-misses that may
   just be formatting (e.g. "$1.6 billion" vs "$1,616 million").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from analyst.config import JUDGE_MODEL

TOLERANCE = 0.02  # 2% relative
NUMBER_RE = re.compile(r"-?\(?\$?\s*\d[\d,]*\.?\d*\)?")
NOT_FOUND_PHRASES = ("couldn't find", "could not find", "not in the provided", "cannot determine")


@dataclass
class Score:
    correct: bool
    method: str  # "numeric" | "judge" | "not_found"
    reason: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0


def parse_number(s: str) -> float | None:
    """'$1,577.00' -> 1577.0, '(370)' -> -370.0, '4.2%' -> 4.2."""
    m = NUMBER_RE.search(s)
    if not m:
        return None
    raw = m.group()
    negative = raw.startswith(("-", "("))
    digits = re.sub(r"[^\d.]", "", raw)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def gold_number(gold: str) -> float | None:
    """Only treat the gold answer as numeric if it's essentially just a number."""
    stripped = re.sub(r"[\s$%,]|USD|million|billion|times|x", "", gold, flags=re.IGNORECASE)
    if not re.fullmatch(r"-?\(?\d*\.?\d+\)?", stripped):
        return None
    return parse_number(gold)


def direct_answer(text: str) -> str:
    """The prompt asks for the direct answer first, so use the first paragraph."""
    for para in text.split("\n\n"):
        if para.strip():
            return para.strip()
    return ""


def numbers_match(expected: float, got: float, tol: float = TOLERANCE) -> bool:
    if expected == 0:
        return abs(got) < 1e-9
    # Allow unit-scale mismatches (millions vs billions vs thousands).
    return any(abs(got * s - expected) / abs(expected) <= tol for s in (1, 1000, 0.001))


class JudgeVerdict(BaseModel):
    correct: bool
    reason: str


JUDGE_PROMPT = """\
You are grading an AI's answer to a financial question against the gold answer.

Mark it CORRECT if the AI's main answer agrees with the gold answer in substance:
- Numbers may differ in formatting, units (e.g. millions vs billions), or rounding \
within about 2%.
- Extra explanation is fine as long as the main conclusion matches.
Mark it INCORRECT if the main answer differs, is missing, hedges between several \
answers, or says the information wasn't found.

Question: {question}

Gold answer: {gold}

AI answer:
{answer}"""


def judge(question: str, gold: str, answer: str, client: anthropic.Anthropic) -> Score:
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, gold=gold, answer=answer),
            }
        ],
        output_format=JudgeVerdict,
    )
    verdict = response.parsed_output
    return Score(
        correct=verdict.correct,
        method="judge",
        reason=verdict.reason,
        judge_input_tokens=response.usage.input_tokens,
        judge_output_tokens=response.usage.output_tokens,
    )


def score(question: str, gold: str, answer: str, client: anthropic.Anthropic) -> Score:
    head = direct_answer(answer)
    if any(p in head.lower() for p in NOT_FOUND_PHRASES):
        return Score(False, "not_found", "model said the answer wasn't in the filings")

    expected = gold_number(gold)
    got = parse_number(head)
    if expected is not None and got is not None and numbers_match(expected, got):
        return Score(True, "numeric", f"{got} matches {expected} within {TOLERANCE:.0%}")

    return judge(question, gold, answer, client)
