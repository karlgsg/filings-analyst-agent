"""Score an answer against FinanceBench's gold answer.

1. Numeric check (free): if the gold answer is a number, compare it to the first
   value in the model's direct answer (skipping years/dates) within a relative tolerance.
2. LLM judge (cheap): everything else, including numeric near-misses that may
   just be formatting (e.g. "$1.6 billion" vs "$1,616 million").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import anthropic
from pydantic import BaseModel

from analyst import tracing
from analyst.config import JUDGE_MODEL, cost_usd

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


# Digits that are labels, not values: dates, fiscal periods, form and company names.
# Without this, "In FY2022, revenue was $5.2B" would be read as the number 2022.
NOT_VALUES_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}\b"  # June 30
    r"|\b(?:FY|Q[1-4])\s?'?\d{0,4}\b"  # FY2022, FY'23, Q2
    r"|'\d{2}\b"  # Jun'23
    r"|\b\d{1,2}-?[KQ]\b"  # 10-K, 10-Q, 8-K
    r"|(?<!\$)\b3M\b"  # the company 3M (but not "$3M")
    r"|(?<![$\d.,])\b(?:19|20)\d{2}\b(?![\d,%]|\.\d)",  # bare years like 2018
    flags=re.IGNORECASE,
)


def answer_number(text: str) -> float | None:
    """The first real value in the answer, skipping years, dates and names."""
    return parse_number(NOT_VALUES_RE.sub(" ", text))


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
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer)
    with tracing.observation("judge", as_type="generation", model=JUDGE_MODEL, input=prompt) as gen:
        response = client.messages.parse(
            model=JUDGE_MODEL,
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
            output_format=JudgeVerdict,
        )
        verdict = response.parsed_output
        u = response.usage
        gen.update(
            output=verdict.model_dump(),
            usage_details={"input": u.input_tokens, "output": u.output_tokens},
            cost_details={"total": cost_usd(JUDGE_MODEL, u.input_tokens, u.output_tokens)},
        )
    return Score(
        correct=verdict.correct,
        method="judge",
        reason=verdict.reason,
        judge_input_tokens=response.usage.input_tokens,
        judge_output_tokens=response.usage.output_tokens,
    )


def says_not_found(answer: str) -> bool:
    """True only if the answer *leads* with 'I couldn't find it'. A correct answer that
    later says 'I could not find any FY2021 deals' must still be scored normally."""
    head = direct_answer(answer).replace("\u2019", "'").lower()  # curly apostrophes
    first_sentence = re.split(r"(?<=[.!?])\s", head, maxsplit=1)[0]
    return any(p in first_sentence for p in NOT_FOUND_PHRASES)


def score(question: str, gold: str, answer: str, client: anthropic.Anthropic) -> Score:
    head = direct_answer(answer)
    if says_not_found(answer):
        return Score(False, "not_found", "model said the answer wasn't in the filings")

    expected = gold_number(gold)
    got = answer_number(head)
    if expected is not None and got is not None and numbers_match(expected, got):
        return Score(True, "numeric", f"{got} matches {expected} within {TOLERANCE:.0%}")

    return judge(question, gold, answer, client)
