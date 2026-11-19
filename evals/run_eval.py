"""Run the answerer on the fixed eval set, score each answer, save results.

Run:  uv run python evals/run_eval.py --config baseline
      ANSWER_MODEL=claude-sonnet-5-5 uv run python evals/run_eval.py --config baseline-sonnet
      uv run python evals/run_eval.py --config baseline --limit 3      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent))  # so `import scorer` works when run as a script

import scorer

from analyst.agent.baseline import answer
from analyst.agent.loop import run_agent
from analyst.config import ANSWER_MODEL, JUDGE_MODEL, cost_usd
from analyst.db.store import connect
from analyst.ingest.financebench import selected_questions

EVAL_DIR = Path(__file__).parent
EVAL_SET = EVAL_DIR / "eval_set.json"
RESULTS_DIR = EVAL_DIR / "results"
EVAL_SIZE = 30
SEED = 42


def load_eval_set() -> list[dict]:
    """Fixed question set, created once and committed so every config is compared fairly."""
    by_id = {q["financebench_id"]: q for q in selected_questions()}
    if not EVAL_SET.exists():
        ids = sorted(random.Random(SEED).sample(sorted(by_id), EVAL_SIZE))
        EVAL_SET.write_text(json.dumps({"seed": SEED, "ids": ids}, indent=2) + "\n")
        print(f"Created {EVAL_SET.name} with {len(ids)} questions")
    ids = json.loads(EVAL_SET.read_text())["ids"]
    return [by_id[i] for i in ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="name for this run, e.g. baseline")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument(
        "--retriever",
        choices=["vector", "hybrid", "rerank", "agent"],
        default="vector",
        help="'agent' = tool-calling loop (search_filings uses rerank internally)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="chunks given to the model")
    args = parser.parse_args()

    questions = load_eval_set()[: args.limit]
    client = anthropic.Anthropic()
    rows = []

    with connect() as conn:
        for i, q in enumerate(questions, 1):
            start = time.perf_counter()
            if args.retriever == "agent":
                a = run_agent(q["question"], conn=conn, client=client)
            else:
                a = answer(
                    q["question"],
                    conn=conn,
                    client=client,
                    retriever=args.retriever,
                    top_k=args.top_k,
                )
            latency = time.perf_counter() - start
            s = scorer.score(q["question"], q["answer"], a.text, client)

            gold_pages = {(q["doc_name"], e["evidence_page_num"] + 1) for e in q["evidence"]}
            retrieved = {(h.doc_name, h.page) for h in a.retrieved}
            rows.append(
                {
                    "id": q["financebench_id"],
                    "company": q["company"],
                    "question": q["question"],
                    "gold": q["answer"],
                    "answer": a.text,
                    "correct": s.correct,
                    "method": s.method,
                    "reason": s.reason,
                    "retrieval_hit": bool(gold_pages & retrieved),
                    "citations": a.citations,
                    "steps": a.extra.get("steps", 1),
                    "finished_by": a.extra.get("finished_by", "rag"),
                    "trace": a.extra.get("trace", []),
                    "latency_s": round(latency, 2),
                    "answer_cost_usd": cost_usd(
                        a.model,
                        a.input_tokens,
                        a.output_tokens,
                        a.cache_write_tokens,
                        a.cache_read_tokens,
                    ),
                    "judge_cost_usd": cost_usd(
                        JUDGE_MODEL, s.judge_input_tokens, s.judge_output_tokens
                    ),
                }
            )
            mark = "✓" if s.correct else "✗"
            print(f"[{i:2d}/{len(questions)}] {mark} {s.method:9s} {q['company']}: {q['answer']}")

    n = len(rows)
    summary = {
        "config": args.config,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "avg_cost_per_q_usd": round(sum(r["answer_cost_usd"] for r in rows) / n, 5),
        "total_cost_usd": round(sum(r["answer_cost_usd"] + r["judge_cost_usd"] for r in rows), 4),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "scored_by": {
            m: sum(r["method"] == m for r in rows) for m in ("numeric", "judge", "not_found")
        },
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"-limit{args.limit}" if args.limit else ""
    out = RESULTS_DIR / f"{args.config}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "results": rows}, indent=2) + "\n")

    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
