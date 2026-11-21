"""Run the answerer on the fixed eval set, score each answer, save results.

Run:  uv run python evals/run_eval.py --config baseline
      ANSWER_MODEL=claude-sonnet-5-5 uv run python evals/run_eval.py --config baseline-sonnet
      uv run python evals/run_eval.py --config baseline --limit 3      # quick smoke test
      uv run python evals/run_eval.py --config agent-full --retriever agent --full  # all 150
      ... add --resume to continue a run that crashed (e.g. out of API credits)
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

from analyst import tracing
from analyst.agent.baseline import answer
from analyst.agent.loop import run_agent
from analyst.config import ANSWER_MODEL, JUDGE_MODEL, cost_usd
from analyst.db.store import connect
from analyst.ingest.financebench import load_questions, selected_questions

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


def load_checkpoint(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def time_breakdown(rows: list[dict]) -> dict:
    """Average seconds per question spent in model calls vs each tool (agent runs only)."""
    totals: dict[str, float] = {}
    for r in rows:
        for t in r["timings"]:
            totals[t["kind"]] = totals.get(t["kind"], 0.0) + t["seconds"]
    return {kind: round(total / len(rows), 2) for kind, total in sorted(totals.items())}


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
    parser.add_argument(
        "--full",
        action="store_true",
        help="all 150 FinanceBench questions instead of the 30-question dev set",
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip questions already saved by a crashed run"
    )
    args = parser.parse_args()

    questions = (load_questions() if args.full else load_eval_set())[: args.limit]
    client = anthropic.Anthropic()

    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"-limit{args.limit}" if args.limit else ""
    out = RESULTS_DIR / f"{args.config}{suffix}.json"
    # Each row is appended here the moment it's scored, so a crash mid-run (rate limit,
    # credits running out) doesn't throw away answers we already paid for.
    checkpoint = out.with_suffix(".partial.jsonl")
    rows = load_checkpoint(checkpoint) if args.resume else []
    if not args.resume:
        checkpoint.unlink(missing_ok=True)
    done = {r["id"] for r in rows}
    if done:
        print(f"Resuming: {len(done)} questions already done")
    run_at = datetime.now(UTC)
    # Groups this run's traces together in Langfuse's "Sessions" view.
    session_id = f"{args.config}-{run_at:%Y%m%d-%H%M%S}"

    with connect() as conn:
        for i, q in enumerate(questions, 1):
            if q["financebench_id"] in done:
                continue
            with (
                tracing.trace_attributes(
                    session_id=session_id,
                    trace_name=f"eval {q['financebench_id']}",
                    tags=["eval", args.config, ANSWER_MODEL],
                    metadata={"financebench_id": q["financebench_id"], "company": q["company"]},
                ),
                tracing.observation("eval question", input={"question": q["question"]}) as obs,
            ):
                trace_id = tracing.current_trace_id()
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
                obs.update(output={"answer": a.text, "gold": q["answer"], "correct": s.correct})

            # Scores show up on the trace, so you can filter Langfuse to the wrong answers.
            tracing.score(trace_id, "correct", 1.0 if s.correct else 0.0, comment=s.reason)

            gold_pages = {(q["doc_name"], e["evidence_page_num"] + 1) for e in q["evidence"]}
            retrieved = {(h.doc_name, h.page) for h in a.retrieved}
            row = {
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
                "timings": a.extra.get("timings", []),
                "trace_id": trace_id,
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
            rows.append(row)
            with checkpoint.open("a") as f:
                f.write(json.dumps(row) + "\n")
            mark = "✓" if s.correct else "✗"
            print(
                f"[{i:2d}/{len(questions)}] {mark} {s.method:9s} {q['company']}: {q['answer']}",
                flush=True,  # show progress immediately even when output goes to a log file
            )

    n = len(rows)
    summary = {
        "config": args.config,
        "retriever": args.retriever,
        "top_k": args.top_k,
        "eval_set": "full" if args.full else "dev",
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "run_at": run_at.isoformat(timespec="seconds"),
        "langfuse_session": session_id if tracing.enabled() else None,
        "n": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows) / n, 3),
        "avg_cost_per_q_usd": round(sum(r["answer_cost_usd"] for r in rows) / n, 5),
        "total_cost_usd": round(sum(r["answer_cost_usd"] + r["judge_cost_usd"] for r in rows), 4),
        "p50_latency_s": round(statistics.median(r["latency_s"] for r in rows), 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "time_breakdown_s": time_breakdown(rows),
        "scored_by": {
            m: sum(r["method"] == m for r in rows) for m in ("numeric", "judge", "not_found")
        },
    }

    out.write_text(json.dumps({"summary": summary, "results": rows}, indent=2) + "\n")
    checkpoint.unlink()

    tracing.flush()
    print("\n" + json.dumps(summary, indent=2))
    print(f"Saved {out.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
