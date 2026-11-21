"""The tool-calling agent loop, written by hand so every step is visible.

Each step: send the conversation to Claude -> it replies with tool_use blocks ->
we run the tools -> append tool_result blocks -> repeat, until it calls final_answer
or hits the step cap.

Run:  uv run python -m analyst.agent.loop "What was PepsiCo's FY2022 EBITDA?"
"""

from __future__ import annotations

import argparse
import time

import anthropic

from analyst import tracing
from analyst.agent.baseline import Answer, parse_citations
from analyst.agent.prompts import AGENT_SYSTEM, FORCE_ANSWER_NOTE
from analyst.agent.tools import list_companies, run_tool, tool_definitions
from analyst.config import ANSWER_MODEL, cost_usd
from analyst.db.store import connect

MAX_STEPS = 6  # model calls per question


def _text(content) -> str:
    return "".join(b.text for b in content if b.type == "text").strip()


def _final_from_tool(args: dict) -> tuple[str, list[tuple[str, int]]]:
    citations = []
    for c in args.get("citations") or []:
        try:
            cite = (str(c["doc_name"]), int(c["page"]))
        except (KeyError, TypeError, ValueError):
            continue
        if cite not in citations:
            citations.append(cite)
    text = str(args.get("answer", "")).strip()
    if citations:
        text += "\n\nSources: " + ", ".join(f"[{d}, p.{p}]" for d, p in citations)
    return text, citations


def run_agent(
    question: str,
    conn=None,
    client: anthropic.Anthropic | None = None,
    max_steps: int = MAX_STEPS,
    company: str | None = None,
    year: int | None = None,
) -> Answer:
    """company / year (optional) restrict every search to those filings, e.g. from a UI filter."""
    client = client or anthropic.Anthropic()
    own_conn = conn is None
    conn = conn or connect()
    try:
        with tracing.observation("agent", as_type="agent", input={"question": question}) as obs:
            scope = {k: v for k, v in (("company", company), ("year", year)) if v}
            answer = _loop(question, conn, client, max_steps, scope)
            answer.extra["trace_id"] = tracing.current_trace_id()
            obs.update(
                output={"answer": answer.text, "citations": answer.citations},
                metadata={k: answer.extra[k] for k in ("steps", "finished_by")},
            )
        return answer
    finally:
        if own_conn:
            conn.close()


def _call_model(client, request: dict, step: int, log: list[dict]):
    """One Claude call, traced as a Langfuse 'generation' and timed locally."""
    with tracing.observation(
        f"claude step {step}",
        as_type="generation",
        model=request["model"],
        input=request["messages"][-1]["content"],  # just the newest turn, not the whole history
    ) as gen:
        start = time.perf_counter()
        response = client.messages.create(**request)
        seconds = time.perf_counter() - start

        u = response.usage
        tokens = {
            "input": u.input_tokens,
            "output": u.output_tokens,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
        }
        cost = cost_usd(
            request["model"],
            tokens["input"],
            tokens["output"],
            tokens["cache_write"],
            tokens["cache_read"],
        )
        gen.update(
            output=tracing.blocks_for_log(response.content),
            usage_details=tokens,
            cost_details={"total": cost},
            metadata={"stop_reason": response.stop_reason},
        )
    log.append(
        {"step": step, "kind": "llm", "seconds": round(seconds, 3), **tokens, "cost_usd": cost}
    )
    return response, tokens


def _run_tool_traced(name: str, args: dict, conn, step: int, log: list[dict]):
    """One tool call, traced as a Langfuse span and timed locally."""
    as_type = "retriever" if name == "search_filings" else "tool"
    with tracing.observation(name, as_type=as_type, input=args) as span:
        start = time.perf_counter()
        output, hits, is_error = run_tool(name, args, conn)
        seconds = time.perf_counter() - start
        span.update(
            output=[
                {"page": h.label, "score": round(h.score, 3), "preview": h.text[:300]} for h in hits
            ]
            if hits
            else output,
            level="ERROR" if is_error else None,
        )
    log.append({"step": step, "kind": name, "seconds": round(seconds, 3)})
    return output, hits, is_error


def _scope_note(scope: dict) -> str:
    where = f"{scope['company']}'s filings" if "company" in scope else "filings"
    when = f" for fiscal year {scope['year']}" if "year" in scope else ""
    return f"(Searches are restricted to {where}{when}.)\n\n"


def _loop(question, conn, client, max_steps, scope: dict | None = None) -> Answer:
    scope = scope or {}
    tools = tool_definitions(list_companies(conn))
    system = AGENT_SYSTEM.format(max_steps=max_steps)
    # The scope goes in the user turn, not the system prompt, so the cached prefix
    # (system + tools) stays identical for every question.
    content = (_scope_note(scope) if scope else "") + question
    messages: list[dict] = [{"role": "user", "content": content}]

    usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    trace: list[dict] = []  # what the agent did (tools + inputs + outputs)
    timings: list[dict] = []  # how long each model call / tool call took
    retrieved = {}  # hit id -> Hit, across all searches
    final_text, citations, finished_by = "", [], "step_limit"

    for step in range(1, max_steps + 1):
        request = {
            "model": ANSWER_MODEL,
            "max_tokens": 4000,
            "system": system,
            "tools": tools,
            "messages": messages,
            "output_config": {"effort": "medium"},
            # Cache the conversation so far: each step re-sends every earlier step,
            # and cached tokens cost ~10% of normal input.
            "cache_control": {"type": "ephemeral"},
        }
        if step == max_steps:
            request["tool_choice"] = {"type": "none"}  # last step: must answer in text

        response, tokens = _call_model(client, request, step, timings)
        for key, value in tokens.items():
            usage[key] += value

        # Append the full content (including thinking blocks) unchanged.
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "refusal":
            final_text, finished_by = "[model declined to answer]", "refusal"
            break

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:  # answered in plain text
            final_text = _text(response.content)
            citations = parse_citations(final_text)
            finished_by = "text" if step < max_steps else "step_limit"
            break

        final_call = next((b for b in tool_uses if b.name == "final_answer"), None)
        if final_call:
            final_text, citations = _final_from_tool(final_call.input)
            trace.append({"step": step, "tool": "final_answer", "input": final_call.input})
            finished_by = "final_answer"
            break

        # Run every requested tool and send all results back in ONE user message.
        results = []
        for block in tool_uses:
            args = block.input
            if block.name == "search_filings":
                args = {**args, **scope}  # enforce the filter, whatever the model asked for
            output, hits, is_error = _run_tool_traced(block.name, args, conn, step, timings)
            for h in hits:
                retrieved.setdefault(h.id, h)
            trace.append(
                {
                    "step": step,
                    "tool": block.name,
                    "input": args,
                    # For searches, log which pages came back rather than the full text.
                    "output": [h.label for h in hits] if hits else output,
                    "is_error": is_error,
                }
            )
            result = {"type": "tool_result", "tool_use_id": block.id, "content": output}
            if is_error:
                result["is_error"] = True
            results.append(result)
        if step == max_steps - 1:
            results.append({"type": "text", "text": FORCE_ANSWER_NOTE})
        messages.append({"role": "user", "content": results})

    return Answer(
        question=question,
        text=final_text or "[no answer]",
        citations=citations,
        retrieved=list(retrieved.values()),
        model=ANSWER_MODEL,
        input_tokens=usage["input"],
        output_tokens=usage["output"],
        cache_write_tokens=usage["cache_write"],
        cache_read_tokens=usage["cache_read"],
        extra={"steps": step, "finished_by": finished_by, "trace": trace, "timings": timings},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    a = run_agent(args.question)
    for t in a.extra["trace"]:
        print(f"step {t['step']}: {t['tool']}({t['input']})")
        if "output" in t:
            print(f"   -> {t['output']}")
    print("\n" + a.text)
    print("\nTimings:")
    for t in a.extra["timings"]:
        print(f"  step {t['step']} {t['kind']:15s} {t['seconds']:6.2f}s")
    print(
        f"\n{a.extra['steps']} steps, finished by {a.extra['finished_by']}; tokens: "
        f"{a.input_tokens} in, {a.cache_read_tokens} cache-read, "
        f"{a.cache_write_tokens} cache-write, {a.output_tokens} out"
    )
    url = tracing.trace_url(a.extra.get("trace_id"))
    if url:
        print(f"Trace: {url}")
    tracing.flush()


if __name__ == "__main__":
    main()
