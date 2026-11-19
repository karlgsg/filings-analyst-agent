"""Optional Langfuse tracing.

If LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY aren't set, every function here is a
cheap no-op, so tests, CI and anyone cloning the repo don't need a Langfuse account.
The rest of the code only talks to this module, never to Langfuse directly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import analyst.config  # noqa: F401  (loads .env before we read the keys)

_client = None


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _keep_span(span) -> bool:
    """The Anthropic SDK emits its own OpenTelemetry span for every API call
    ('anthropic.messages.create'). We already log each call ourselves, with a readable
    name and our cost, so drop the SDK's copy to avoid duplicates and double-counted cost."""
    from langfuse import is_default_export_span

    return is_default_export_span(span) and not span.name.startswith("anthropic.")


def _get_client():
    global _client
    if _client is None and enabled():
        from langfuse import Langfuse

        # Reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / BASE_URL from the environment.
        _client = Langfuse(should_export_span=_keep_span)
    return _client


class _NoOp:
    """Stands in for a Langfuse observation when tracing is off."""

    def update(self, **kwargs: Any) -> _NoOp:
        return self


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """A timed step in the trace: as_type is 'span', 'generation' (an LLM call), 'tool',
    'retriever', 'agent', ... Nested `with` blocks become nested steps."""
    client = _get_client()
    if client is None:
        yield _NoOp()
        return
    with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
        yield obs


@contextmanager
def trace_attributes(**kwargs: Any) -> Iterator[None]:
    """Attach session_id / tags / metadata / trace_name to everything traced inside."""
    if _get_client() is None:
        yield
        return
    from langfuse import propagate_attributes

    with propagate_attributes(**kwargs):
        yield


def current_trace_id() -> str | None:
    client = _get_client()
    return client.get_current_trace_id() if client else None


def trace_url(trace_id: str | None) -> str | None:
    client = _get_client()
    return client.get_trace_url(trace_id=trace_id) if client and trace_id else None


def score(trace_id: str | None, name: str, value: float, comment: str | None = None) -> None:
    client = _get_client()
    if client and trace_id:
        client.create_score(trace_id=trace_id, name=name, value=value, comment=comment)


def flush() -> None:
    """Send buffered traces. Call before a script exits."""
    client = _get_client()
    if client:
        client.flush()


def blocks_for_log(content) -> list[dict]:
    """Model output as plain dicts for the trace, minus thinking blocks (large, and empty
    text by default anyway)."""
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "name": b.name, "input": b.input})
    return out
