from analyst import tracing


def test_tracing_is_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setattr(tracing, "_client", None)

    assert not tracing.enabled()
    with tracing.trace_attributes(session_id="s"), tracing.observation("x") as obs:
        obs.update(output="anything")
        assert tracing.current_trace_id() is None
    tracing.score(None, "correct", 1.0)
    tracing.flush()
    assert tracing._client is None  # never created a Langfuse client
