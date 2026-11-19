"""Tests for the agent loop logic, using a fake client (no API calls, no database)."""

from types import SimpleNamespace as NS

from analyst.agent import loop
from analyst.retrieval.vector import Hit


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


def response(*blocks, stop="tool_use"):
    usage = NS(
        input_tokens=100, output_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    return NS(content=list(blocks), stop_reason=stop, usage=usage)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        # Snapshot: the loop keeps appending to the same messages list after sending.
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_run_tool(name, args, conn):
    if name == "search_filings":
        return "[DOC, p.1]\nRevenue was 100.", [Hit(1, "DOC", 1, "Revenue was 100.", 1.0)], False
    return "42", [], False


def setup(monkeypatch):
    monkeypatch.setattr(loop, "list_companies", lambda conn: ["ACME"])
    monkeypatch.setattr(loop, "run_tool", fake_run_tool)


def test_search_then_calculate_then_final_answer(monkeypatch):
    setup(monkeypatch)
    client = FakeClient(
        [
            response(tool_use("t1", "search_filings", {"query": "revenue"})),
            response(tool_use("t2", "calculate", {"expression": "40 + 2"})),
            response(
                tool_use(
                    "t3",
                    "final_answer",
                    {"answer": "42", "citations": [{"doc_name": "DOC", "page": 1}]},
                )
            ),
        ]
    )
    a = loop.run_agent("q?", conn=object(), client=client)
    assert a.text.startswith("42")
    assert a.citations == [("DOC", 1)]
    assert a.extra["finished_by"] == "final_answer"
    assert [t["tool"] for t in a.extra["trace"]] == ["search_filings", "calculate", "final_answer"]
    assert [h.id for h in a.retrieved] == [1]
    assert a.input_tokens == 300
    # Each tool_result must reference the tool_use id it answers.
    last_messages = client.requests[-1]["messages"]
    assert last_messages[-1]["content"][0]["tool_use_id"] == "t2"


def test_parallel_tool_calls_answered_in_one_message(monkeypatch):
    setup(monkeypatch)
    client = FakeClient(
        [
            response(
                tool_use("a", "search_filings", {"query": "x"}),
                tool_use("b", "search_filings", {"query": "y"}),
            ),
            response(tool_use("c", "final_answer", {"answer": "done", "citations": []})),
        ]
    )
    loop.run_agent("q?", conn=object(), client=client)
    tool_results = client.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in tool_results] == ["a", "b"]


def test_step_cap_forces_text_answer(monkeypatch):
    setup(monkeypatch)
    searching = [response(tool_use(f"t{i}", "search_filings", {"query": "x"})) for i in range(2)]
    text_reply = response(NS(type="text", text="Revenue was 100 [DOC, p.1]."), stop="end_turn")
    client = FakeClient([*searching, text_reply])
    a = loop.run_agent("q?", conn=object(), client=client, max_steps=3)
    assert len(client.requests) == 3
    assert client.requests[-1]["tool_choice"] == {"type": "none"}
    assert "tool_choice" not in client.requests[0]
    # The step before the last tells the model to stop using tools.
    assert client.requests[-1]["messages"][-1]["content"][-1]["type"] == "text"
    assert a.citations == [("DOC", 1)]
    assert a.extra["finished_by"] == "step_limit"
