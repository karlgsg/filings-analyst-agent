"""API tests with the agent and database swapped out (no API calls, no Postgres)."""

import anthropic
import httpx
import pytest
from fastapi.testclient import TestClient

from analyst.agent.baseline import Answer
from analyst.api import main
from analyst.retrieval.vector import Hit


def fake_answer(question, conn, client, company=None, year=None):
    return Answer(
        question=question,
        text="$1,577 million\n\nSources: [3M_2018_10K, p.60]",
        citations=[("3M_2018_10K", 60)],
        retrieved=[Hit(1, "3M_2018_10K", 60, "Purchases of PP&E (1,577)", 9.1)],
        model="claude-haiku-5-5",
        input_tokens=1000,
        output_tokens=100,
        extra={
            "trace_id": "abc123",
            "trace": [
                {
                    "step": 1,
                    "tool": "search_filings",
                    "input": {"query": "capex"},
                    "output": ["[3M_2018_10K, p.60]"],
                    "is_error": False,
                },
                {"step": 2, "tool": "final_answer", "input": {"answer": "$1,577 million"}},
            ],
        },
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "run_agent", fake_answer)
    main.app.dependency_overrides[main.get_conn] = lambda: object()
    # No `with TestClient(...)`, so the lifespan (model warm-up) doesn't run.
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_ask_returns_answer_citations_and_steps(client):
    r = client.post("/ask", json={"question": "What was 3M's FY2018 capex?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"].startswith("$1,577 million")
    assert body["citations"] == [
        {"doc_name": "3M_2018_10K", "page": 60, "text": "Purchases of PP&E (1,577)"}
    ]
    assert [s["tool"] for s in body["steps"]] == ["search_filings", "final_answer"]
    assert body["trace_id"] == "abc123"
    assert body["cost_usd"] > 0


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_model_rate_limit_becomes_503(client, monkeypatch):
    def rate_limited(*args, **kwargs):
        response = httpx.Response(429, request=httpx.Request("POST", "https://api"))
        raise anthropic.RateLimitError("slow down", response=response, body=None)

    monkeypatch.setattr(main, "run_agent", rate_limited)
    r = client.post("/ask", json={"question": "What was 3M's FY2018 capex?"})
    assert r.status_code == 503


def test_ask_passes_filters_to_the_agent(client, monkeypatch):
    calls = []

    def recording(question, conn, client, company=None, year=None):
        calls.append((company, year))
        return fake_answer(question, conn, client)

    monkeypatch.setattr(main, "run_agent", recording)
    r = client.post("/ask", json={"question": "Revenue?", "company": "3M", "year": 2018})
    assert r.status_code == 200
    assert calls == [("3M", 2018)]
