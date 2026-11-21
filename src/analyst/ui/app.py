"""Streamlit chat UI. It only talks to the API over HTTP (UI -> API -> agent).

Run (two terminals):
    uv run uvicorn analyst.api.main:app
    uv run streamlit run src/analyst/ui/app.py

Streamlit re-runs this whole script on every interaction; st.session_state is what
survives between runs (here: the chat history).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

from analyst.ui.format import describe_step, md_safe, strip_sources

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
EXAMPLES = [
    "What was 3M's FY2018 capital expenditure?",
    "Is Boeing's gross margin improving between FY2021 and FY2022?",
    "What is Amcor's FY2023 quick ratio?",
]

st.set_page_config(page_title="Filings Analyst", page_icon="📄", layout="centered")


@st.cache_data(ttl=600)
def load_companies() -> list[dict]:
    r = httpx.get(f"{API_URL}/companies", timeout=10)
    r.raise_for_status()
    return r.json()


def ask(question: str, company: str | None, year: int | None) -> dict:
    payload = {"question": question, "company": company, "year": year}
    r = httpx.post(f"{API_URL}/ask", json=payload, timeout=120)
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text)
        raise RuntimeError(detail if isinstance(detail, str) else "Invalid question.")
    return r.json()


def show_answer(result: dict, show_steps: bool) -> None:
    st.markdown(md_safe(strip_sources(result["answer"])))

    if result["citations"]:
        st.caption("Sources (click to see the passage)")
        for c in result["citations"]:
            with st.expander(f"📄 {c['doc_name']}, page {c['page']}"):
                st.text(c["text"] or "(passage not in the retrieved excerpts)")

    if show_steps and result["steps"]:
        with st.expander(f"Reasoning steps ({len(result['steps'])})", expanded=True):
            for s in result["steps"]:
                st.markdown(f"**Step {s['step']}.** `{describe_step(s)}`")

    meta = f"{result['model']} · ${result['cost_usd']:.4f} · {result['latency_s']:.1f}s"
    if result.get("trace_url"):
        meta += f" · [trace]({result['trace_url']})"
    st.caption(md_safe(meta))


# ---- sidebar: filters ----

st.sidebar.header("Filters")
try:
    companies = load_companies()
except httpx.HTTPError:
    st.error(
        f"Can't reach the API at {API_URL}. Start it with `uv run uvicorn analyst.api.main:app`."
    )
    st.stop()

names = ["All companies"] + [c["company"] for c in companies]
choice = st.sidebar.selectbox("Company", names)
company = None if choice == "All companies" else choice
years = next((c["years"] for c in companies if c["company"] == company), [])
year_choice = st.sidebar.selectbox("Filing year", ["Any"] + years, disabled=not company)
year = None if year_choice == "Any" else year_choice
show_steps = st.sidebar.toggle("Show reasoning steps", value=False)
st.sidebar.caption(f"{len(companies)} companies indexed. Answers cite the exact filing page.")

# ---- main: chat ----

st.title("📄 Filings Analyst")
st.caption("Ask about 10-Ks, 10-Qs and earnings releases. Every answer cites its sources.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, result-or-error)

for question, result in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(md_safe(question))
    with st.chat_message("assistant"):
        if isinstance(result, str):
            st.error(result)
        else:
            show_answer(result, show_steps)

question = st.chat_input("Ask a question about a company's filings")
if not st.session_state.history and not question:
    st.markdown("**Try:**")
    for ex in EXAMPLES:
        if st.button(ex):
            question = ex

if question:
    with st.chat_message("user"):
        st.markdown(md_safe(question))
    with st.chat_message("assistant"):
        with st.spinner("Searching filings..."):
            try:
                result = ask(question, company, year)
            except (RuntimeError, httpx.HTTPError) as e:
                result = f"Sorry, that failed: {e}"
        if isinstance(result, str):
            st.error(result)
        else:
            show_answer(result, show_steps)
    st.session_state.history.append((question, result))
    st.rerun()  # redraw from history (hides the example buttons)
