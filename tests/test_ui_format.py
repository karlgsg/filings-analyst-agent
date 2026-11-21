from analyst.ui.format import describe_step, md_safe, strip_sources


def test_md_safe_escapes_dollars_so_streamlit_doesnt_render_math():
    assert md_safe("$1,577 million vs $4,000") == r"\$1,577 million vs \$4,000"


def test_strip_sources_removes_trailing_sources_line():
    assert strip_sources("$5M.\n\nBecause X.\n\nSources: [A, p.1]") == "$5M.\n\nBecause X."
    assert strip_sources("No sources here.") == "No sources here."


def test_describe_step():
    search = {
        "step": 1,
        "tool": "search_filings",
        "input": {"query": "capex", "company": "3M"},
        "output": ["[3M_2018_10K, p.60]"],
    }
    assert describe_step(search) == (
        "search_filings(query='capex', company='3M') → 1 pages: [3M_2018_10K, p.60]"
    )
    calc = {"step": 2, "tool": "calculate", "input": {"expression": "1+1"}, "output": "2"}
    assert describe_step(calc) == "calculate(expression='1+1') → 2"
    assert describe_step({"step": 3, "tool": "final_answer", "input": {}}) == "final_answer"
