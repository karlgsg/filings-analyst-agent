from analyst.agent.baseline import parse_citations


def test_parse_citations_dedupes_and_keeps_order():
    text = (
        "Capex was $1,577M [3M_2018_10K, p.60]. Also see [AMD_2022_10K, p.12] "
        "and again [3M_2018_10K, p.60]."
    )
    assert parse_citations(text) == [("3M_2018_10K", 60), ("AMD_2022_10K", 12)]


def test_parse_citations_handles_names_with_dashes():
    text = "[PEPSICO_2023_8K_dated-2023-05-05, p.2]"
    assert parse_citations(text) == [("PEPSICO_2023_8K_dated-2023-05-05", 2)]


def test_no_citations():
    assert parse_citations("I couldn't find this in the provided filings.") == []
