from analyst.retrieval.keyword import clean_query


def test_fiscal_years_get_bare_year_added():
    assert "2018" in clean_query("What was FY2018 capex?").split()
    assert "2022" in clean_query("FY22 revenue").split()


def test_instruction_words_removed_content_kept():
    q = clean_query("Answer in units of percents and round to one decimal place: Best Buy margin?")
    words = q.lower().split()
    for gone in ("answer", "units", "percents", "round", "decimal", "place"):
        assert gone not in words
    for kept in ("best", "buy", "margin"):
        assert kept in words
