import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "evals"))

from scorer import direct_answer, gold_number, numbers_match, parse_number


def test_parse_number_formats():
    assert parse_number("$1,577.00") == 1577.0
    assert parse_number("about 4.2%") == 4.2
    assert parse_number("(370)") == -370.0
    assert parse_number("no numbers here") is None


def test_gold_number_only_for_numeric_answers():
    assert gold_number("$1577.00") == 1577.0
    assert gold_number("4.2%") == 4.2
    assert gold_number("There are none") is None
    assert gold_number("Yes, because revenue grew 5% in 2022") is None


def test_numbers_match_tolerance_and_units():
    assert numbers_match(1616.0, 1615.9)
    assert numbers_match(1616.0, 1.616)  # billions vs millions
    assert not numbers_match(2.8, 2.7)  # 3.6% off
    assert numbers_match(0, 0)


def test_direct_answer_is_first_paragraph():
    assert direct_answer("\n\n**Answer: $5M**\n\nBecause...") == "**Answer: $5M**"


def test_not_found_only_when_answer_leads_with_it():
    from scorer import says_not_found

    assert says_not_found("**Direct answer:** I couldn’t find this in the provided filings.")
    assert says_not_found("I could not find the FY2015 figures.\n\nDetails...")
    # A real answer that mentions a gap later must NOT be auto-failed.
    assert not says_not_found(
        "Amcor made three acquisitions in FY2023: A, B and C. "
        "I could not find any FY2021 acquisitions in the indexed filings."
    )
