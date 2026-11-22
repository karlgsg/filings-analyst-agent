"""EDGAR parsing tests (no network)."""

from analyst.ingest.edgar import clean_company_name, html_pages, pick_latest_10k

FILING = """<html><head><title>10-K</title></head><body>
<div style="display:none"><ix:header><ix:hidden>dei:EntityCentralIndexKey 0001045810</ix:hidden>
</ix:header></div>
<p>Page one: Item 1. Business</p>
<hr style="page-break-after: always"/>
<div><p>Page two: Revenue was <span>$60,922</span> million.</p>
<table><tr><td>Revenue</td><td>60,922</td></tr><tr><td>Cost</td><td>16,621</td></tr></table></div>
<div style="page-break-before:always"><p>Page three: Risk factors</p></div>
<p style="page-break-after:always">Page four</p><p>Page five</p>
</body></html>"""


def test_html_pages_splits_on_page_breaks_and_skips_hidden_xbrl():
    pages = html_pages(FILING)
    assert len(pages) == 4
    assert pages[0] == "Page one: Item 1. Business"  # no XBRL junk, no <title>
    assert "Revenue was $60,922 million." in pages[1]
    assert "Revenue 60,922\nCost 16,621" in pages[1]  # table rows -> lines
    # "Page four" has no break *before* it, so it shares page three; its break is after.
    assert pages[2:] == ["Page three: Risk factors\nPage four", "Page five"]


def test_html_without_page_breaks_falls_back_to_fixed_pages():
    html = "<p>" + " ".join(f"w{i}" for i in range(1500)) + "</p>"
    pages = html_pages(html)
    assert [len(p.split()) for p in pages] == [600, 600, 300]


def test_pick_latest_10k_skips_other_forms():
    subs = {
        "name": "NVIDIA CORP",
        "filings": {
            "recent": {
                "form": ["10-Q", "8-K", "10-K", "10-K"],
                "accessionNumber": ["a", "b", "0001045810-25-000023", "old"],
                "reportDate": ["2025-04-27", "", "2025-01-26", "2024-01-28"],
                "primaryDocument": ["q.htm", "k.htm", "nvda-20250126.htm", "old.htm"],
            }
        },
    }
    assert pick_latest_10k(subs) == {
        "accession": "0001045810-25-000023",
        "report_date": "2025-01-26",
        "document": "nvda-20250126.htm",
    }


def test_clean_company_name():
    assert clean_company_name("NVIDIA CORP") == "Nvidia"
    assert clean_company_name("Apple Inc.") == "Apple"
    assert clean_company_name("3M CO") == "3M"
    assert clean_company_name("MICROSOFT CORPORATION") == "Microsoft"
