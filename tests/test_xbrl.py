from filings_analyst.xbrl import extract_facts, headline_financials


def test_extracts_cover_page_and_scaled_facts(mini_html: bytes) -> None:
    facts, dei = extract_facts(mini_html)
    assert dei["dei:EntityRegistrantName"] == "Acme Robotics Inc."
    assert dei["dei:DocumentFiscalYearFocus"] == "2024"
    rev = {
        f.period_end: f.value
        for f in facts
        if f.concept == "us-gaap:Revenues" and f.duration_days and f.duration_days > 300
    }
    assert rev == {"2024-12-31": 5.4e9, "2023-12-31": 4.2e9}


def test_dimensional_facts_are_excluded(mini_html: bytes) -> None:
    facts, _ = extract_facts(mini_html)
    assert all(f.value != 999e6 for f in facts)


def test_headline_financials_annual_only_and_sign(mini_html: bytes) -> None:
    facts, _ = extract_facts(mini_html)
    table = headline_financials(facts)
    assert table["Revenue"] == {"2023-12-31": 4.2e9, "2024-12-31": 5.4e9}  # Q4 excluded
    assert table["Net income"] == {"2024-12-31": -3e8}
    assert table["Total assets"] == {"2024-12-31": 9e9}
