from filings_analyst.parse import html_to_blocks, is_page_noise, segment_items
from filings_analyst.risk import split_risk_factors


def test_blocks_skip_hidden_header_and_page_numbers(mini_html: bytes) -> None:
    texts = [b.text for b in html_to_blocks(mini_html)]
    assert not any(t == "Acme Robotics Inc." for t in texts)  # hidden ix:header
    assert "12" not in texts and "Table of Contents" not in texts


def test_paragraph_split_by_page_break_is_rejoined(mini_html: bytes) -> None:
    texts = [b.text for b in html_to_blocks(mini_html)]
    assert any("halt production for several weeks and reduce revenue" in t for t in texts)


def test_items_use_body_not_table_of_contents(mini_html: bytes) -> None:
    sections = segment_items(html_to_blocks(mini_html))
    assert {"1", "1A", "7", "8"} <= set(sections)
    assert "servo motors" in sections["1A"].text
    assert "Revenue increased" in sections["7"].text


def test_risk_factors_from_bold_and_run_in_headings(mini_html: bytes) -> None:
    sections = segment_items(html_to_blocks(mini_html))
    factors = split_risk_factors(sections["1A"], min_body_chars=50)
    assert [f.heading for f in factors] == [
        "We depend on a small number of suppliers for servo motors.",
        "Cyberattacks could disrupt our factories.",
    ]
    assert all(f.category == "Risks Related to Our Operations" for f in factors)
    assert factors[1].body.startswith("Attackers have targeted")


def test_page_noise_patterns() -> None:
    assert is_page_noise("37")
    assert is_page_noise("Table of Contents")
    assert is_page_noise("Apple Inc. | 2024 Form 10-K | 17")
    assert not is_page_noise("Revenue grew 37% in 2024.")
