from filings_analyst.diff import diff_risk_factors, tag_themes, theme_intensity, word_diff
from filings_analyst.risk import RiskFactor


def rf(i: int, heading: str, body: str) -> RiskFactor:
    return RiskFactor(str(i), "General", heading, body, i)


PREV = [
    rf(
        0,
        "We depend on key suppliers.",
        "Our chips come from one foundry in Taiwan. A disruption could halt supply.",
    ),
    rf(
        1,
        "The COVID-19 pandemic could hurt demand.",
        "Lockdowns reduced store traffic. Remote work changed demand.",
    ),
    rf(2, "Competition could reduce our margins.", "Competitors may cut prices. We may lose market share."),
]
CURR = [
    rf(
        0,
        "We depend on key suppliers.",
        "Our chips come from one foundry in Taiwan. A disruption could halt supply. "
        "New export controls restrict shipments to China.",
    ),
    rf(2, "Competition could reduce our margins.", "Competitors may cut prices. We may lose market share."),
    rf(3, "Generative AI regulation is uncertain.", "New AI laws could restrict how our models are trained."),
]


def test_statuses() -> None:
    d = diff_risk_factors(PREV, CURR)
    status = {c.heading: c.status for c in d.changes}
    assert status["We depend on key suppliers."] == "modified"
    assert status["Competition could reduce our margins."] == "unchanged"
    assert status["Generative AI regulation is uncertain."] == "new"
    assert status["The COVID-19 pandemic could hurt demand."] == "removed"
    assert d.counts() == {"new": 1, "removed": 1, "modified": 1, "unchanged": 1}


def test_modified_lists_only_added_sentence() -> None:
    d = diff_risk_factors(PREV, CURR)
    mod = next(c for c in d.changes if c.status == "modified")
    assert mod.added_sentences == ["New export controls restrict shipments to China."]
    assert mod.removed_sentences == []


def test_identical_filings_are_all_unchanged() -> None:
    assert diff_risk_factors(PREV, PREV).counts()["unchanged"] == len(PREV)


def test_themes() -> None:
    assert "Export controls & China" in tag_themes("New export controls restrict shipments to China.")
    assert theme_intensity(CURR)["AI & machine learning"] > 0


def test_reworded_sentence_is_an_edit_not_add_remove() -> None:
    prev = [rf(0, "Suppliers.", "We buy servo motors from two suppliers in Japan and Korea today.")]
    curr = [rf(0, "Suppliers.", "We buy servo motors from three suppliers in Japan and Korea today.")]
    (c,) = diff_risk_factors(prev, curr).changes
    assert c.status == "modified"
    assert not c.added_sentences and not c.removed_sentences
    assert len(c.edited) == 1
    assert ("+", "three") in word_diff(*c.edited[0])
