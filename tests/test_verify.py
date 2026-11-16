import pytest

from filings_analyst.verify import Verdict, check_claim, extract_quantities, split_claims, verify_answer

EV = (
    "Net cash used for investing activities increased by $14.4 billion in fiscal 2025 relative to "
    "fiscal 2024 primarily due to the increase in capital expenditures."
)


def test_quantity_normalisation() -> None:
    qs = extract_quantities("revenue of $130,497 million, up 114% to 2.5 billion units in 2024")
    assert [(q.kind, q.value) for q in qs] == [
        ("money", 130_497e6),
        ("percent", 114.0),
        ("number", 2.5e9),
        ("year", 2024.0),
    ]
    assert extract_quantities("a loss of -$19.7 billion")[0].value == pytest.approx(-19.7e9)


@pytest.mark.parametrize(
    "claim",
    [
        "Net cash used for investing activities increased by $14.4 billion in fiscal 2025.",
        "Net cash used for investing activities increased by $14,400 million in fiscal 2025.",
        "Oracle said cash used for investing activities increased in fiscal 2025 due to capital expenditures.",
    ],
)
def test_faithful_claims_are_supported(claim: str) -> None:
    res = check_claim(claim, [EV], aliases={"Oracle", "Oracle Corporation"})
    assert res.verdict == Verdict.SUPPORTED, res.reasons


@pytest.mark.parametrize(
    ("claim", "reason"),
    [
        ("Net cash used for investing activities increased by $16.1 billion in fiscal 2025.", "number"),
        ("Net cash used for investing activities increased by $14.4 billion in fiscal 2023.", "number"),
        ("Net cash used for investing activities decreased by $14.4 billion in fiscal 2025.", "direction"),
        ("Net cash used for investing activities did not increase in fiscal 2025.", "negation"),
        ("Microsoft said net cash used for investing activities increased by $14.4 billion.", "entity"),
        ("Cloud license revenue grew strongly because of AI demand in Europe.", "terms"),
    ],
)
def test_unsupported_claims_are_flagged_with_reason(claim: str, reason: str) -> None:
    res = check_claim(claim, [EV], aliases={"Oracle"})
    assert res.verdict == Verdict.UNSUPPORTED
    assert any(reason in r for r in res.reasons), res.reasons


def test_no_evidence_means_no_citation() -> None:
    assert check_claim("Anything.", []).verdict == Verdict.NO_CITATION


def test_split_claims_reattaches_leading_citations() -> None:
    claims = split_claims("Revenue rose. [1] Margins fell [2, 3]. Costs were flat.")
    assert claims == [("Revenue rose.", [1]), ("Margins fell.", [2, 3]), ("Costs were flat.", [])]


def test_verify_answer_end_to_end() -> None:
    out = verify_answer("Investing cash increased by $14.4 billion in fiscal 2025 [1].", {1: EV})
    assert [c.verdict for c in out] == [Verdict.SUPPORTED]
