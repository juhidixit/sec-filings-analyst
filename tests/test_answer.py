from filings_analyst.answer import Analyst
from filings_analyst.verify import Verdict


def test_corpus_invariants(corpus) -> None:  # type: ignore[no-untyped-def]
    assert len(corpus) == 9
    for f in corpus:
        assert len(f.risk_factors) >= 20, f.doc_id
        assert "Revenue" in f.financials, f.doc_id
        assert f.fiscal_year >= 2022


def test_structured_route_uses_ixbrl(analyst: Analyst) -> None:
    ans = analyst.ask("What was NVIDIA's revenue in fiscal 2025?")
    assert ans.route == "structured"
    assert "$130.5 billion" in ans.answer
    assert all(c.verdict == Verdict.SUPPORTED for c in ans.checks)


def test_restatement_is_surfaced(analyst: Analyst) -> None:
    ans = analyst.ask("What was NVIDIA's diluted EPS in fiscal 2023?")
    assert "$1.74" in ans.answer and "$0.17" in ans.answer
    assert ans.notes and "Restatement" in ans.notes[0]


def test_narrative_answer_is_cited_and_verified(analyst: Analyst) -> None:
    ans = analyst.ask("Where is Tesla building its next Gigafactory?")
    assert ans.route == "narrative"
    assert ans.scope == ["TSLA-10-K-FY2024"]
    assert any("Monterrey" in s.text for s in ans.sources)
    assert ans.checks and all(c.citations for c in ans.checks)
    assert ans.supported_ratio == 1.0


def test_scope_by_year(analyst: Analyst) -> None:
    assert analyst.retriever.scope("What did Citi say about LIBOR in 2022?") == ["C-10-K-FY2022"]


def test_strict_mode_drops_unsupported(analyst: Analyst) -> None:
    class Liar:
        name = "liar"

        def generate(self, question, sources):  # type: ignore[no-untyped-def]
            return f"{sources[0].text} [1] Tesla plans a factory on Mars by 2026 [1]."

    a = Analyst(list(analyst.filings.values()), Liar())
    ans = a.ask("Where is Tesla building its next Gigafactory?", strict=True)
    assert ans.removed_claims and "Mars" in ans.removed_claims[0]
    assert "Mars" not in ans.answer
