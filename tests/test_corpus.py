from filings_analyst.corpus import load_corpus, process_filing, save_filing


def test_process_and_roundtrip(mini_html: bytes, tmp_path) -> None:  # type: ignore[no-untyped-def]
    f = process_filing(mini_html)
    assert f.doc_id == "ACME-10-K-FY2024"
    assert f.period_end == "2024-12-31"
    assert f.financials["Revenue"]["2024-12-31"] == 5.4e9
    assert any(s.item == "1A" and s.risk_id for s in f.sentences)
    save_filing(f, tmp_path)
    (loaded,) = load_corpus(tmp_path)
    assert loaded.to_json() == f.to_json()
