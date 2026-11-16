import httpx
import pytest

from filings_analyst.edgar import EdgarClient


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "@" in request.headers["User-Agent"]
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA"}})
        if "submissions" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "filings": {
                        "recent": {
                            "form": ["10-Q", "10-K", "10-K"],
                            "accessionNumber": ["0001-1", "0001045810-25-000023", "0001045810-24-000029"],
                            "filingDate": ["2025-05-28", "2025-02-26", "2024-02-21"],
                            "reportDate": ["2025-04-27", "2025-01-26", "2024-01-28"],
                            "primaryDocument": ["q.htm", "nvda-20250126.htm", "nvda-20240128.htm"],
                        }
                    }
                },
            )
        return httpx.Response(200, content=b"<html></html>")

    return httpx.MockTransport(handler)


def test_requires_contact_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError):
        EdgarClient()


def test_lists_10k_filings_and_builds_urls(tmp_path) -> None:  # type: ignore[no-untyped-def]
    c = EdgarClient("Test Runner test@example.com", min_interval=0, transport=_transport())
    refs = c.list_filings("nvda", limit=5)
    assert [r.report_date for r in refs] == ["2025-01-26", "2024-01-28"]
    assert refs[0].url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000023/nvda-20250126.htm"
    )
    path = c.download(refs[0], tmp_path)
    assert path.read_bytes() == b"<html></html>"


def test_find_filing_by_period() -> None:
    c = EdgarClient("Test Runner test@example.com", min_interval=0, transport=_transport())
    assert c.find_filing("NVDA", "2024-01-28").primary_document == "nvda-20240128.htm"
    with pytest.raises(LookupError):
        c.find_filing("NVDA", "2019-01-27")
