from fastapi.testclient import TestClient

from filings_analyst.api import app

client = TestClient(app)


def test_health_and_filings() -> None:
    assert client.get("/health").json()["filings"] == 9
    tickers = {f["ticker"] for f in client.get("/filings").json()}
    assert {"NVDA", "ORCL", "C", "JPM", "AAPL", "TSLA"} == tickers


def test_ask() -> None:
    r = client.post("/ask", json={"question": "What was Tesla's revenue in 2024?"})
    body = r.json()
    assert r.status_code == 200
    assert body["route"] == "structured" and body["supported_ratio"] == 1.0


def test_ask_validates_input() -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_diff_and_404() -> None:
    body = client.get("/diff/NVDA").json()
    assert body["prev"] == "NVDA FY2023" and body["curr"] == "NVDA FY2025"
    assert sum(body["counts"].values()) == len(body["changes"])
    assert client.get("/diff/AAPL").status_code == 404
    assert client.get("/filings/NOPE/financials").status_code == 404
