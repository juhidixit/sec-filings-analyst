"""Minimal, polite client for SEC EDGAR.

SEC asks automated clients to identify themselves with a User-Agent that
contains contact details and to stay under 10 requests per second
(https://www.sec.gov/os/accessing-edgar-data). Both are enforced here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, acc=self.accession.replace("-", ""), doc=self.primary_document
        )


class EdgarClient:
    def __init__(
        self,
        user_agent: str | None = None,
        min_interval: float = 0.15,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not ua or "@" not in ua:
            raise ValueError(
                "SEC requires a descriptive User-Agent with contact email. "
                "Set SEC_USER_AGENT='Your Name your.email@example.com'."
            )
        self.http = httpx.Client(
            headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
            timeout=30.0,
            transport=transport,
            follow_redirects=True,
        )
        self.min_interval = min_interval
        self._last = 0.0
        self._tickers: dict[str, int] | None = None

    def _get(self, url: str) -> httpx.Response:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            resp = self.http.get(url)
            self._last = time.monotonic()
            if resp.status_code not in (429, 503):
                resp.raise_for_status()
                return resp
            time.sleep(2**attempt)
        resp.raise_for_status()
        return resp

    def cik_for(self, ticker: str) -> int:
        if self._tickers is None:
            data = self._get(TICKERS_URL).json()
            self._tickers = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
        try:
            return self._tickers[ticker.upper()]
        except KeyError as exc:
            raise KeyError(f"Unknown ticker {ticker!r}") from exc

    def list_filings(self, ticker: str, form: str = "10-K", limit: int = 3) -> list[FilingRef]:
        cik = self.cik_for(ticker)
        recent = self._get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        out: list[FilingRef] = []
        for i, f in enumerate(recent["form"]):
            if f != form:
                continue
            out.append(
                FilingRef(
                    ticker.upper(),
                    cik,
                    f,
                    recent["accessionNumber"][i],
                    recent["filingDate"][i],
                    recent["reportDate"][i],
                    recent["primaryDocument"][i],
                )
            )
            if len(out) >= limit:
                break
        return out

    def download(self, ref: FilingRef, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{ref.ticker}_{ref.form}_{ref.report_date}.html"
        if not path.exists():
            path.write_bytes(self._get(ref.url).content)
        return path
