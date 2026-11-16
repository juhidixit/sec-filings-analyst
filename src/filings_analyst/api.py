"""FastAPI service exposing Q&A, risk diffs and financials."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .answer import Analyst
from .corpus import Filing, load_corpus
from .diff import diff_risk_factors, theme_intensity

CORPUS_DIR = Path(
    os.environ.get("FILINGS_CORPUS", Path(__file__).resolve().parents[2] / "data" / "processed")
)

app = FastAPI(
    title="SEC Filings Analyst",
    version=__version__,
    description="Grounded Q&A over SEC 10-K filings with sentence-level citations "
    "and automatic claim verification.",
)


@lru_cache(maxsize=1)
def _filings() -> dict[str, Filing]:
    return {f.doc_id: f for f in load_corpus(CORPUS_DIR)}


@lru_cache(maxsize=1)
def _analyst() -> Analyst:
    return Analyst(list(_filings().values()))


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=500, examples=["What does NVIDIA say about export controls?"]
    )
    k: int = Field(3, ge=1, le=8)
    strict: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "filings": len(_filings()), "version": __version__}


@app.get("/filings")
def filings() -> list[dict[str, Any]]:
    return [
        {
            "doc_id": f.doc_id,
            "company": f.company,
            "ticker": f.ticker,
            "cik": f.cik,
            "fiscal_year": f.fiscal_year,
            "period_end": f.period_end,
            "risk_factors": len(f.risk_factors),
            "sentences": len(f.sentences),
        }
        for f in _filings().values()
    ]


@app.get("/filings/{doc_id}/financials")
def financials(doc_id: str) -> dict[str, dict[str, float]]:
    f = _filings().get(doc_id)
    if f is None:
        raise HTTPException(404, f"unknown filing {doc_id}")
    return f.financials


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, object]:
    return _analyst().ask(req.question, k=req.k, strict=req.strict).to_dict()


@app.get("/diff/{ticker}")
def diff(ticker: str) -> dict[str, Any]:
    fs = sorted(
        (f for f in _filings().values() if f.ticker == ticker.upper()),
        key=lambda f: f.fiscal_year,
    )
    if len(fs) < 2:
        raise HTTPException(404, f"need two filings for {ticker}")
    prev, curr = fs[-2], fs[-1]
    d = diff_risk_factors(prev.risks(), curr.risks(), prev.label, curr.label)
    return {
        "prev": prev.label,
        "curr": curr.label,
        "counts": d.counts(),
        "themes": {prev.label: theme_intensity(prev.risks()), curr.label: theme_intensity(curr.risks())},
        "changes": [c.to_dict() for c in d.changes],
    }
