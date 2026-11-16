"""Build and load the processed corpus.

A *filing* is processed into a single JSON document holding cover metadata,
headline iXBRL financials, the narrative Items, individual risk factors and a
flat list of citable sentences. Every downstream feature (retrieval,
answering, verification, diffing, UI) reads only this format.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .parse import html_to_blocks, segment_items
from .risk import RiskFactor, split_risk_factors
from .text import split_sentences
from .xbrl import extract_facts, headline_financials

NARRATIVE_ITEMS = ("1", "1A", "7", "7A")
MAX_CHUNK_CHARS = 1200


@dataclass
class Sentence:
    sid: str
    text: str
    item: str
    chunk: int
    risk_id: str | None = None


@dataclass
class Filing:
    doc_id: str
    company: str
    ticker: str
    cik: str
    form: str
    fiscal_year: int
    period_end: str
    financials: dict[str, dict[str, float]]
    risk_factors: list[dict[str, object]]
    sentences: list[Sentence] = field(default_factory=list)
    source: str = ""

    def to_json(self) -> dict[str, object]:
        d = asdict(self)
        d["sentences"] = [asdict(s) for s in self.sentences]
        return d

    @classmethod
    def from_json(cls, d: dict[str, object]) -> Filing:
        sents = [Sentence(**s) for s in d.pop("sentences", [])]  # type: ignore[arg-type,attr-defined]
        f = cls(**d)  # type: ignore[arg-type]
        f.sentences = sents
        return f

    @property
    def label(self) -> str:
        return f"{self.ticker} FY{self.fiscal_year}"

    def risks(self) -> list[RiskFactor]:
        return [RiskFactor(**r) for r in self.risk_factors]  # type: ignore[arg-type]


def _period_end(raw: str) -> str:
    for fmt in ("%B %d, %Y", "%B %d %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw.replace(" ,", ","), fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def process_filing(raw_html: bytes, ticker: str = "", source: str = "") -> Filing:
    facts, dei = extract_facts(raw_html)
    blocks = html_to_blocks(raw_html)
    sections = segment_items(blocks)
    fy = int(dei.get("dei:DocumentFiscalYearFocus", "0") or 0)
    if not ticker:
        symbols = dei.get("dei:TradingSymbol", "").split()
        ticker = symbols[0] if symbols else "UNK"
    form = dei.get("dei:DocumentType", "10-K")
    doc_id = f"{ticker}-{form}-FY{fy}"
    risks = split_risk_factors(sections["1A"]) if "1A" in sections else []

    sentences: list[Sentence] = []
    chunk_no = 0
    # Map each risk-factor body sentence back to its factor for UI linking.
    risk_lookup: dict[str, str] = {}
    for r in risks:
        for s in split_sentences(r.heading + " " + r.body):
            risk_lookup.setdefault(s, r.id)
    for item in NARRATIVE_ITEMS:
        sec = sections.get(item)
        if not sec:
            continue
        buf: list[str] = []
        size = 0

        def flush(item: str = item) -> None:
            nonlocal buf, size, chunk_no
            if not buf:
                return
            for s in buf:
                sentences.append(
                    Sentence(f"{doc_id}#{len(sentences)}", s, item, chunk_no, risk_lookup.get(s))
                )
            chunk_no += 1
            buf, size = [], 0

        for b in sec.blocks:
            if b.is_table:
                continue
            sents = [s for s in split_sentences(b.text) if len(s) > 20]
            if b.emphasized:
                flush()  # headings start a new chunk
            for s in sents:
                if size + len(s) > MAX_CHUNK_CHARS and buf:
                    flush()
                buf.append(s)
                size += len(s)
            if size > MAX_CHUNK_CHARS // 2:
                flush()
        flush()

    return Filing(
        doc_id=doc_id,
        company=dei.get("dei:EntityRegistrantName", ticker),
        ticker=ticker,
        cik=dei.get("dei:EntityCentralIndexKey", ""),
        form=form,
        fiscal_year=fy,
        period_end=_period_end(dei.get("dei:DocumentPeriodEndDate", "")),
        financials=headline_financials(facts),
        risk_factors=[r.to_dict() for r in risks],
        sentences=sentences,
        source=source,
    )


def save_filing(filing: Filing, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{filing.doc_id}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(filing.to_json(), fh, ensure_ascii=False)
    return path


def load_corpus(directory: Path) -> list[Filing]:
    filings = []
    for path in sorted(directory.glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            filings.append(Filing.from_json(json.load(fh)))
    return sorted(filings, key=lambda f: (f.ticker, f.fiscal_year))
