"""Extract structured financial facts from inline XBRL (iXBRL) 10-K filings.

Since 2019 the SEC requires 10-K filings to tag every financial-statement
number inline with ``<ix:nonFraction>``. That means headline financials can be
read straight from the filing with exact provenance, with no table parsing
and no model in the loop.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date

from lxml import etree

IX = "http://www.xbrl.org/2013/inlineXBRL"
XBRLI = "http://www.xbrl.org/2003/instance"
XBRLDI = "http://xbrl.org/2006/xbrldi"


@dataclass(frozen=True)
class Fact:
    concept: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    decimals: str | None
    fact_id: str | None

    @property
    def duration_days(self) -> int | None:
        if not self.period_start:
            return None
        return (date.fromisoformat(self.period_end) - date.fromisoformat(self.period_start)).days

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Headline metrics -> ordered list of us-gaap concepts that can carry them.
METRICS: dict[str, list[str]] = {
    "Revenue": [
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenuesNetOfInterestExpense",
        "us-gaap:SalesRevenueNet",
    ],
    "Net income": ["us-gaap:NetIncomeLoss", "us-gaap:ProfitLoss"],
    "Operating income": ["us-gaap:OperatingIncomeLoss"],
    "R&D expense": ["us-gaap:ResearchAndDevelopmentExpense"],
    "Diluted EPS": ["us-gaap:EarningsPerShareDiluted"],
    "Operating cash flow": ["us-gaap:NetCashProvidedByUsedInOperatingActivities"],
    "Total assets": ["us-gaap:Assets"],
    "Total liabilities": ["us-gaap:Liabilities"],
    "Stockholders' equity": [
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "Cash and equivalents": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashAndDueFromBanks",
    ],
}
INSTANT_METRICS = {"Total assets", "Total liabilities", "Stockholders' equity", "Cash and equivalents"}

_NUM = re.compile(r"[^0-9.]")


def _parse_number(text: str, fmt: str | None) -> float:
    text = text.strip()
    if not re.search(r"\d", text):
        return 0.0  # ixt:fixed-zero renders as a dash
    if fmt and "comma-decimal" in fmt.replace("numcomma", "comma-decimal"):
        text = text.replace(".", "").replace(",", ".")
    return float(_NUM.sub("", text) or 0)


def _contexts(root: etree._Element) -> dict[str, tuple[str | None, str, bool]]:
    """context id -> (start, end, has_dimensions)."""
    out: dict[str, tuple[str | None, str, bool]] = {}
    for ctx in root.iter(f"{{{XBRLI}}}context"):
        cid = ctx.get("id")
        if not cid:
            continue
        period = ctx.find(f"{{{XBRLI}}}period")
        if period is None:
            continue
        instant = period.findtext(f"{{{XBRLI}}}instant")
        start = period.findtext(f"{{{XBRLI}}}startDate")
        end = period.findtext(f"{{{XBRLI}}}endDate")
        has_dims = ctx.find(f".//{{{XBRLDI}}}explicitMember") is not None or (
            ctx.find(f".//{{{XBRLDI}}}typedMember") is not None
        )
        if instant:
            out[cid] = (None, instant.strip(), has_dims)
        elif end:
            out[cid] = ((start or "").strip() or None, end.strip(), has_dims)
    return out


def extract_facts(raw_html: bytes) -> tuple[list[Fact], dict[str, str]]:
    """Return consolidated (dimensionless) numeric facts and ``dei:`` cover data."""
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(raw_html, parser)
    if root is None:
        return [], {}
    ctxs = _contexts(root)
    facts: list[Fact] = []
    seen: set[tuple[str, str | None, str]] = set()
    for el in root.iter(f"{{{IX}}}nonFraction"):
        name = el.get("name")
        ctx = ctxs.get(el.get("contextRef", ""))
        if not name or ctx is None or ctx[2]:
            continue
        if el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
            continue
        text = "".join(el.itertext())
        try:
            value = _parse_number(text, el.get("format"))
        except ValueError:
            continue
        value *= 10 ** int(el.get("scale", "0") or 0)
        if el.get("sign") == "-":
            value = -value
        key = (name, ctx[0], ctx[1])
        if key in seen:
            continue
        seen.add(key)
        facts.append(
            Fact(name, value, el.get("unitRef", ""), ctx[0], ctx[1], el.get("decimals"), el.get("id"))
        )
    dei: dict[str, str] = {}
    for el in root.iter(f"{{{IX}}}nonNumeric"):
        name = el.get("name", "")
        if name.startswith("dei:") and name not in dei:
            dei[name] = re.sub(r"\s+", " ", "".join(el.itertext())).strip()
    return facts, dei


def headline_financials(facts: list[Fact]) -> dict[str, dict[str, float]]:
    """Metric -> {period_end: value} for annual durations and balance-sheet instants."""
    by_concept: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_concept[f.concept].append(f)
    table: dict[str, dict[str, float]] = {}
    for metric, concepts in METRICS.items():
        for concept in concepts:
            series: dict[str, float] = {}
            for f in by_concept.get(concept, []):
                if metric in INSTANT_METRICS:
                    if f.period_start is None:
                        series[f.period_end] = f.value
                elif f.duration_days and 350 <= f.duration_days <= 380:
                    series[f.period_end] = f.value
            if series:
                table[metric] = dict(sorted(series.items()))
                break
    return table
