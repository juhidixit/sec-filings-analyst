"""Claim verification: does each answer sentence follow from what it cites?

Every answer sentence is treated as a *claim* and checked against the
sentences it cites with four independent, explainable tests:

1. **Numbers** - every quantity in the claim (money, percentages, counts,
   years) must appear in the evidence, after unit normalisation
   ("$14.4 billion" == "14,412 million" within rounding tolerance).
2. **Entities** - capitalised names in the claim must occur in the evidence
   or be an alias of the filer itself.
3. **Polarity** - "increased" vs "decreased" style direction words and
   negations must not contradict the evidence.
4. **Coverage** - most of the claim's content words must be present in the
   evidence (stemmed), otherwise the claim says something the source does not.

The checks are deliberately symbolic. They are cheap, deterministic and give
a human-readable reason for every flag, which matters more in a compliance
setting than squeezing out the last point of recall.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from .text import STOPWORDS, content_terms, split_sentences

_SCALE = {
    "thousand": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
}
_QTY = re.compile(
    r"(?P<neg>[-\u2212(])?(?P<cur>\$|usd\s?)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s?(?P<pct>%|percent\b|percentage points?\b|bps\b|basis points\b)?"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn|tn)\b)?",
    re.I,
)
UP = frozenset(
    [
        "increase",
        "increased",
        "increases",
        "increasing",
        "rose",
        "rise",
        "rises",
        "grew",
        "grow",
        "grows",
        "growth",
        "higher",
        "gain",
        "gains",
        "gained",
        "improved",
        "improve",
        "up",
        "expanded",
        "expand",
    ]
)
DOWN = frozenset(
    [
        "decrease",
        "decreased",
        "decreases",
        "decreasing",
        "fell",
        "fall",
        "falls",
        "declined",
        "decline",
        "declines",
        "lower",
        "loss",
        "losses",
        "reduced",
        "reduce",
        "down",
        "contracted",
        "shrank",
        "worsened",
    ]
)
NEGATIONS = frozenset(["not", "no", "never", "none", "neither", "nor", "without", "cannot"])
_CAP = re.compile(r"\b([A-Z][a-zA-Z&\-]+(?:\s+[A-Z][a-zA-Z&\-]+)*)\b")
_GENERIC_CAPS = frozenset(
    [
        "The",
        "This",
        "These",
        "Those",
        "Our",
        "We",
        "In",
        "On",
        "For",
        "As",
        "It",
        "Its",
        "If",
        "Any",
        "Such",
        "Additionally",
        "Also",
        "However",
        "Company",
        "Board",
        "Item",
        "Form",
        "Annual",
        "Report",
        "United",
        "States",
        "Federal",
        "Reserve",
        "Act",
        "Rule",
        "Fiscal",
        "Year",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        "Risk",
        "Risks",
        "Factors",
        "Note",
        "Part",
        "Management",
        "Directors",
        "Total",
        "Net",
        "Revenue",
        "Revenues",
        "Income",
        "Based",
        "According",
        "Citations",
        "Not",
        "Unsupported",
        "Answer",
    ]
)


_SCALE_WORDS = frozenset({"thousand", "million", "billion", "trillion", "percent", "usd"})


class Verdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    NO_CITATION = "no_citation"


@dataclass(frozen=True)
class Quantity:
    value: float
    kind: str  # money | percent | year | number
    raw: str
    #: Smallest unit the writer expressed, e.g. 0.1e9 for "$2.4 billion".
    #: A claim matches evidence that rounds to it at that precision.
    step: float = 0.0


@dataclass
class ClaimCheck:
    claim: str
    citations: list[str]
    verdict: Verdict
    coverage: float
    reasons: list[str] = field(default_factory=list)


def extract_quantities(text: str) -> list[Quantity]:
    out: list[Quantity] = []
    for m in _QTY.finditer(text):
        raw = m.group(0).strip()
        digits = m.group("num")
        num = float(digits.replace(",", ""))
        if m.group("neg") and m.group("cur"):
            num = -num  # "-$19.7 billion" / "($19.7 billion)"; never "2020-2024"
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        if m.group("pct"):
            out.append(Quantity(num, "percent", raw, 10.0**-decimals))
            continue
        scale = _SCALE.get((m.group("scale") or "").lower(), 1.0)
        if m.group("cur") or m.group("scale"):
            kind = "money" if m.group("cur") else "number"
            out.append(Quantity(num * scale, kind, raw, 10.0**-decimals * scale))
        elif re.fullmatch(r"(19|20)\d\d", m.group("num")):
            out.append(Quantity(num, "year", raw))
        elif num >= 10 or "." in m.group("num"):
            out.append(Quantity(num, "number", raw, 10.0**-decimals))
        # bare single digits ("one of 3 factors") are too noisy to check
    return out


def _qty_match(q: Quantity, evidence: list[Quantity], rel_tol: float = 0.001) -> bool:
    """True if ``q`` matches an unused evidence quantity (which is then consumed)."""
    for i, e in enumerate(evidence):
        if _same(q, e, rel_tol):
            evidence.pop(i)
            return True
    return False


def _same(q: Quantity, e: Quantity, rel_tol: float) -> bool:
    if q.kind == "year" or e.kind == "year":
        return q.kind == e.kind and q.value == e.value
    if (q.kind == "percent") != (e.kind == "percent"):
        return False
    if e.value == q.value or abs(e.value - q.value) <= q.step / 2 + 1e-9:
        return True
    denom = max(abs(e.value), abs(q.value))
    return bool(denom) and abs(e.value - q.value) / denom <= rel_tol


def _entities(text: str) -> set[str]:
    ents = set()
    for m in _CAP.finditer(text):
        words = [w for w in m.group(1).split() if w not in _GENERIC_CAPS and w.lower() not in STOPWORDS]
        if words:
            ents.add(" ".join(words).lower())
    return ents


def check_claim(
    claim: str,
    evidence: list[str],
    citations: list[str] | None = None,
    aliases: set[str] | None = None,
    min_coverage: float = 0.7,
) -> ClaimCheck:
    citations = citations or []
    if not evidence:
        return ClaimCheck(claim, citations, Verdict.NO_CITATION, 0.0, ["claim cites no evidence"])
    ev_text = " ".join(evidence)
    reasons: list[str] = []

    ev_q = extract_quantities(ev_text)  # consumed as claim quantities are matched
    for q in extract_quantities(claim):
        if not _qty_match(q, ev_q):
            reasons.append(f"number '{q.raw}' not found in cited evidence")

    ev_low = ev_text.lower()
    alias_low = {a.lower() for a in (aliases or set())}
    for ent in _entities(claim):
        if ent in ev_low or any(ent in a or a in ent for a in alias_low):
            continue
        if all(w in ev_low for w in ent.split()):
            continue
        reasons.append(f"entity '{ent}' not mentioned in cited evidence")

    c_list = re.findall(r"[a-z']+", claim.lower())
    e_list = re.findall(r"[a-z']+", ev_low)
    c_count, e_count = Counter(c_list), Counter(e_list)
    for words, opposite in ((UP, DOWN), (DOWN, UP)):
        extra = [w for w in words if c_count[w] > e_count[w]]
        if extra and any(e_count[o] for o in opposite):
            reasons.append(f"direction word '{extra[0]}' contradicts evidence")
            break
    if sum(c_count[n] for n in NEGATIONS) > sum(e_count[n] for n in NEGATIONS):
        reasons.append("claim contains a negation the evidence does not")

    c_terms = [
        t
        for t in content_terms(claim)
        if t not in STOPWORDS and t not in _SCALE_WORDS and not t.replace(".", "").isdigit()
    ]
    e_terms = set(content_terms(ev_text)) | {w for a in alias_low for w in content_terms(a)}
    coverage = sum(t in e_terms for t in c_terms) / len(c_terms) if c_terms else 1.0
    if coverage < min_coverage:
        reasons.append(f"only {coverage:.0%} of the claim's terms appear in the evidence")

    verdict = Verdict.UNSUPPORTED if reasons else Verdict.SUPPORTED
    return ClaimCheck(claim, citations, verdict, round(coverage, 3), reasons)


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split an answer into (claim, [citation numbers]) pairs.

    Citations are written as ``[1]`` or ``[1, 3]`` at the end of a sentence.
    """
    out: list[tuple[str, list[int]]] = []
    for sent in split_sentences(answer):
        # A citation that lands at the start of a sentence ("... text. [2] Next")
        # belongs to the previous sentence.
        lead = re.match(r"^((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)", sent)
        if lead and out:
            out[-1][1].extend(int(n) for grp in _CITE.findall(lead.group(1)) for n in grp.split(","))
            sent = sent[lead.end() :]
        nums = [int(n) for grp in _CITE.findall(sent) for n in grp.split(",")]
        clean = _CITE.sub("", sent).strip()
        clean = re.sub(r"\s+([.,;])", r"\1", clean)
        if len(clean) > 3:
            out.append((clean, nums))
    return out


def verify_answer(
    answer: str,
    sources: dict[int, str],
    aliases: set[str] | None = None,
) -> list[ClaimCheck]:
    """Verify every sentence of ``answer`` against the numbered ``sources``."""
    results = []
    for claim, nums in split_claims(answer):
        ev = [sources[n] for n in nums if n in sources]
        results.append(check_claim(claim, ev, [str(n) for n in nums], aliases))
    return results
