"""Year-over-year comparison of risk factors.

Risk factors are matched across two filings of the same company with an
optimal one-to-one assignment (Hungarian algorithm) on TF-IDF cosine
similarity. Each factor is then labelled:

* ``new``        - no counterpart last year
* ``removed``    - no counterpart this year
* ``modified``   - matched, but sentences were added or removed
* ``unchanged``  - matched with (near) identical text

Investors and analysts care most about *what changed*, because boilerplate
risk language is repeated every year and newly added language is where
material information tends to hide.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .risk import RiskFactor
from .text import content_terms, split_sentences

THEMES: dict[str, str] = {
    "AI & machine learning": r"\b(artificial intelligence|AI|generative|machine learning|LLMs?|large language models?)\b",
    "Export controls & China": r"\b(export controls?|China|Chinese|PRC|Taiwan|entity list)\b",
    "Tariffs & trade": r"\b(tariffs?|trade (policy|disputes?|war|restrictions?)|sanctions)\b",
    "Cybersecurity": r"\b(cyber\w*|ransomware|data breach\w*|hack\w*|malware|security incidents?)\b",
    "Climate & ESG": r"\b(climate|ESG|sustainability|greenhouse|emissions|net zero|carbon)\b",
    "Interest rates & inflation": r"\b(interest rates?|inflation\w*|monetary policy|rate hikes?)\b",
    "Regulation & compliance": r"\b(regulat\w+|consent orders?|supervis\w+|compliance|enforcement)\b",
    "Supply chain": r"\b(supply chain|suppliers?|shortages?|components?|manufactur\w+)\b",
    "Competition": r"\b(competition|competitors?|competitive)\b",
    "Litigation": r"\b(litigation|lawsuits?|legal proceedings|class actions?)\b",
    "Talent & key people": r"\b(key (employees|personnel)|talent|retain\w* (qualified|key)|hiring)\b",
    "Geopolitics & conflict": r"\b(geopolitic\w*|war|Ukraine|Russia|Israel|Middle East|hostilities|terroris\w+)\b",
    "Capital & liquidity": r"\b(liquidity|capital requirements?|funding|credit ratings?|indebtedness)\b",
}
_THEME_RE = {k: re.compile(v, re.I) for k, v in THEMES.items()}


def tag_themes(text: str) -> list[str]:
    return [k for k, rx in _THEME_RE.items() if rx.search(text)]


def theme_intensity(factors: list[RiskFactor]) -> dict[str, float]:
    """Share of risk-factor sentences (%) that mention each theme."""
    sents = [s for f in factors for s in split_sentences(f.text)]
    if not sents:
        return {k: 0.0 for k in THEMES}
    return {
        k: round(100 * sum(bool(rx.search(s)) for s in sents) / len(sents), 2) for k, rx in _THEME_RE.items()
    }


@dataclass
class FactorChange:
    status: str  # new | removed | modified | unchanged
    category: str
    heading: str
    prev_heading: str | None
    similarity: float
    added_sentences: list[str] = field(default_factory=list)
    removed_sentences: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    risk_id: str | None = None
    prev_risk_id: str | None = None
    body: str = ""
    prev_body: str = ""
    #: (old, new) sentence pairs that were reworded rather than added/removed
    edited: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RiskDiff:
    prev_label: str
    curr_label: str
    changes: list[FactorChange]

    def counts(self) -> dict[str, int]:
        out = {"new": 0, "removed": 0, "modified": 0, "unchanged": 0}
        for c in self.changes:
            out[c.status] += 1
        return out


def _sentence_diff(a: str, b: str) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Sentence-level diff: (added, removed, edited pairs)."""
    sa, sb = split_sentences(a), split_sentences(b)
    sm = difflib.SequenceMatcher(a=sa, b=sb, autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(sa[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(sb[j1:j2])
    # Sentences that merely moved are not changes.
    moved = set(added) & set(removed)
    added = [s for s in added if s not in moved]
    removed = [s for s in removed if s not in moved]
    # A removed + added pair that is mostly the same words is an *edit*.
    edited: list[tuple[str, str]] = []
    for new in list(added):
        best, score = None, 0.0
        for old in removed:
            r = difflib.SequenceMatcher(a=old.split(), b=new.split(), autojunk=False).ratio()
            if r > score:
                best, score = old, r
        if best is not None and score >= EDIT_RATIO:
            edited.append((best, new))
            added.remove(new)
            removed.remove(best)
    return added, removed, edited


EDIT_RATIO = 0.6


def word_diff(old: str, new: str) -> list[tuple[str, str]]:
    """Word-level diff as (op, text) runs with op in {"=", "+", "-"}."""
    a, b = old.split(), new.split()
    # Compare words without trailing punctuation so "costs" vs "costs," is not a change.
    na = [w.rstrip(".,;:") for w in a]
    nb = [w.rstrip(".,;:") for w in b]
    out: list[tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=na, b=nb, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(("=", " ".join(b[j1:j2])))
        else:
            if i2 > i1:
                out.append(("-", " ".join(a[i1:i2])))
            if j2 > j1:
                out.append(("+", " ".join(b[j1:j2])))
    return out


def diff_risk_factors(
    prev: list[RiskFactor],
    curr: list[RiskFactor],
    prev_label: str = "previous",
    curr_label: str = "current",
    match_threshold: float = 0.3,
) -> RiskDiff:
    changes: list[FactorChange] = []
    matched_prev: set[int] = set()
    matched_curr: dict[int, tuple[int, float]] = {}
    if prev and curr:
        vec = TfidfVectorizer(analyzer=content_terms, sublinear_tf=True)
        docs = [f"{f.heading} {f.heading} {f.body}" for f in prev + curr]
        m = vec.fit_transform(docs)
        sim = cosine_similarity(m[len(prev) :], m[: len(prev)])
        rows, cols = linear_sum_assignment(-sim)
        for r, c in zip(rows, cols, strict=True):
            if sim[r, c] >= match_threshold:
                matched_curr[int(r)] = (int(c), float(sim[r, c]))
                matched_prev.add(int(c))

    for i, f in enumerate(curr):
        if i in matched_curr:
            j, s = matched_curr[i]
            p = prev[j]
            added, removed, edited = _sentence_diff(p.text, f.text)
            status = "unchanged" if not (added or removed or edited) else "modified"
            changes.append(
                FactorChange(
                    status,
                    f.category,
                    f.heading,
                    p.heading,
                    round(s, 3),
                    added,
                    removed,
                    tag_themes(f.text),
                    f.id,
                    p.id,
                    f.body,
                    p.body,
                    edited,
                )
            )
        else:
            changes.append(
                FactorChange(
                    "new",
                    f.category,
                    f.heading,
                    None,
                    0.0,
                    split_sentences(f.text),
                    [],
                    tag_themes(f.text),
                    f.id,
                    None,
                    f.body,
                    "",
                )
            )
    for j, p in enumerate(prev):
        if j not in matched_prev:
            changes.append(
                FactorChange(
                    "removed",
                    p.category,
                    p.heading,
                    p.heading,
                    0.0,
                    [],
                    split_sentences(p.text),
                    tag_themes(p.text),
                    None,
                    p.id,
                    "",
                    p.body,
                )
            )
    return RiskDiff(prev_label, curr_label, changes)


def change_magnitude(c: FactorChange) -> int:
    """Characters of text added, removed or reworded - used to rank what changed most."""
    edits = sum(len(t) for old, new in c.edited for op, t in word_diff(old, new) if op != "=")
    return sum(len(s) for s in c.added_sentences) + sum(len(s) for s in c.removed_sentences) + edits


def similarity_matrix(prev: list[RiskFactor], curr: list[RiskFactor]) -> np.ndarray:
    vec = TfidfVectorizer(analyzer=content_terms, sublinear_tf=True)
    m = vec.fit_transform([f.text for f in prev + curr])
    return np.asarray(cosine_similarity(m[len(prev) :], m[: len(prev)]))
