"""Sentence splitting and light normalisation tuned for SEC prose."""

from __future__ import annotations

import re

_ABBREV = [
    "U.S.",
    "U.K.",
    "Inc.",
    "Corp.",
    "Co.",
    "Ltd.",
    "No.",
    "Nos.",
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "approx.",
    "Mr.",
    "Ms.",
    "Dr.",
    "St.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
    "N.A.",
    "L.P.",
    "S.A.",
    "p.m.",
    "a.m.",
]
_PLACEHOLDER = "․"  # one-dot leader: looks like a period, never a sentence end
_SPLIT = re.compile(r"(?<=[.!?”])\s+(?=[\"“(]?[A-Z0-9•\[])")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "our",
        "we",
        "us",
        "their",
        "they",
        "them",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
        "why",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "shall",
        "should",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "not",
        "no",
        "than",
        "then",
        "such",
        "so",
        "also",
        "any",
        "all",
        "each",
        "other",
        "into",
        "about",
        "over",
        "under",
        "more",
        "most",
        "less",
        "least",
        "very",
        "there",
        "here",
        "including",
        "include",
        "includes",
        "certain",
    ]
)


def split_sentences(text: str) -> list[str]:
    protected = text
    for abbr in _ABBREV:
        protected = protected.replace(abbr, abbr.replace(".", _PLACEHOLDER))
    # decimals such as 1.5 and section numbers are safe because the split
    # requires whitespace after the period.
    parts = _SPLIT.split(protected)
    out = []
    for p in parts:
        s = p.replace(_PLACEHOLDER, ".").strip()
        if s:
            out.append(s)
    return out


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("’", "'"))


def stem(tok: str) -> str:
    """Tiny suffix stripper: enough to match 'risks'/'risk', 'declined'/'decline'."""
    for suf in ("ings", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)] + ("y" if suf in ("ies", "ied") else "")
    return tok


def content_terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]
