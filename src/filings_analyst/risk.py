"""Split Item 1A into individual risk factors.

Issuers typeset each risk factor as an emphasized (bold or italic) headline
sentence followed by one or more body paragraphs, grouped under short
category headings such as "Risks Related to Our Industry". We use exactly
those typographic cues; no model is involved.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .parse import Block, Section

_BULLET = re.compile(r"^[•●▪–\-\*]\s*")


@dataclass(frozen=True)
class RiskFactor:
    id: str
    category: str
    heading: str
    body: str
    position: int

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_heading(b: Block) -> bool:
    if b.is_table or not b.emphasized:
        return False
    text = b.text.strip()
    return 25 <= len(text) <= 700 and not _BULLET.match(text)


def _is_category(b: Block) -> bool:
    if b.is_table or not b.emphasized:
        return False
    text = b.text.strip()
    if len(text) > 120 or text[-1:] in ".:;?!":
        return False
    low = text.lower()
    return "risk" in low or text.isupper() or "summary" in low or len(text.split()) <= 4


def _rid(heading: str) -> str:
    return hashlib.sha1(heading.lower().encode()).hexdigest()[:10]


def split_risk_factors(section: Section, min_body_chars: int = 150) -> list[RiskFactor]:
    factors: list[RiskFactor] = []
    category = "General"
    heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal heading, body
        if heading and "summary" not in category.lower():
            text = " ".join(body).strip()
            if len(text) >= min_body_chars:
                factors.append(RiskFactor(_rid(heading), category, heading, text, len(factors)))
        heading, body = None, []

    for b in section.blocks:
        if _is_category(b):
            flush()
            category = b.text.strip().title() if b.text.isupper() else b.text.strip()
        elif _is_heading(b):
            # Consecutive emphasized lines are one wrapped headline.
            if heading is not None and not body:
                heading = f"{heading} {b.text.strip()}"
            else:
                flush()
                heading = b.text.strip()
        elif b.lead and not _BULLET.match(b.lead):
            flush()
            heading = b.lead
            body.append(b.text[len(b.lead) :].strip())
        elif heading is not None and not b.is_table:
            body.append(b.text.strip())
    flush()
    return factors
