"""Convert EDGAR 10-K HTML into ordered text blocks and segment it into Items.

The parser keeps three pieces of layout information that downstream steps rely on:

* block boundaries (paragraphs, list items, table rows), so sentences never
  straddle two unrelated paragraphs;
* emphasis (bold / italic), because risk-factor headings are typeset that way;
* whether a block came from a table, so prose-only steps can skip tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import unescape

from lxml import html as lxml_html
from lxml.etree import _Element

BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "center"}
SKIP_TAGS = {"script", "style", "head", "title", "{http://www.xbrl.org/2013/inlineXBRL}header"}
_WS = re.compile(r"\s+")
_BOLD = re.compile(r"font-weight\s*:\s*(bold|[6-9]00)", re.I)
_ITALIC = re.compile(r"font-style\s*:\s*italic", re.I)
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)


@dataclass(frozen=True)
class Block:
    """A visually separate run of text in the filing."""

    idx: int
    text: str
    bold: bool = False
    italic: bool = False
    is_table: bool = False
    #: Emphasized run-in heading at the start of an otherwise plain paragraph
    #: (e.g. Oracle's "**Acquisitions present many risks.** We may not...").
    lead: str = ""

    @property
    def emphasized(self) -> bool:
        return self.bold or self.italic


@dataclass
class Section:
    """A 10-K Item (e.g. ``1A`` Risk Factors) as a contiguous range of blocks."""

    item: str
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)


def _clean(text: str) -> str:
    return _WS.sub(" ", unescape(text).replace("\xa0", " ")).strip()


def _local(tag: object) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].lower()


def _style_flags(el: _Element, inherited: tuple[bool, bool]) -> tuple[bool, bool]:
    bold, italic = inherited
    style = el.get("style") or ""
    tag = _local(el.tag)
    if tag in {"b", "strong"} or _BOLD.search(style):
        bold = True
    if tag in {"i", "em"} or _ITALIC.search(style):
        italic = True
    return bold, italic


def _collect(el: _Element, flags: tuple[bool, bool], out: list[tuple[str, bool, bool]]) -> None:
    """Collect (text, bold, italic) runs under ``el`` in document order."""
    if el.text:
        out.append((el.text, *flags))
    for child in el:
        tag = child.tag
        if not isinstance(tag, str):  # comments / processing instructions
            if child.tail:
                out.append((child.tail, *flags))
            continue
        if _local(tag) in SKIP_TAGS or _HIDDEN.search(child.get("style") or ""):
            if child.tail:
                out.append((child.tail, *flags))
            continue
        if _local(tag) == "br":
            out.append((" ", *flags))
        else:
            _collect(child, _style_flags(child, flags), out)
        if child.tail:
            out.append((child.tail, *flags))


def _has_block_descendant(el: _Element) -> bool:
    return any(_local(d.tag) in BLOCK_TAGS for d in el.iterdescendants())


def html_to_blocks(raw_html: str | bytes) -> list[Block]:
    """Flatten a filing into leaf-level text blocks in reading order."""
    if isinstance(raw_html, str):
        raw_html = raw_html.encode("utf-8", errors="ignore")
    root = lxml_html.fromstring(raw_html)
    blocks: list[Block] = []

    def emit(el: _Element, is_table: bool) -> None:
        runs: list[tuple[str, bool, bool]] = []
        if is_table:
            cells = []
            for cell in el:
                if _local(cell.tag) not in {"td", "th"}:
                    continue
                cell_runs: list[tuple[str, bool, bool]] = []
                _collect(cell, _style_flags(cell, (False, False)), cell_runs)
                ctext = _clean("".join(r[0] for r in cell_runs))
                if ctext and ctext not in {"$", "%", ")"}:
                    cells.append(ctext)
                runs.extend(cell_runs)
            text = " | ".join(cells)
        else:
            _collect(el, _style_flags(el, (False, False)), runs)
            text = _clean("".join(r[0] for r in runs))
        if not text:
            return
        visible = [r for r in runs if r[0].strip()]
        bold = bool(visible) and all(r[1] for r in visible)
        italic = bool(visible) and all(r[2] for r in visible)
        lead = ""
        if not is_table and not (bold or italic):
            head: list[str] = []
            for r in runs:
                if r[0].strip() and not (r[1] or r[2]):
                    break
                head.append(r[0])
            cand = _clean("".join(head))
            if len(cand) >= 25 and cand[-1:] in ".?!" and len(cand) < len(text):
                lead = cand
        blocks.append(Block(len(blocks), text, bold, italic, is_table, lead))

    def walk(el: _Element) -> None:
        for child in el:
            tag = _local(child.tag)
            if not tag or tag in SKIP_TAGS or _HIDDEN.search(child.get("style") or ""):
                continue
            if tag == "tr":
                emit(child, is_table=True)
            elif tag in BLOCK_TAGS and not _has_block_descendant(child):
                emit(child, is_table=False)
            else:
                walk(child)

    walk(root)
    return _merge_page_breaks(blocks)


_PAGE_NOISE = re.compile(
    r"^(\d{1,3}|page \d+|part i{1,3}v?|table of contents|[a-z .&,]{0,40}\|\s*\d{4} form 10-k\s*\|\s*\d+)$",
    re.I,
)


def is_page_noise(text: str) -> bool:
    """Page numbers and running headers that interrupt the body text."""
    return bool(_PAGE_NOISE.match(text.strip()))


def _merge_page_breaks(blocks: list[Block]) -> list[Block]:
    """Drop page furniture and re-join paragraphs split across a page break."""
    kept = [b for b in blocks if not is_page_noise(b.text)]
    merged: list[Block] = []
    for b in kept:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and not prev.is_table
            and not b.is_table
            and (prev.bold, prev.italic) == (b.bold, b.italic)
            and prev.text[-1:] not in '.:;?!\u201d"'
            and b.text[:1].islower()
        ):
            merged[-1] = Block(prev.idx, f"{prev.text} {b.text}", prev.bold, prev.italic, False, prev.lead)
        else:
            merged.append(b)
    return [Block(i, b.text, b.bold, b.italic, b.is_table, b.lead) for i, b in enumerate(merged)]


# --- Item segmentation -------------------------------------------------------

ITEM_TITLES: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}
_ITEM_RE = re.compile(r"^item\s*(1[0-6]|[1-9])\s*([A-C])?\s*[.:\-—–|]?\s*(.*)$", re.I)
# Banks such as Citi and JPMorgan do not use "Item 1A" headings; their risk
# section is introduced by a bare "RISK FACTORS" heading instead.
_BARE_HEADINGS = {"risk factors": "1A"}


def _item_key(block: Block) -> str | None:
    if block.is_table and block.text.count("|") > 3:
        return None
    text = block.text.replace("|", " ")
    text = _WS.sub(" ", text).strip()
    if len(text) > 160:
        return None
    m = _ITEM_RE.match(text)
    if m:
        key = (m.group(1) + (m.group(2) or "")).upper()
        return key if key in ITEM_TITLES else None
    if text.lower().rstrip(".") in _BARE_HEADINGS and (block.emphasized or text.isupper()):
        return _BARE_HEADINGS[text.lower().rstrip(".")]
    return None


def segment_items(blocks: list[Block]) -> dict[str, Section]:
    """Split blocks into 10-K Items.

    Every Item heading appears at least twice (table of contents + body, and
    sometimes in page headers), so for each Item we keep the occurrence that
    yields the longest span of prose before the next heading.
    """
    hits = [(b.idx, k) for b in blocks if (k := _item_key(b))]
    hits += _bare_section_ends(blocks, hits)
    hits.sort()
    best: dict[str, tuple[int, int]] = {}
    for i, (start, key) in enumerate(hits):
        end = len(blocks)
        for nxt_start, nxt_key in hits[i + 1 :]:
            if nxt_key != key:
                end = nxt_start
                break
        span = sum(len(b.text) for b in blocks[start + 1 : end] if not b.is_table)
        if key == _END:
            continue
        if key not in best or span > _span(blocks, best[key]):
            best[key] = (start, end)
    sections: dict[str, Section] = {}
    for key, (start, end) in best.items():
        body = blocks[start + 1 : end]
        if not body:
            continue
        sections[key] = Section(key, ITEM_TITLES[key], body)
    return sections


def _span(blocks: list[Block], rng: tuple[int, int]) -> int:
    return sum(len(b.text) for b in blocks[rng[0] + 1 : rng[1]] if not b.is_table)


_END = "_END"


def _bare_section_ends(blocks: list[Block], hits: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Synthesize an end marker after a bare "RISK FACTORS" heading.

    Without Item headings the section would otherwise run to the end of the
    document. We stop at the first all-caps emphasized heading that is not a
    risk *category* (those end in "RISKS", e.g. "CREDIT RISKS").
    """
    ends: list[tuple[int, str]] = []
    for start, _key in hits:
        b = blocks[start]
        if _ITEM_RE.match(b.text):
            continue
        for nxt in blocks[start + 1 :]:
            t = nxt.text.strip()
            if (
                not nxt.is_table
                and nxt.emphasized
                and t.isupper()
                and len(t) < 80
                and not t.rstrip(".").endswith("RISKS")
                and _item_key(nxt) is None
            ):
                ends.append((nxt.idx, _END))
                break
    return ends
