"""Question answering with sentence-level citations and post-hoc verification.

Questions are routed two ways:

* **Structured route** - questions about a headline metric ("What was Tesla's
  revenue in 2024?") are answered from iXBRL facts, so the number is exactly
  what the company tagged, never a model's reading of a table.
* **Narrative route** - everything else goes through hybrid retrieval. The
  default generator is *extractive* (it quotes the best sentences), which
  needs no API key. Set ``ANTHROPIC_API_KEY`` to let Claude write an abstractive
  answer from the same numbered evidence instead.

Either way, the final answer is passed through :mod:`verify`, and each
sentence is labelled supported / unsupported with a reason.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .corpus import Filing
from .retrieve import Retriever
from .verify import ClaimCheck, Verdict, verify_answer
from .xbrl import METRICS

METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Diluted EPS", re.compile(r"\b(eps|earnings per share)\b", re.I)),
    (
        "Operating cash flow",
        re.compile(
            r"\b(operating cash flow|cash (flow )?from operations|cash provided by operating)\b", re.I
        ),
    ),
    ("Operating income", re.compile(r"\boperating (income|profit)\b", re.I)),
    ("Net income", re.compile(r"\b(net income|net profit|net earnings|profit)\b", re.I)),
    ("R&D expense", re.compile(r"\b(r&d|research and development)\b", re.I)),
    ("Total assets", re.compile(r"\btotal assets\b", re.I)),
    ("Total liabilities", re.compile(r"\btotal liabilities\b", re.I)),
    ("Stockholders' equity", re.compile(r"\b(stockholders'?|shareholders'?) equity\b", re.I)),
    ("Cash and equivalents", re.compile(r"\bcash and (cash )?equivalents\b", re.I)),
    ("Revenue", re.compile(r"\b(revenue|revenues|net sales|total sales|top line)\b", re.I)),
]
_NARRATIVE_CUES = re.compile(
    r"\b(why|how (does|did|do|is|are|could|would|might|will)|risks?|describe|explain|drivers?|due to|reasons?|strategy)\b",
    re.I,
)


@dataclass
class Source:
    n: int
    ref: str
    doc_id: str
    text: str
    kind: str  # "sentence" | "xbrl" | "derived"
    item: str = ""


@dataclass
class Answer:
    question: str
    answer: str
    route: str
    generator: str
    scope: list[str]
    sources: list[Source]
    checks: list[ClaimCheck] = field(default_factory=list)
    removed_claims: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def supported_ratio(self) -> float:
        if not self.checks:
            return 0.0
        return sum(c.verdict == Verdict.SUPPORTED for c in self.checks) / len(self.checks)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["supported_ratio"] = round(self.supported_ratio, 3)
        for c in d["checks"]:
            c["verdict"] = c["verdict"].value if isinstance(c["verdict"], Verdict) else c["verdict"]
        return d


class Generator(Protocol):
    name: str

    def generate(self, question: str, sources: list[Source]) -> str: ...


class ExtractiveGenerator:
    """Quotes the retrieved sentences verbatim, each with its citation."""

    name = "extractive"

    def generate(self, question: str, sources: list[Source]) -> str:
        return " ".join(f"{s.text.rstrip().rstrip('.')} [{s.n}]." for s in sources)


class ClaudeGenerator:
    """Abstractive answers written by Claude, constrained to numbered evidence."""

    SYSTEM = (
        "You are a financial analyst answering questions about SEC filings. "
        "Use ONLY the numbered evidence sentences provided. After every sentence of your "
        "answer, cite the evidence it relies on in square brackets, e.g. [2] or [1, 3]. "
        "Do not add facts, numbers or names that are not in the evidence. If the evidence "
        "does not answer the question, reply exactly: The provided filings do not address this. "
        "Keep the answer to at most four sentences."
    )

    def __init__(self, model: str | None = None) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("FILINGS_LLM_MODEL", "claude-opus-5-5")
        self.name = f"claude:{self.model}"

    def generate(self, question: str, sources: list[Source]) -> str:
        evidence = "\n".join(f"[{s.n}] ({s.doc_id}) {s.text}" for s in sources)
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Evidence:\n{evidence}\n\nQuestion: {question}"}],
        )
        if response.stop_reason == "refusal":
            return "The provided filings do not address this."
        return "".join(b.text for b in response.content if b.type == "text").strip()


def default_generator() -> Generator:
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("FILINGS_USE_LLM", "1") != "0":
        try:
            return ClaudeGenerator()
        except Exception:  # pragma: no cover - SDK missing or misconfigured
            pass
    return ExtractiveGenerator()


def human_money(v: float, concept_metric: str) -> str:
    if concept_metric == "Diluted EPS":
        return f"${v:,.2f}"
    a = abs(v)
    sign = "-" if v < 0 else ""
    for scale, word in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if a >= scale:
            return f"{sign}${a / scale:,.1f} {word}"
    return f"{sign}${a:,.0f}"


class Analyst:
    def __init__(self, filings: list[Filing], generator: Generator | None = None) -> None:
        self.filings = {f.doc_id: f for f in filings}
        self.retriever = Retriever(filings)
        self.generator = generator or default_generator()

    # -- routing ---------------------------------------------------------------
    @staticmethod
    def detect_metric(question: str) -> str | None:
        if _NARRATIVE_CUES.search(question):
            return None
        for metric, pat in METRIC_PATTERNS:
            if pat.search(question):
                return metric
        return None

    def aliases(self, doc_ids: list[str]) -> set[str]:
        out: set[str] = set()
        for d in doc_ids:
            f = self.filings[d]
            out |= {f.company, f.ticker, f.company.split()[0].rstrip(",")}
            if f.ticker == "C":
                out |= {"Citi", "Citigroup", "Citibank"}
            if f.ticker == "JPM":
                out |= {"JPMorganChase", "JPMorgan", "JPMorgan Chase"}
        return out

    def _query_terms(self, question: str, doc_ids: list[str]) -> str:
        q = question
        for a in sorted(self.aliases(doc_ids), key=len, reverse=True):
            q = re.sub(rf"\b{re.escape(a)}(['’]s)?\b", " ", q, flags=re.I)
        return q

    # -- structured route ------------------------------------------------------
    def _structured(self, question: str, metric: str, scope: list[str]) -> Answer | None:
        years = [int(y) for y in re.findall(r"\b(20\d\d)\b", question)]
        tickers = {self.filings[d].ticker for d in scope}
        if len(tickers) != 1:
            return None
        cands = sorted(
            (f for f in self.filings.values() if f.ticker in tickers and f.financials.get(metric)),
            key=lambda f: -f.fiscal_year,
        )
        if not cands:
            return None
        # Every period any filing reports for this metric, newest first.
        periods = sorted({p for f in cands for p in f.financials[metric]}, reverse=True)
        target = next((p for p in periods if not years or int(p[:4]) in years), None)
        if target is None:
            return None
        # Prefer the filing for that fiscal year itself (the value "as reported");
        # otherwise the newest filing that presents the period as a comparative.
        primary = next((f for f in cands if f.period_end == target), None) or next(
            f for f in cands if target in f.financials[metric]
        )
        series = primary.financials[metric]
        value = series[target]
        concepts = ", ".join(METRICS[metric])
        name = primary.company
        src = [
            Source(
                1,
                f"{primary.doc_id}::{metric}::{target}",
                primary.doc_id,
                f"{name} reported {metric} of {human_money(value, metric)} ({value:,.2f} USD) "
                f"for the fiscal period ending {target} [iXBRL fact, concept in: {concepts}].",
                "xbrl",
            ),
        ]
        text = f"{name} reported {metric} of {human_money(value, metric)} for the fiscal period ending {target} [1]."
        own = sorted(series)
        idx = own.index(target)
        if idx > 0 and series[own[idx - 1]]:
            prev_p = own[idx - 1]
            prev = series[prev_p]
            change = (value - prev) / abs(prev) * 100
            word = "increase" if change >= 0 else "decrease"
            sign_flip = (value < 0) != (prev < 0)
            src.append(
                Source(
                    2,
                    f"{primary.doc_id}::{metric}::{prev_p}",
                    primary.doc_id,
                    f"{name} reported {metric} of {human_money(prev, metric)} ({prev:,.2f} USD) "
                    f"for the prior fiscal period ending {prev_p} [iXBRL fact].",
                    "xbrl",
                )
            )
            src.append(
                Source(
                    3,
                    "derived",
                    primary.doc_id,
                    f"Computed change in {metric} versus the prior fiscal period: "
                    f"({value:,.2f} - {prev:,.2f}) / {abs(prev):,.2f} = {change:.1f}% ({word}).",
                    "derived",
                )
            )
            if sign_flip:
                text += (
                    f" In the prior fiscal period ending {prev_p} it reported {human_money(prev, metric)} [2],"
                    f" so the figure changed sign year over year."
                )
                src.pop()
            else:
                text += (
                    f" In the prior fiscal period ending {prev_p} it reported {human_money(prev, metric)} [2],"
                    f" a {abs(change):.1f}% {word} [3]."
                )
        # Restatement check: does a later filing show a different number for the same period?
        ans_notes: list[str] = []
        for f in cands:
            later = f.financials[metric].get(target)
            if f is primary or later is None or f.fiscal_year <= primary.fiscal_year:
                continue
            if abs(later - value) > max(abs(value) * 0.001, 0.005):
                n = len(src) + 1
                src.append(
                    Source(
                        n,
                        f"{f.doc_id}::{metric}::{target}",
                        f.doc_id,
                        f"The later {f.label} 10-K presents {metric} for the fiscal period ending "
                        f"{target} as {human_money(later, metric)} ({later:,.2f} USD) [iXBRL fact].",
                        "xbrl",
                    )
                )
                text += (
                    f" The later {f.label} 10-K presents {metric} for this period as "
                    f"{human_money(later, metric)} [{n}]."
                )
                ans_notes.append(
                    f"Restatement detected: the {f.label} 10-K reports a different {metric} for the "
                    f"period ending {target} (for example after a stock split or recast). "
                    "The as-reported figure is shown first."
                )
        ans = Answer(question, text, "structured", "xbrl", [primary.doc_id], src)
        ans.notes = ans_notes
        return ans

    # -- main entry point ------------------------------------------------------
    def ask(self, question: str, k: int = 3, strict: bool = False) -> Answer:
        scope = self.retriever.scope(question)
        metric = self.detect_metric(question)
        ans: Answer | None = None
        if metric:
            ans = self._structured(question, metric, scope)
        if ans is None:
            query = self._query_terms(question, scope)
            hits = self.retriever.search_sentences(query, k=k, doc_ids=scope)
            sources = [
                Source(i + 1, h.sentence.sid, h.doc_id, h.sentence.text, "sentence", h.sentence.item)
                for i, h in enumerate(hits)
            ]
            text = (
                self.generator.generate(question, sources)
                if sources
                else ("The provided filings do not address this.")
            )
            ans = Answer(question, text, "narrative", self.generator.name, scope, sources)
        ans.checks = verify_answer(ans.answer, {s.n: s.text for s in ans.sources}, self.aliases(ans.scope))
        if strict:
            kept = [c for c in ans.checks if c.verdict == Verdict.SUPPORTED]
            ans.removed_claims = [c.claim for c in ans.checks if c.verdict != Verdict.SUPPORTED]
            ans.answer = (
                " ".join(f"{c.claim} [{', '.join(c.citations)}]" for c in kept)
                or "No claim in the generated answer could be verified against the filings."
            )
            ans.checks = kept
        return ans
