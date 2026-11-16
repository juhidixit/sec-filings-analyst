"""Hybrid lexical retrieval over sentence-addressable chunks.

Chunks (≈ one paragraph) are scored by a fusion of BM25 and TF-IDF cosine
similarity; the best chunks are then re-scored sentence by sentence so the
answer layer can cite individual sentences rather than whole pages.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .corpus import Filing, Sentence
from .text import content_terms

ITEM_NAMES = {"1": "Business", "1A": "Risk Factors", "7": "MD&A", "7A": "Market Risk"}


@dataclass(frozen=True)
class Chunk:
    key: tuple[str, int]
    doc_id: str
    item: str
    sentences: tuple[Sentence, ...]

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass(frozen=True)
class Hit:
    sentence: Sentence
    doc_id: str
    score: float
    chunk_score: float


class Retriever:
    def __init__(self, filings: list[Filing]) -> None:
        self.filings = {f.doc_id: f for f in filings}
        grouped: dict[tuple[str, int], list[Sentence]] = defaultdict(list)
        for f in filings:
            for s in f.sentences:
                grouped[(f.doc_id, s.chunk)].append(s)
        self.chunks = [Chunk(k, k[0], v[0].item, tuple(v)) for k, v in grouped.items()]
        self._doc_of = np.array([c.doc_id for c in self.chunks])
        docs = [content_terms(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(docs)
        self.tfidf = TfidfVectorizer(analyzer=lambda s: content_terms(s), sublinear_tf=True, min_df=1)
        self.matrix = self.tfidf.fit_transform([c.text for c in self.chunks])

    # -- scoping -------------------------------------------------------------
    def scope(self, question: str) -> list[str]:
        """Restrict to filings whose company/ticker (and year) the question names."""
        q = question.lower()
        companies: set[str] = set()
        for f in self.filings.values():
            names = {f.ticker.lower(), f.company.lower().split()[0].rstrip(",")}
            if f.ticker == "C":
                names = {"citi", "citigroup"}
            if f.ticker == "JPM":
                names |= {"jpmorgan", "jpmorganchase", "jp morgan", "chase"}
            if any(re.search(rf"\b{re.escape(n)}\b", q) for n in names if len(n) > 1):
                companies.add(f.ticker)
        years = {int(y) for y in re.findall(r"\b(?:fy\s?|fiscal\s(?:year\s)?)?(20\d\d)\b", q)}
        docs = [d for d, f in self.filings.items() if (not companies or f.ticker in companies)]
        if years:
            by_year = [d for d in docs if self.filings[d].fiscal_year in years]
            docs = by_year or docs
        elif companies:
            # default to the latest filing of each named company
            latest: dict[str, str] = {}
            for d in docs:
                f = self.filings[d]
                if f.ticker not in latest or f.fiscal_year > self.filings[latest[f.ticker]].fiscal_year:
                    latest[f.ticker] = d
            docs = list(latest.values())
        return docs

    # -- search --------------------------------------------------------------
    def search(
        self,
        question: str,
        k: int = 5,
        doc_ids: list[str] | None = None,
        items: tuple[str, ...] | None = None,
        alpha: float = 0.5,
    ) -> list[Chunk]:
        terms = content_terms(question)
        if not terms:
            return []
        bm = np.asarray(self.bm25.get_scores(terms))
        cos = (self.matrix @ self.tfidf.transform([question]).T).toarray().ravel()
        score = alpha * _minmax(bm) + (1 - alpha) * _minmax(cos)
        if re.search(r"\brisk", question, re.I):
            # Risk questions are best answered from Item 1A.
            score = score * np.where(np.array([c.item == "1A" for c in self.chunks]), 1.25, 1.0)
        mask = np.ones(len(self.chunks), dtype=bool)
        if doc_ids is not None:
            mask &= np.isin(self._doc_of, doc_ids)
        if items:
            mask &= np.array([c.item in items for c in self.chunks])
        score = np.where(mask, score, -1.0)
        order = np.argsort(-score)[:k]
        return [self.chunks[i] for i in order if score[i] > 0]

    def search_sentences(
        self,
        question: str,
        k: int = 3,
        n_chunks: int = 6,
        doc_ids: list[str] | None = None,
        alpha: float = 0.5,
    ) -> list[Hit]:
        chunks = self.search(question, n_chunks, doc_ids, alpha=alpha)
        q = set(content_terms(question))
        hits: list[Hit] = []
        for rank, c in enumerate(chunks):
            cscore = 1.0 / (rank + 1)
            for s in c.sentences:
                st = content_terms(s.text)
                if not st:
                    continue
                overlap = len(q & set(st)) / max(len(q), 1)
                density = len(q & set(st)) / (len(set(st)) ** 0.5)
                hits.append(Hit(s, c.doc_id, 0.6 * overlap + 0.2 * density + 0.2 * cscore, cscore))
        hits.sort(key=lambda h: -h.score)
        return hits[:k]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x, dtype=float) if hi <= lo else (x - lo) / (hi - lo)
