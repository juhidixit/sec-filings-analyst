"""Reproducible evaluation harness.

Run:  python eval/run_eval.py            (writes eval/results/results.{json,md})

Three suites, all measured on the processed corpus in data/processed:

1. Retrieval   - 36 hand-written questions over 9 filings, each with a regex
                 that a correct evidence sentence must match. Reports Hit@k
                 and MRR for BM25-only, TF-IDF-only and the hybrid retriever.
2. Verifier    - synthetic claim perturbations of real filing sentences
                 (wrong number, wrong year, flipped direction, inserted
                 negation, swapped company, wrong evidence) plus faithful
                 controls. Reports precision/recall/F1 for flagging
                 unsupported claims, vs. a lexical-overlap-only baseline.
3. Numeric QA  - "What was <company>'s <metric> in <year>?" for every filing
                 and headline metric. Compares the iXBRL route against
                 retrieval-only answering.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filings_analyst.answer import Analyst, ExtractiveGenerator, human_money  # noqa: E402
from filings_analyst.corpus import Filing, load_corpus  # noqa: E402
from filings_analyst.text import content_terms  # noqa: E402
from filings_analyst.verify import Verdict, check_claim, extract_quantities  # noqa: E402

SEED = 13
OUT = ROOT / "eval" / "results"


# ---------------------------------------------------------------- retrieval
def eval_retrieval(analyst: Analyst) -> dict[str, object]:
    rows = [json.loads(line) for line in (ROOT / "eval" / "qa_narrative.jsonl").open()]
    configs = {"bm25": 1.0, "tfidf": 0.0, "hybrid": 0.5}
    results: dict[str, object] = {"n_questions": len(rows)}
    scope_ok = 0
    per_q: list[dict[str, object]] = []
    for name, alpha in configs.items():
        hits = {1: 0, 3: 0, 5: 0}
        rr: list[float] = []
        lat: list[float] = []
        for r in rows:
            t0 = time.perf_counter()
            scope = analyst.retriever.scope(r["question"])
            query = analyst._query_terms(r["question"], scope)
            ranked = analyst.retriever.search_sentences(query, k=5, n_chunks=10, doc_ids=scope, alpha=alpha)
            lat.append(time.perf_counter() - t0)
            if name == "hybrid":
                scope_ok += r["doc"] in scope
            rank = next(
                (
                    i + 1
                    for i, h in enumerate(ranked)
                    if h.doc_id == r["doc"] and re.search(r["gold"], h.sentence.text, re.I)
                ),
                None,
            )
            for k in hits:
                hits[k] += rank is not None and rank <= k
            rr.append(1.0 / rank if rank else 0.0)
            if name == "hybrid":
                per_q.append(
                    {"id": r["id"], "rank": rank, "top1": ranked[0].sentence.text[:160] if ranked else None}
                )
        n = len(rows)
        results[name] = {
            "hit@1": round(hits[1] / n, 3),
            "hit@3": round(hits[3] / n, 3),
            "hit@5": round(hits[5] / n, 3),
            "mrr": round(sum(rr) / n, 3),
            "median_latency_ms": round(1000 * statistics.median(lat), 1),
        }
    results["scope_accuracy"] = round(scope_ok / len(rows), 3)
    results["per_question"] = per_q
    return results


# ---------------------------------------------------------------- verifier
_DIR_SWAP = {
    "increased": "decreased",
    "decreased": "increased",
    "increase": "decrease",
    "decrease": "increase",
    "higher": "lower",
    "lower": "higher",
    "rose": "fell",
    "fell": "rose",
    "grew": "declined",
    "declined": "grew",
}
_NEG_SITES = re.compile(r"\b(may|could|will|would|can|is|are|was|were|has|have)\b")


def _aliases(f: Filing) -> set[str]:
    a = {f.company, f.ticker, f.company.split()[0].rstrip(",")}
    if f.ticker == "C":
        a |= {"Citi", "Citigroup", "Citibank"}
    if f.ticker == "JPM":
        a |= {"JPMorganChase", "JPMorgan", "JPMorgan Chase"}
    return a


def _perturb_number(s: str, rng: random.Random) -> str | None:
    # Only perturb genuine quantities, not identifiers such as "Form 10-K" or "Rule 10b5-1".
    qs = [
        q
        for q in extract_quantities(s)
        if q.kind in ("money", "percent", "number")
        and not re.search(
            rf"(Form|Rule|Note|Item|Section)\s{re.escape(q.raw)}|{re.escape(q.raw)}-[A-Za-z0-9]", s
        )
    ]
    if not qs:
        return None
    q = rng.choice(qs)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", q.raw)
    if not m:
        return None
    digits = m.group(0)
    val = float(digits.replace(",", ""))
    factor = rng.choice([0.6, 0.75, 0.85, 1.2, 1.35, 1.6])
    new = val * factor
    decimals = len(digits.split(".")[1]) if "." in digits else 0
    new_txt = f"{new:,.{decimals}f}" if "," in digits else f"{new:.{decimals}f}"
    if new_txt == digits:
        return None
    return s.replace(q.raw, q.raw.replace(digits, new_txt, 1), 1)


def _perturb_year(s: str, rng: random.Random) -> str | None:
    years = re.findall(r"\b(20[0-2]\d)\b", s)
    if not years:
        return None
    y = rng.choice(years)
    delta = rng.choice([-2, -1, 1, 2])
    return re.sub(rf"\b{y}\b", str(int(y) + delta), s, count=1)


def _perturb_direction(s: str, rng: random.Random) -> str | None:
    words = [w for w in _DIR_SWAP if re.search(rf"\b{w}\b", s)]
    if not words:
        return None
    w = rng.choice(words)
    return re.sub(rf"\b{w}\b", _DIR_SWAP[w], s, count=1)


def _perturb_negation(s: str, rng: random.Random) -> str | None:
    if re.search(r"\b(not|no|never|without)\b", s, re.I):
        return None
    sites = list(_NEG_SITES.finditer(s))
    if not sites:
        return None
    m = rng.choice(sites)
    return s[: m.end()] + " not" + s[m.end() :]


def _perturb_entity(s: str, own: set[str], others: list[str], rng: random.Random) -> str | None:
    for a in sorted(own, key=len, reverse=True):
        if len(a) > 2 and re.search(rf"\b{re.escape(a)}\b", s):
            return re.sub(rf"\b{re.escape(a)}\b", rng.choice(others), s, count=1)
    return None


def _subspan(s: str) -> str | None:
    parts = re.split(r"[;,]\s", s)
    first = parts[0]
    if len(parts) > 1 and len(first.split()) >= 8:
        return first.rstrip(".") + "."
    return None


def _rounded(s: str) -> str | None:
    """Restate a dollar amount in another unit, e.g. "$14.4 billion" -> "$14,400 million"."""
    m = re.search(r"\$(\d+(?:\.\d+)?) billion", s)
    if m:
        return s.replace(m.group(0), f"${float(m.group(1)) * 1000:,.0f} million", 1)
    m = re.search(r"\$(\d{1,3}(?:,\d{3})+) million", s)
    if m:
        return s.replace(m.group(0), f"${float(m.group(1).replace(',', '')) / 1000:.1f} billion", 1)
    return None


def baseline_lexical(claim: str, evidence: list[str]) -> bool:
    """Coverage-only baseline: flag if < 70% of claim terms are in the evidence."""
    c = content_terms(claim)
    e = set(content_terms(" ".join(evidence)))
    return (sum(t in e for t in c) / len(c) if c else 1.0) < 0.7


def eval_verifier(filings: list[Filing], per_type: int = 120) -> dict[str, object]:
    rng = random.Random(SEED)
    pool = [
        (f, s.text)
        for f in filings
        for s in f.sentences
        if 60 <= len(s.text) <= 400 and not s.text.startswith("•")
    ]
    rng.shuffle(pool)
    companies = sorted({f.company.split()[0].rstrip(",").title() for f in filings} | {"Microsoft", "Amazon"})
    cases: list[dict[str, object]] = []
    want = Counter[str]()

    def add(kind: str, claim: str | None, evidence: list[str], label: bool) -> None:
        if claim and want[kind] < per_type and claim != (evidence[0] if label else ""):
            cases.append({"kind": kind, "claim": claim, "evidence": evidence, "unsupported": label})
            want[kind] += 1

    by_doc: dict[str, list[str]] = defaultdict(list)
    for f, s in pool:
        by_doc[f.doc_id].append(s)
    for f, s in pool:
        ev = [s]
        if want["faithful_verbatim"] < per_type:
            cases.append({"kind": "faithful_verbatim", "claim": s, "evidence": ev, "unsupported": False})
            want["faithful_verbatim"] += 1
        add("faithful_subspan", _subspan(s), ev, False)
        add("faithful_rounded", _rounded(s), ev, False)
        add("wrong_number", _perturb_number(s, rng), ev, True)
        add("wrong_year", _perturb_year(s, rng), ev, True)
        add("flipped_direction", _perturb_direction(s, rng), ev, True)
        add("inserted_negation", _perturb_negation(s, rng), ev, True)
        own = _aliases(f)
        others = [c for c in companies if all(c.lower() not in a.lower() for a in own)]
        add("swapped_entity", _perturb_entity(s, own, others, rng), ev, True)
        other = rng.choice(by_doc[f.doc_id])
        if other != s:
            add("wrong_evidence", s, [other], True)

    tp = fp = fn = tn = 0
    btp = bfp = bfn = btn = 0
    per_kind: dict[str, list[bool]] = defaultdict(list)
    base_kind: dict[str, list[bool]] = defaultdict(list)
    examples: list[dict[str, object]] = []
    for c in cases:
        res = check_claim(str(c["claim"]), list(c["evidence"]))  # type: ignore[arg-type]
        flagged = res.verdict != Verdict.SUPPORTED
        bflag = baseline_lexical(str(c["claim"]), list(c["evidence"]))  # type: ignore[arg-type]
        gold = bool(c["unsupported"])
        per_kind[str(c["kind"])].append(flagged == gold)
        base_kind[str(c["kind"])].append(bflag == gold)
        tp += flagged and gold
        fp += flagged and not gold
        fn += (not flagged) and gold
        tn += (not flagged) and not gold
        btp += bflag and gold
        bfp += bflag and not gold
        bfn += (not bflag) and gold
        btn += (not bflag) and not gold
        if flagged != gold and len(examples) < 12:
            examples.append({"kind": c["kind"], "claim": c["claim"], "reasons": res.reasons})

    def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
        }

    return {
        "n_cases": len(cases),
        "n_by_kind": dict(want),
        "verifier": prf(tp, fp, fn),
        "baseline_lexical_overlap": prf(btp, bfp, bfn),
        "accuracy_by_kind": {k: round(sum(v) / len(v), 3) for k, v in sorted(per_kind.items())},
        "baseline_accuracy_by_kind": {k: round(sum(v) / len(v), 3) for k, v in sorted(base_kind.items())},
        "error_examples": examples,
    }


def eval_verifier_hard(filings: list[Filing]) -> dict[str, object]:
    """30 hand-written claims: faithful paraphrases vs. subtle unsupported claims."""
    sents = {s.sid: s.text for f in filings for s in f.sentences}
    by_doc = {f.doc_id: f for f in filings}
    rows = [json.loads(line) for line in (ROOT / "eval" / "claims_hard.jsonl").open()]
    tp = fp = fn = tn = 0
    misses: list[dict[str, object]] = []
    for r in rows:
        f = by_doc[r["evidence_sid"].split("#")[0]]
        res = check_claim(r["claim"], [sents[r["evidence_sid"]]], aliases=_aliases(f))
        flagged = res.verdict != Verdict.SUPPORTED
        gold = bool(r["unsupported"])
        tp += flagged and gold
        fp += flagged and not gold
        fn += (not flagged) and gold
        tn += (not flagged) and not gold
        if flagged != gold:
            misses.append(
                {"claim": r["claim"], "gold_unsupported": gold, "note": r["note"], "reasons": res.reasons}
            )
    p = tp / (tp + fp) if tp + fp else 0.0
    rcl = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n_cases": len(rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(p, 3),
        "recall": round(rcl, 3),
        "f1": round(2 * p * rcl / (p + rcl), 3) if p + rcl else 0.0,
        "accuracy": round((tp + tn) / len(rows), 3),
        "errors": misses,
    }


# ---------------------------------------------------------------- numeric QA
TEMPLATES = [
    "What was {company}'s {metric} in fiscal {year}?",
    "How much {metric} did {company} report for {year}?",
    "{company} {metric} {year}",
]
METRIC_WORDS = {
    "Revenue": "revenue",
    "Net income": "net income",
    "Operating income": "operating income",
    "R&D expense": "research and development expense",
    "Diluted EPS": "diluted earnings per share",
    "Operating cash flow": "cash from operations",
    "Total assets": "total assets",
}


def _short(company: str) -> str:
    return {
        "Citigroup Inc": "Citigroup",
        "JPMorgan Chase & Co": "JPMorgan",
        "NVIDIA CORP": "NVIDIA",
        "Apple Inc.": "Apple",
        "Tesla, Inc.": "Tesla",
        "Oracle Corporation": "Oracle",
    }.get(company, company)


def _answer_has(answer: str, gold: float, metric: str) -> bool:
    for q in extract_quantities(answer):
        if q.kind not in ("money", "number"):
            continue
        # filings often state values "in millions" without a unit next to the number
        for cand in (q.value, q.value * 1e6):
            if abs(cand - gold) <= max(q.step * (1e6 if cand != q.value else 1) / 2, abs(gold) * 0.001):
                return True
    return False


def eval_numeric(filings: list[Filing]) -> dict[str, object]:
    structured = Analyst(filings, ExtractiveGenerator())
    rows = []
    for f in filings:
        for metric, word in METRIC_WORDS.items():
            series = f.financials.get(metric, {})
            # gold = the value as reported in this filing for its own fiscal year
            target = f.period_end if f.period_end in series else None
            if target is None:
                continue
            year = int(target[:4])
            for t in TEMPLATES:
                rows.append(
                    (f, metric, t.format(company=_short(f.company), metric=word, year=year), series[target])
                )
    s_ok = r_ok = 0
    routed = 0
    failures = []
    for _f, metric, q, gold in rows:
        ans = structured.ask(q)
        routed += ans.route == "structured"
        ok = ans.route == "structured" and _answer_has(ans.answer.split("[1]")[0], gold, metric)
        s_ok += ok
        if not ok and len(failures) < 10:
            failures.append({"q": q, "gold": human_money(gold, metric), "answer": ans.answer[:200]})
        # retrieval-only: force the narrative route on the same question
        scope = structured.retriever.scope(q)
        hits = structured.retriever.search_sentences(structured._query_terms(q, scope), k=3, doc_ids=scope)
        r_ok += any(_answer_has(h.sentence.text, gold, metric) for h in hits)
    n = len(rows)
    return {
        "n_questions": n,
        "routing_rate": round(routed / n, 3),
        "xbrl_route_accuracy": round(s_ok / n, 3),
        "retrieval_only_top3_contains_answer": round(r_ok / n, 3),
        "failures": failures,
    }


def main() -> None:
    filings = load_corpus(ROOT / "data" / "processed")
    analyst = Analyst(filings, ExtractiveGenerator())
    t0 = time.time()
    results = {
        "corpus": {
            "filings": len(filings),
            "sentences": sum(len(f.sentences) for f in filings),
            "risk_factors": sum(len(f.risk_factors) for f in filings),
        },
        "retrieval": eval_retrieval(analyst),
        "verifier": eval_verifier(filings),
        "verifier_hard": eval_verifier_hard(filings),
        "numeric_qa": eval_numeric(filings),
    }
    results["runtime_s"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (OUT / "results.md").write_text(render_md(results))
    print(render_md(results))


def render_md(r: dict) -> str:  # type: ignore[type-arg]
    ret, ver, num = r["retrieval"], r["verifier"], r["numeric_qa"]
    lines = [
        "# Evaluation results",
        "",
        f"Corpus: {r['corpus']['filings']} filings, {r['corpus']['sentences']:,} citable sentences, "
        f"{r['corpus']['risk_factors']} risk factors. Generated by `python eval/run_eval.py` "
        f"(seed {SEED}).",
        "",
        f"## 1. Retrieval ({ret['n_questions']} hand-written questions)",
        "",
        f"Company/year scoping accuracy: **{ret['scope_accuracy']:.0%}**",
        "",
        "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---|---|---|---|",
    ]
    for name in ("bm25", "tfidf", "hybrid"):
        m = ret[name]
        lines.append(f"| {name} | {m['hit@1']:.3f} | {m['hit@3']:.3f} | {m['hit@5']:.3f} | {m['mrr']:.3f} |")
    lines += [
        "",
        f"## 2. Claim verifier ({ver['n_cases']} synthetic cases)",
        "",
        "Positive class = claim is *not* supported by its cited evidence.",
        "",
        "| Method | Precision | Recall | F1 |",
        "|---|---|---|---|",
        f"| Lexical-overlap baseline | {ver['baseline_lexical_overlap']['precision']:.3f} | {ver['baseline_lexical_overlap']['recall']:.3f} | {ver['baseline_lexical_overlap']['f1']:.3f} |",
        f"| **This verifier** | {ver['verifier']['precision']:.3f} | {ver['verifier']['recall']:.3f} | {ver['verifier']['f1']:.3f} |",
        "",
        "| Case type | n | Baseline acc. | Verifier acc. |",
        "|---|---|---|---|",
    ]
    for k, v in ver["accuracy_by_kind"].items():
        lines.append(f"| {k} | {ver['n_by_kind'][k]} | {ver['baseline_accuracy_by_kind'][k]:.3f} | {v:.3f} |")
    hard = r["verifier_hard"]
    lines += [
        "",
        f"### 2b. Hand-written hard set ({hard['n_cases']} claims)",
        "",
        "Faithful *paraphrases* and subtly unsupported claims written by hand against real sentences. "
        "This is where a symbolic verifier is expected to struggle.",
        "",
        f"Accuracy **{hard['accuracy']:.3f}** · precision {hard['precision']:.3f} · recall {hard['recall']:.3f} · "
        f"F1 {hard['f1']:.3f} (TP {hard['tp']}, FP {hard['fp']}, FN {hard['fn']}, TN {hard['tn']})",
        "",
        "| Claim | Gold | Why it failed |",
        "|---|---|---|",
    ]
    for e in hard["errors"]:
        gold = "unsupported" if e["gold_unsupported"] else "supported"
        why = "; ".join(e["reasons"]) or f"missed ({e['note']})"
        lines.append(f"| {e['claim']} | {gold} | {why} |")
    lines += [
        "",
        f"## 3. Numeric questions ({num['n_questions']} questions: {num['n_questions'] // 3} facts × 3 phrasings)",
        "",
        "| Method | Correct |",
        "|---|---|",
        f"| Retrieval only (answer in top-3 sentences) | {num['retrieval_only_top3_contains_answer']:.3f} |",
        f"| iXBRL structured route | {num['xbrl_route_accuracy']:.3f} |",
        "",
        f"Questions routed to iXBRL: {num['routing_rate']:.1%}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
