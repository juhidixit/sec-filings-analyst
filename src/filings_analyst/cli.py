"""Command-line entry point: ``filings-analyst {fetch,build,ask,diff,serve}``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def _fetch(args: argparse.Namespace) -> None:
    from .edgar import EdgarClient

    client = EdgarClient()
    for ticker in args.tickers:
        for ref in client.list_filings(ticker, args.form, args.limit):
            path = client.download(ref, Path(args.out))
            print(f"{ref.ticker} {ref.form} {ref.report_date} -> {path}")


def _fetch_manifest(args: argparse.Namespace) -> None:
    from .edgar import EdgarClient

    client = EdgarClient()
    for row in json.loads(Path(args.manifest).read_text()):
        ref = client.find_filing(row["ticker"], row["report_date"], row["form"])
        path = client.download(ref, Path(args.out))
        print(f"{ref.ticker} {ref.form} {ref.report_date} {ref.accession} -> {path}")


def _build(args: argparse.Namespace) -> None:
    from .corpus import process_filing, save_filing

    for path in sorted(Path(args.raw).glob("*.htm*")):
        filing = process_filing(path.read_bytes(), ticker=path.name.split("_")[0], source=path.name)
        out = save_filing(filing, Path(args.out))
        print(
            f"{filing.doc_id:22s} risk_factors={len(filing.risk_factors):3d} "
            f"sentences={len(filing.sentences):5d} metrics={len(filing.financials):2d} -> {out.name}"
        )


def _ask(args: argparse.Namespace) -> None:
    from .answer import Analyst
    from .corpus import load_corpus

    analyst = Analyst(load_corpus(Path(args.corpus)))
    ans = analyst.ask(" ".join(args.question), strict=args.strict)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2))
        return
    print(f"\n{ans.answer}\n")
    print(f"route={ans.route} generator={ans.generator} scope={','.join(ans.scope)}")
    for s in ans.sources:
        print(f"  [{s.n}] {s.ref}: {s.text[:160]}")
    print("\nverification:")
    for c in ans.checks:
        mark = "OK " if c.verdict.value == "supported" else "!! "
        print(f"  {mark}{c.claim[:100]}  {'; '.join(c.reasons)}")


def _diff(args: argparse.Namespace) -> None:
    from .corpus import load_corpus
    from .diff import change_magnitude, diff_risk_factors

    filings = [f for f in load_corpus(Path(args.corpus)) if f.ticker == args.ticker.upper()]
    if len(filings) < 2:
        sys.exit(f"Need two filings for {args.ticker}; found {len(filings)}.")
    prev, curr = filings[-2], filings[-1]
    d = diff_risk_factors(prev.risks(), curr.risks(), prev.label, curr.label)
    print(f"{prev.label} -> {curr.label}: {d.counts()}")
    for c in sorted(d.changes, key=change_magnitude, reverse=True)[: args.top]:
        print(f"  [{c.status:9s}] {c.heading[:110]}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="filings-analyst", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download filings from SEC EDGAR")
    f.add_argument("tickers", nargs="+")
    f.add_argument("--form", default="10-K")
    f.add_argument("--limit", type=int, default=2)
    f.add_argument("--out", default=str(RAW))
    f.set_defaults(func=_fetch)

    fm = sub.add_parser("fetch-manifest", help="download the exact filings listed in a manifest")
    fm.add_argument("manifest", nargs="?", default=str(ROOT / "data" / "manifest.json"))
    fm.add_argument("--out", default=str(RAW))
    fm.set_defaults(func=_fetch_manifest)

    b = sub.add_parser("build", help="parse raw HTML into the processed corpus")
    b.add_argument("--raw", default=str(RAW))
    b.add_argument("--out", default=str(PROCESSED))
    b.set_defaults(func=_build)

    a = sub.add_parser("ask", help="ask a question with citations + verification")
    a.add_argument("question", nargs="+")
    a.add_argument("--corpus", default=str(PROCESSED))
    a.add_argument("--strict", action="store_true", help="drop unverified claims")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=_ask)

    d = sub.add_parser("diff", help="year-over-year risk factor changes")
    d.add_argument("ticker")
    d.add_argument("--corpus", default=str(PROCESSED))
    d.add_argument("--top", type=int, default=10)
    d.set_defaults(func=_diff)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
