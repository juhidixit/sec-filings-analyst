"""Interactive explorer: grounded Q&A, risk-factor changes, themes and financials.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filings_analyst.answer import Analyst  # noqa: E402
from filings_analyst.corpus import Filing, load_corpus  # noqa: E402
from filings_analyst.diff import (  # noqa: E402
    THEMES,
    change_magnitude,
    diff_risk_factors,
    theme_intensity,
    word_diff,
)
from filings_analyst.verify import Verdict  # noqa: E402

# Palette: validated categorical slots 1-3 + neutral gray (see README "Design").
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
STATUS_STYLE = {
    "new": (BLUE, "circle", "New"),
    "removed": (ORANGE, "x", "Removed"),
    "modified": (AQUA, "diamond", "Modified"),
    "unchanged": (GRAY, "circle-open", "Unchanged"),
}
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"

st.set_page_config(page_title="SEC Filings Analyst", layout="wide")
st.markdown(
    """
<style>
.claim {padding:2px 0; line-height:1.7}
.claim-ok {border-bottom:2px solid #1baf7a}
.claim-bad {background:#fde8e1; border-bottom:2px dashed #eb6834}
.cite {font-size:0.75em; vertical-align:super; color:#256abf; font-weight:600}
.ev {border-left:3px solid #e1e0d9; padding:6px 12px; margin:6px 0; color:#52514e; font-size:0.92em}
.ev mark {background:#cde2fb; color:#0b0b0b; padding:0 2px}
.added {background:#e3f5ee; border-left:3px solid #1baf7a; padding:4px 10px; margin:4px 0; font-size:0.9em}
.removed {background:#fdeee7; border-left:3px solid #eb6834; padding:4px 10px; margin:4px 0; font-size:0.9em; text-decoration:line-through; color:#52514e}
.edited {background:#f3f2ee; border-left:3px solid #2a78d6; padding:4px 10px; margin:4px 0; font-size:0.9em}
.edited ins {background:#d3f1e5; text-decoration:none; padding:0 2px}
.edited del {background:#fbdccf; color:#52514e; padding:0 2px}
.badge {display:inline-block; padding:1px 8px; border-radius:10px; font-size:0.8em; border:1px solid #c3c2b7; margin-right:6px}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load() -> tuple[dict[str, Filing], Analyst]:
    filings = load_corpus(ROOT / "data" / "processed")
    return {f.doc_id: f for f in filings}, Analyst(filings)


FILINGS, ANALYST = load()
SENT_INDEX = {s.sid: (f, i) for f in FILINGS.values() for i, s in enumerate(f.sentences)}


def base_layout(fig: go.Figure, height: int = 420, **kw: object) -> go.Figure:
    layout: dict[str, object] = {
        "height": height,
        "paper_bgcolor": "#fcfcfb",
        "plot_bgcolor": "#fcfcfb",
        "font": {"color": INK2, "size": 13},
        "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
        "hoverlabel": {"bgcolor": "#ffffff", "font_color": INK},
    }
    layout.update(kw)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False)
    return fig


def context_html(sid: str, window: int = 2) -> str:
    if sid not in SENT_INDEX:
        return ""
    f, i = SENT_INDEX[sid]
    target = f.sentences[i]
    sents = [s for s in f.sentences[max(0, i - window) : i + window + 1] if s.chunk == target.chunk]
    parts = [f"<mark>{html.escape(s.text)}</mark>" if s.sid == sid else html.escape(s.text) for s in sents]
    return " ".join(parts)


st.title("SEC Filings Analyst")
st.caption(
    f"{len(FILINGS)} real 10-K filings · "
    f"{sum(len(f.sentences) for f in FILINGS.values()):,} citable sentences · "
    "every answer sentence is cited and machine-checked against its source"
)

tab_ask, tab_diff, tab_themes, tab_fin = st.tabs(
    ["Ask with citations", "What changed in risk factors", "Risk theme map", "Financials (iXBRL)"]
)

# ------------------------------------------------------------------ Ask
with tab_ask:
    examples = [
        "What does NVIDIA say about export controls to China?",
        "What did the OCC find about Citibank's progress on the 2020 Consent Order?",
        "What was NVIDIA's diluted EPS in fiscal 2023?",
        "Where is Tesla building its next Gigafactory?",
        "What would happen if Oracle can no longer host TikTok?",
        "What was JPMorgan's net income in 2024?",
    ]
    c1, c2 = st.columns([3, 1])
    with c2:
        pick = st.selectbox("Examples", ["(type your own)", *examples])
        strict = st.toggle("Strict mode", help="Drop any claim the verifier cannot support.")
        k = st.slider("Evidence sentences", 1, 6, 3)
    with c1:
        q = st.text_input(
            "Question",
            value="" if pick == "(type your own)" else pick,
            placeholder="Ask about any company in the corpus...",
        )
    if q:
        ans = ANALYST.ask(q, k=k, strict=strict)
        st.markdown(
            f"<span class='badge'>route: {ans.route}</span>"
            f"<span class='badge'>generator: {ans.generator}</span>"
            f"<span class='badge'>scope: {', '.join(ans.scope)}</span>"
            f"<span class='badge'>verified: {ans.supported_ratio:.0%} of claims</span>",
            unsafe_allow_html=True,
        )
        out = []
        for c in ans.checks:
            ok = c.verdict == Verdict.SUPPORTED
            icon = "✓" if ok else "⚠"
            tip = "Supported by cited evidence" if ok else "Unsupported: " + "; ".join(c.reasons)
            cites = "".join(f"<span class='cite'>[{n}]</span>" for n in c.citations)
            out.append(
                f"<span class='claim {'claim-ok' if ok else 'claim-bad'}' title='{html.escape(tip)}'>"
                f"{html.escape(c.claim)}{cites} <small>{icon}</small></span>"
            )
        st.markdown(" ".join(out) or html.escape(ans.answer), unsafe_allow_html=True)
        for note in ans.notes:
            st.info(note)
        if ans.removed_claims:
            st.warning(f"Strict mode removed {len(ans.removed_claims)} unverifiable claim(s).")
        bad = [c for c in ans.checks if c.verdict != Verdict.SUPPORTED]
        for c in bad:
            st.error(f"⚠ Unsupported claim: “{c.claim}” — " + "; ".join(c.reasons))

        st.subheader("Evidence")
        for s in ans.sources:
            label = {"sentence": f"Item {s.item}", "xbrl": "iXBRL fact", "derived": "computed"}[s.kind]
            body = context_html(s.ref) if s.kind == "sentence" else html.escape(s.text)
            st.markdown(
                f"**[{s.n}]** `{s.doc_id}` · {label}<div class='ev'>{body}</div>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------ Diff
with tab_diff:
    tickers = sorted(
        {f.ticker for f in FILINGS.values() if sum(g.ticker == f.ticker for g in FILINGS.values()) > 1}
    )
    ticker = st.selectbox("Company", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    fs = sorted((f for f in FILINGS.values() if f.ticker == ticker), key=lambda f: f.fiscal_year)
    prev, curr = fs[-2], fs[-1]
    d = diff_risk_factors(prev.risks(), curr.risks(), prev.label, curr.label)
    counts = d.counts()
    m = st.columns(4)
    for col, key in zip(m, ["new", "removed", "modified", "unchanged"], strict=True):
        col.metric(f"{STATUS_STYLE[key][2]} risk factors", counts[key])

    left, right = st.columns([3, 2])
    with left:
        rows = []
        for c in d.changes:
            rows.append(
                {
                    "status": c.status,
                    "category": c.category,
                    "heading": c.heading,
                    "magnitude": change_magnitude(c),
                    "added": sum(len(s) for s in c.added_sentences),
                    "removed": sum(len(s) for s in c.removed_sentences),
                }
            )
        df = pd.DataFrame(rows)
        fig = go.Figure()
        cats = sorted(df["category"].unique())
        for status, (color, symbol, label) in STATUS_STYLE.items():
            sub = df[df.status == status]
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["magnitude"],
                    y=sub["category"],
                    mode="markers",
                    name=f"{label} ({len(sub)})",
                    marker={
                        "color": color,
                        "symbol": symbol,
                        "size": 13,
                        "line": {"width": 2, "color": "#fcfcfb" if symbol != "circle-open" else color},
                    },
                    customdata=sub[["heading", "added", "removed"]].values,
                    hovertemplate="<b>%{customdata[0]}</b><br>+%{customdata[1]:,} chars added, "
                    "−%{customdata[2]:,} removed<extra>" + label + "</extra>",
                )
            )
        base_layout(
            fig,
            height=max(320, 60 * len(cats)),
            title={
                "text": f"Each risk factor: how much text changed, {prev.label} → {curr.label}",
                "font": {"color": INK, "size": 15},
            },
            legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0},
            margin={"l": 10, "r": 10, "t": 80, "b": 10},
        )
        fig.update_xaxes(title="characters added + removed")
        fig.update_yaxes(categoryorder="array", categoryarray=cats[::-1])
        st.plotly_chart(fig, use_container_width=True)
    with right:
        tp, tc = theme_intensity(prev.risks()), theme_intensity(curr.risks())
        order = sorted(THEMES, key=lambda k: tc[k] - tp[k])
        fig = go.Figure()
        for t in order:
            fig.add_trace(
                go.Scatter(
                    x=[tp[t], tc[t]],
                    y=[t, t],
                    mode="lines",
                    line={"color": AXIS, "width": 2},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[tp[t] for t in order],
                y=order,
                mode="markers",
                name=prev.label,
                marker={"color": GRAY, "size": 10},
                hovertemplate="%{y}: %{x:.1f}% of sentences<extra>" + prev.label + "</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[tc[t] for t in order],
                y=order,
                mode="markers",
                name=curr.label,
                marker={"color": BLUE, "size": 12, "line": {"width": 2, "color": "#fcfcfb"}},
                hovertemplate="%{y}: %{x:.1f}% of sentences<extra>" + curr.label + "</extra>",
            )
        )
        base_layout(
            fig,
            height=480,
            title={
                "text": "Share of risk sentences mentioning each theme",
                "font": {"color": INK, "size": 15},
            },
            legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0},
            margin={"l": 10, "r": 10, "t": 80, "b": 10},
        )
        fig.update_xaxes(title="% of Item 1A sentences", ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Read the changes")
    status_filter = st.multiselect(
        "Show", ["new", "removed", "modified"], default=["new", "removed", "modified"]
    )
    for c in sorted(d.changes, key=change_magnitude, reverse=True):
        if c.status not in status_filter:
            continue
        color, _, label = STATUS_STYLE[c.status]
        with st.expander(f"[{label}] {c.heading[:150]}  ·  {change_magnitude(c):,} chars changed"):
            st.caption(
                f"{c.category} · themes: {', '.join(c.themes) or '—'} · match similarity {c.similarity:.2f}"
            )
            for old, new in c.edited[:12]:
                runs = []
                for op, t in word_diff(old, new):
                    t = html.escape(t)
                    runs.append(t if op == "=" else f"<ins>{t}</ins>" if op == "+" else f"<del>{t}</del>")
                st.markdown(f"<div class='edited'>~ {' '.join(runs)}</div>", unsafe_allow_html=True)
            for s in c.added_sentences[:12]:
                st.markdown(f"<div class='added'>+ {html.escape(s)}</div>", unsafe_allow_html=True)
            for s in c.removed_sentences[:12]:
                st.markdown(f"<div class='removed'>− {html.escape(s)}</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Themes
with tab_themes:
    labels = [f.label for f in sorted(FILINGS.values(), key=lambda f: (f.ticker, f.fiscal_year))]
    by_label = {f.label: f for f in FILINGS.values()}
    z = [[theme_intensity(by_label[lbl].risks())[t] for lbl in labels] for t in THEMES]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=list(THEMES),
            xgap=2,
            ygap=2,
            colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
            colorbar={"title": "% of<br>sentences", "ticksuffix": "%"},
            hovertemplate="%{x}<br>%{y}: %{z:.1f}% of risk sentences<extra></extra>",
        )
    )
    base_layout(
        fig,
        height=560,
        title={
            "text": "How much of each 10-K's risk section is about each theme",
            "font": {"color": INK, "size": 15},
        },
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(z, index=list(THEMES), columns=labels), use_container_width=True)

# ------------------------------------------------------------------ Financials
with tab_fin:
    fl = st.selectbox(
        "Filing",
        sorted(FILINGS),
        index=sorted(FILINGS).index("NVDA-10-K-FY2025") if "NVDA-10-K-FY2025" in FILINGS else 0,
    )
    f = FILINGS[fl]
    metrics = list(f.financials)
    metric = st.radio("Metric", metrics, horizontal=True)
    series = f.financials[metric]
    xs, ys = list(series), list(series.values())
    scale, unit = (1, "$") if metric == "Diluted EPS" else (1e9, "$B")
    fig = go.Figure(
        go.Bar(
            x=xs,
            y=[v / scale for v in ys],
            marker={"color": BLUE, "cornerradius": 4},
            text=[f"{v / scale:,.2f}" for v in ys],
            textposition="outside",
            hovertemplate="period ending %{x}<br>%{y:,.2f} " + unit + "<extra></extra>",
        )
    )
    base_layout(
        fig,
        height=420,
        title={
            "text": f"{f.company} · {metric} ({unit}), as tagged in iXBRL",
            "font": {"color": INK, "size": 15},
        },
        bargap=0.45,
    )
    fig.update_xaxes(type="category", title="fiscal period end")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        pd.DataFrame(f.financials).T.map(lambda v: f"{v:,.2f}" if pd.notna(v) else ""),
        use_container_width=True,
    )
