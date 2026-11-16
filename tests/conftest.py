from __future__ import annotations

from pathlib import Path

import pytest

from filings_analyst.answer import Analyst, ExtractiveGenerator
from filings_analyst.corpus import Filing, load_corpus

ROOT = Path(__file__).resolve().parents[1]

MINI_10K = b"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
      xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:xbrldi="http://xbrl.org/2006/xbrldi">
<body>
<div style="display:none"><ix:header><ix:resources>
  <xbrli:context id="FY24"><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="FY23"><xbrli:period><xbrli:startDate>2023-01-01</xbrli:startDate><xbrli:endDate>2023-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="Q4"><xbrli:period><xbrli:startDate>2024-10-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="I24"><xbrli:period><xbrli:instant>2024-12-31</xbrli:instant></xbrli:period></xbrli:context>
  <xbrli:context id="SEG"><xbrli:entity><xbrli:segment><xbrldi:explicitMember dimension="srt:ProductAxis">acme:WidgetsMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate><xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
</ix:resources>
<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="FY24">Acme Robotics Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="FY24">ACME</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="FY24">2024</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="FY24">December 31, 2024</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentType" contextRef="FY24">10-K</ix:nonNumeric>
</ix:header></div>
<div><span>Table of Contents</span></div>
<div><span>Item 1A. Risk Factors</span></div>
<div><span>Item 7. Management's Discussion and Analysis</span></div>
<div><span>Item 1. Business</span></div>
<div><span>Acme builds industrial robots for automotive plants in Michigan and Germany. Our robots weld car bodies.</span></div>
<div><span>Item 1A. Risk Factors</span></div>
<div><span style="font-weight:bold">Risks Related to Our Operations</span></div>
<div><span style="font-weight:700">We depend on a small number of suppliers for servo motors.</span></div>
<div><span>Our servo motors come from two suppliers in Japan. A disruption at either supplier could halt production for several</span></div>
<div><span>12</span></div>
<div><span>weeks and reduce revenue. We have not qualified a third supplier.</span></div>
<p><span style="font-style:italic">Cyberattacks could disrupt our factories.</span><span> Attackers have targeted industrial control systems across the sector. A successful ransomware attack could stop our production lines and expose customer data.</span></p>
<div><span>Item 7. Management's Discussion and Analysis</span></div>
<div><span>Revenue increased by $1.2 billion in 2024 compared with 2023, driven by higher robot shipments to automotive customers.</span></div>
<table><tr><td>Revenue</td><td>$</td><td><ix:nonFraction name="us-gaap:Revenues" contextRef="FY24" unitRef="usd" scale="6" decimals="-6" format="ixt:num-dot-decimal">5,400</ix:nonFraction></td>
<td><ix:nonFraction name="us-gaap:Revenues" contextRef="FY23" unitRef="usd" scale="6" decimals="-6">4,200</ix:nonFraction></td></tr>
<tr><td>Q4 revenue</td><td><ix:nonFraction name="us-gaap:Revenues" contextRef="Q4" unitRef="usd" scale="6">1,500</ix:nonFraction></td></tr>
<tr><td>Widgets</td><td><ix:nonFraction name="us-gaap:Revenues" contextRef="SEG" unitRef="usd" scale="6">999</ix:nonFraction></td></tr>
<tr><td>Net loss</td><td><ix:nonFraction name="us-gaap:NetIncomeLoss" contextRef="FY24" unitRef="usd" scale="6" sign="-">(300</ix:nonFraction>)</td></tr>
<tr><td>Assets</td><td><ix:nonFraction name="us-gaap:Assets" contextRef="I24" unitRef="usd" scale="6">9,000</ix:nonFraction></td></tr></table>
<div><span>Item 8. Financial Statements</span></div>
<div><span>See the consolidated financial statements.</span></div>
</body></html>
"""


@pytest.fixture(scope="session")
def mini_html() -> bytes:
    return MINI_10K


@pytest.fixture(scope="session")
def corpus() -> list[Filing]:
    return load_corpus(ROOT / "data" / "processed")


@pytest.fixture(scope="session")
def analyst(corpus: list[Filing]) -> Analyst:
    return Analyst(corpus, ExtractiveGenerator())
