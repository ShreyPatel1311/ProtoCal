#!/usr/bin/env python
"""Stage 3: pull numeric tables out of cached Europe PMC full-text XML.

The shortlist tells you which papers are worth reading; this turns the chosen
ones into data. JATS XML carries real <table> markup, so the numbers come out
structurally rather than by scraping a PDF layout.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data_work" / "epmc_fulltext"
OUT = ROOT / "data_raw" / "external_papers"
OUT.mkdir(parents=True, exist_ok=True)


def tables(pmcid: str) -> list[tuple[str, pd.DataFrame]]:
    xml = (CACHE / f"{pmcid}.xml").read_text(errors="replace")
    caps = re.findall(r"<label>(Table [^<]+)</label>\s*<caption>(.*?)</caption>",
                      xml, re.S)
    out = []
    try:
        dfs = pd.read_html(xml)          # lxml parses the JATS <table> elements
    except Exception as e:
        print(f"  {pmcid}: no parseable tables ({str(e)[:50]})")
        return out
    for i, df in enumerate(dfs):
        label = caps[i][0] if i < len(caps) else f"Table {i+1}"
        cap = re.sub(r"<[^>]+>", " ", caps[i][1])[:110].strip() if i < len(caps) else ""
        out.append((f"{label}: {cap}", df))
    return out


def main() -> int:
    for pmcid in sys.argv[1:]:
        print(f"\n{'='*78}\n{pmcid}")
        for label, df in tables(pmcid):
            # count plain numerics AND "mean +/- sd" cells, which are strings:
            # filtering on numerics alone silently drops every results table.
            plain_num = df.apply(lambda c: pd.to_numeric(c, errors="coerce")).notna().sum().sum()
            as_text = df.astype(str)
            meansd = as_text.apply(
                lambda c: c.str.contains(r"\d\s*(?:\u00b1|\+/-)\s*\d", regex=True, na=False)
            ).sum().sum()
            numeric = int(plain_num + meansd)
            if numeric < 6:
                continue
            path = OUT / f"{pmcid}_{label.split(':')[0].replace(' ','')}.csv"
            df.to_csv(path, index=False)
            print(f"\n--- {label}")
            print(f"    shape {df.shape}, {numeric} numeric cells -> {path.name}")
            with pd.option_context("display.width", 200, "display.max_columns", 12):
                print(df.head(6).to_string()[:1100])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
