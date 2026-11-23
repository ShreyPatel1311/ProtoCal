"""Loader for the Kruger microhardness corpus (Dryad doi:10.5061/dryad.68dq135, CC0).

An INDEPENDENT laboratory (Rostock) and by far the widest protocol design
available: 7 storage media x 5 incubation times x 3 composites of differing
filler content, n = 10 per cell, Vickers microhardness.

  media  artificial saliva; lactic acid pH 3; lactic acid pH 5;
         citric acid pH 3; citric acid pH 5; ethanol 40 vol%; ethanol 60 vol%
  times  0, 14, 30, 90, 180 days

Day 0 is a single shared baseline that the spreadsheet repeats inside every
medium block, so it is emitted once per material rather than seven times --
treating those repeats as independent observations would fabricate data.

Layout per block: a row carrying the medium name, five specimen rows, an
interleaved sub-header, then five more specimen rows. Time points sit in
columns 2, 5, 8, 11, 14.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SHEETS = {"MHV AraFlow": "Arabesk Flow", "MHV AraTop": "Arabesk Top",
          "MHV GrandioSO": "GrandioSO"}
MEDIA = ["artificial saliva", "lactic acid pH 3", "lactic acid pH 5",
         "citric acid pH 3", "citric acid pH 5", "ethanol 40 Vol%", "ethanol 60 Vol%"]
TIMES = [0, 14, 30, 90, 180]
VALUE_COLS = [2, 5, 8, 11, 14]
# filler loading (wt%) from the source publication, for composition features
FILLER_WT = {"Arabesk Flow": 58.0, "Arabesk Top": 77.0, "GrandioSO": 89.0}


def load(root: Path | None = None) -> pd.DataFrame:
    base = (root or Path(__file__).resolve().parent.parent)
    xl = pd.ExcelFile(base / "data_raw" / "kruger_dryad" / "data.xls")
    rows = []
    for sheet, material in SHEETS.items():
        df = xl.parse(sheet, header=None)
        col0 = df.iloc[:, 0].astype(str).str.strip()
        for medium in MEDIA:
            hits = col0[col0 == medium].index
            if not len(hits):
                continue
            start = int(hits[0])
            block = df.iloc[start + 1: start + 12]
            for ti, (t, vc) in enumerate(zip(TIMES, VALUE_COLS)):
                if vc >= df.shape[1]:
                    continue
                vals = pd.to_numeric(block.iloc[:, vc], errors="coerce").dropna()
                for i, v in enumerate(vals):
                    rows.append({"material": material, "medium": medium,
                                 "days": t, "specimen": i + 1,
                                 "value": float(v), "property": "vickers_microhardness",
                                 "filler_wt_pct": FILLER_WT[material]})
    out = pd.DataFrame(rows)

    # Day 0 is one shared baseline repeated in every medium block: keep one copy.
    base0 = (out[out["days"] == 0]
             .drop_duplicates(subset=["material", "specimen"])
             .assign(medium="baseline"))
    aged = out[out["days"] > 0]
    out = pd.concat([base0, aged], ignore_index=True)
    out["protocol"] = out.apply(
        lambda r: "baseline (0 d)" if r["days"] == 0 else f"{r['medium']}, {r['days']} d",
        axis=1)
    return out
