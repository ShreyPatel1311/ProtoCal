#!/usr/bin/env python
"""Are the three aged protocols EQUIVALENT? A formal TOST analysis.

"We found no significant difference between protocols" is weak evidence of
redundancy -- absence of evidence. The correct question is the reverse, and it
has a standard answer: two one-sided tests (TOST). We ask whether the
difference between two aged protocols is demonstrably SMALLER than a margin of
practical interest. That is a positive claim, and crucially it does not need
more protocol levels -- it needs enough materials and properties, which pooling
the two Lodz deposits now provides.

Corpus: Zenodo 14576242 (12 materials) + 13165877 (3 materials, same four
protocols) = 15 materials, four protocols, four shared properties.

Margin: declared in advance as a 10% change in property value (0.0953 on the
log scale). We also report the smallest margin at which equivalence would hold,
so the conclusion can be read against any threshold the reader prefers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal.stats import canonical_materials, two_way_anova   # noqa: E402

PROP2024 = {"DTS": "diametral_tensile_MPa", "FS": "flexural_strength_MPa",
            "FT": "fracture_toughness", "HV": "vickers_hardness",
            "FFL": "flexural_fatigue_limit"}
MARGIN_PCT = 10.0
MARGIN = np.log1p(MARGIN_PCT / 100.0)


def load_2024() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data_raw" / "lodz_protocols_2024").glob("*.xlsx")):
        parts = [p.strip() for p in f.stem.split(",")]
        code = parts[0]
        if code not in PROP2024:
            continue
        mat = re.sub(r"\s+", " ", parts[1]).strip()
        mat = {"FlowArt": "Flow Art", "ResinF": "Resin F"}.get(mat.replace(" ", ""), mat)
        proto = ", ".join(p.strip() for p in parts[2:])
        proto = re.sub(r"\s+", " ", proto).strip()
        vals = pd.to_numeric(pd.read_excel(f, header=None).iloc[1:, 0], errors="coerce").dropna()
        for i, v in enumerate(vals):
            rows.append({"material": mat, "property": PROP2024[code], "protocol": proto,
                         "specimen": i + 1, "value": float(v), "study": "lodz2024"})
    return pd.DataFrame(rows)


def main():
    a = canonical_materials(pd.read_csv(ROOT / "output" / "aging_long.csv"))
    a = a[["material", "property", "protocol", "value"]].assign(study="lodz2025")
    b = load_2024()[["material", "property", "protocol", "value", "study"]]
    # normalise protocol strings across deposits
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    a["pkey"] = a["protocol"].map(norm); b["pkey"] = b["protocol"].map(norm)
    shared_p = sorted(set(a["pkey"]) & set(b["pkey"]))
    pooled = pd.concat([a, b], ignore_index=True)
    pooled = pooled[pooled["pkey"].isin(shared_p)]

    print(f"pooled corpus: {len(pooled)} specimens, "
          f"{pooled['material'].nunique()} materials, "
          f"{pooled['pkey'].nunique()} protocols, "
          f"{pooled['property'].nunique()} properties, 2 deposits")
    print(f"  materials: {sorted(pooled['material'].unique())}\n")

    base = [p for p in shared_p if "24 h" in p][0]
    aged = [p for p in shared_p if p != base]
    short = {p: ("NaOH" if p.startswith("0.1m") else
                 "water5d+NaOH" if "5 days" in p else "thermo+NaOH") for p in aged}

    results = []
    print(f"EQUIVALENCE OF THE THREE AGED PROTOCOLS  (TOST, margin = {MARGIN_PCT}% "
          f"= {MARGIN:.4f} log units)\n")
    print(f"{'property':<24}{'pair':<30}{'mean diff':>11}{'90% CI':>19}{'p_TOST':>9}  verdict")
    for prop, g in pooled.groupby("property"):
        cm = np.log(g.groupby(["material", "pkey"])["value"].mean()).unstack()
        if not set(aged).issubset(cm.columns):
            continue
        for i in range(len(aged)):
            for j in range(i + 1, len(aged)):
                d = (cm[aged[i]] - cm[aged[j]]).dropna()
                if len(d) < 4:
                    continue
                n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
                t1 = (m + MARGIN) / se; t2 = (m - MARGIN) / se
                p = max(stats.t.sf(t1, n - 1), stats.t.cdf(t2, n - 1))
                lo, hi = m - stats.t.ppf(0.95, n - 1) * se, m + stats.t.ppf(0.95, n - 1) * se
                smallest = abs(m) + stats.t.ppf(0.95, n - 1) * se
                eq = "EQUIVALENT" if p < 0.05 else "not shown"
                pair = f"{short[aged[i]]} vs {short[aged[j]]}"
                print(f"{prop:<24}{pair:<30}{m:>+11.4f}  [{lo:+.3f},{hi:+.3f}]{p:>9.4f}  {eq}")
                results.append({"property": prop, "pair": pair, "n_materials": n,
                                "mean_log_diff": round(float(m), 4),
                                "ci90": [round(float(lo), 4), round(float(hi), 4)],
                                "p_tost": round(float(p), 4), "equivalent_at_10pct": bool(p < 0.05),
                                "smallest_margin_log": round(float(smallest), 4),
                                "smallest_margin_pct": round(float(np.expm1(smallest) * 100), 1)})

    ok = sum(r["equivalent_at_10pct"] for r in results)
    print(f"\nequivalent at the {MARGIN_PCT}% margin: {ok}/{len(results)} pairwise comparisons")
    worst = max(results, key=lambda r: r["smallest_margin_pct"]) if results else None
    if worst:
        print(f"all pairs are equivalent at a margin of "
              f"{max(r['smallest_margin_pct'] for r in results):.1f}% "
              f"(driven by {worst['property']}, {worst['pair']})")

    # control-vs-aged, for contrast: these should NOT be equivalent
    print(f"\nCONTRAST -- control vs aged (expected to be clearly different):")
    for prop, g in pooled.groupby("property"):
        cm = np.log(g.groupby(["material", "pkey"])["value"].mean()).unstack()
        if base not in cm.columns: continue
        for p_ in aged:
            if p_ not in cm.columns: continue
            d = (cm[base] - cm[p_]).dropna()
            if len(d) < 4: continue
            print(f"   {prop:<24}{'control vs ' + short[p_]:<24}"
                  f"{d.mean():+.4f}  ({np.expm1(abs(d.mean()))*100:.0f}% change)")
            break

    out = {"margin_pct": MARGIN_PCT, "n_specimens": int(len(pooled)),
           "n_materials": int(pooled["material"].nunique()), "results": results}
    (ROOT / "output" / "equivalence.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/equivalence.json")


if __name__ == "__main__":
    main()
