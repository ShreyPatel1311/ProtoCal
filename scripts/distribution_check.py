#!/usr/bin/env python
"""Is a Gaussian likelihood appropriate for these properties?

Brittle-material strength is classically Weibull-distributed (failure governed
by the worst flaw, not the average one). ProtoCal as drafted assumes Gaussian
noise. If strength data is really Weibull, the Gaussian NLL is mis-specified
and the robust-z outlier flags in the audit are partly an artefact of forcing
a symmetric distribution onto a skewed one.

Compares Normal vs Weibull vs Lognormal by AIC, pooled within each
(property, material, protocol) cell after centring, plus skewness.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "output" / "aging_long.csv")

def aic(loglik: float, k: int) -> float:
    return 2 * k - 2 * loglik

rows = []
for (prop, mat, proto), g in df.groupby(["property", "material", "protocol"]):
    v = g["value"].to_numpy(float)
    if len(v) < 5 or v.min() <= 0:
        continue
    try:
        n_p = stats.norm.fit(v)
        a_n = aic(stats.norm.logpdf(v, *n_p).sum(), 2)
        w_p = stats.weibull_min.fit(v, floc=0)
        a_w = aic(stats.weibull_min.logpdf(v, *w_p).sum(), 2)
        l_p = stats.lognorm.fit(v, floc=0)
        a_l = aic(stats.lognorm.logpdf(v, *l_p).sum(), 2)
    except Exception:
        continue
    best = min([("normal", a_n), ("weibull", a_w), ("lognormal", a_l)], key=lambda t: t[1])
    rows.append({"property": prop, "material": mat, "protocol": proto, "n": len(v),
                 "skew": float(stats.skew(v)),
                 "shapiro_p": float(stats.shapiro(v).pvalue) if len(v) >= 3 else np.nan,
                 "aic_normal": a_n, "aic_weibull": a_w, "aic_lognormal": a_l,
                 "best": best[0], "weibull_k": float(w_p[0])})

res = pd.DataFrame(rows)
out = {}
print(f"cells analysed: {len(res)}\n")
print("best-fitting distribution by property (AIC):")
for prop, g in res.groupby("property"):
    c = Counter(g["best"])
    tot = len(g)
    share = {k: round(100 * v / tot, 1) for k, v in c.most_common()}
    med_skew = float(g["skew"].median())
    frac_nonnormal = float((g["shapiro_p"] < 0.05).mean())
    med_k = float(g["weibull_k"].median())
    out[prop] = {"n_cells": tot, "best_share_pct": share,
                 "median_skew": round(med_skew, 2),
                 "frac_shapiro_reject": round(frac_nonnormal, 3),
                 "median_weibull_shape_k": round(med_k, 1)}
    print(f"  {prop:<24} {share}")
    print(f"  {'':24} median skew {med_skew:+.2f} | Shapiro rejects normality in "
          f"{frac_nonnormal:.0%} of cells | median Weibull k={med_k:.1f}")

# Overall: does Weibull beat Normal more often than chance for strength props?
strength = res[res["property"].isin(["flexural_strength_MPa", "diametral_tensile_MPa"])]
wins = int((strength["aic_weibull"] < strength["aic_normal"]).sum())
print(f"\nstrength cells where Weibull beats Normal by AIC: {wins}/{len(strength)} "
      f"({100*wins/max(len(strength),1):.0f}%)")
out["_strength_weibull_wins"] = {"wins": wins, "total": int(len(strength))}

(ROOT / "output" / "distribution_check.json").write_text(json.dumps(out, indent=2))
res.to_csv(ROOT / "output" / "distribution_per_cell.csv", index=False)
print("\nwrote output/distribution_check.json")
