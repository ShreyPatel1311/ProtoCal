#!/usr/bin/env python
"""Headline results under every filler definition.

Filler loading is the single composition feature in this project, so how it is
defined is not a detail. Randolph reports a manufacturer figure, a TGA
inorganic residue (W_i) and an acetone-insoluble fraction (W_total) for the
same material, and for prepolymerised-filler products they differ by up to
12 wt%. Those products are disproportionately the cross-laboratory anchors, so
the definition propagates straight into both headline analyses.

Reports, for each variant:
  A. the paired filler x solvent slope (the primary claim)
  B. the per-laboratory calibration sweep, median and best subset
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal import alrahlah, filler, pooled, randolph   # noqa: E402

import itertools


def slope_paired(x, d):
    lr = stats.linregress(x, d)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(10000):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2:
            continue
        bs.append(np.polyfit(x[i], d[i], 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(n=int(len(x)), slope=round(float(lr.slope), 5),
                r=round(float(lr.rvalue), 3), p=round(float(lr.pvalue), 5),
                ci=[round(float(lo), 5), round(float(hi), 5)])


def part_a(variant: str) -> dict:
    """Paired within-material water-vs-ethanol slope on filler wt%."""
    ft = filler.table(variant, pooled.PRODUCT_ALIASES).set_index("product")
    out = {}
    r = randolph.load()
    for prop in ("vickers_hardness", "flexural_strength_MPa", "flexural_modulus_GPa"):
        w = (r[r["property"] == prop]
             .pivot_table(index="material", columns="medium", values="value")
             .reset_index())
        w["product"] = w["material"].replace(pooled.PRODUCT_ALIASES)
        w["f"] = w["product"].map(ft["filler_wt_pct"])
        w = w.dropna(subset=["water", "ethanol75", "f"])
        if len(w) >= 5:
            out[f"randolph:{prop}"] = slope_paired(
                w["f"].to_numpy(float),
                np.log(w["ethanol75"].to_numpy()) - np.log(w["water"].to_numpy()))
    A = alrahlah.load_hardness()
    A = A[A["value"].notna()].copy()
    A["product"] = A["material"].replace(pooled.PRODUCT_ALIASES)
    aw = A[(A["medium"] == "distilled_water") & (A["days"] == 30)].set_index("product")
    ae = A[(A["medium"] == "ethanol_75pct_in_water") & (A["days"] == 30)].set_index("product")
    common = [p for p in aw.index if p in ae.index and np.isfinite(
        ft["filler_wt_pct"].get(p, np.nan))]
    if len(common) >= 5:
        out["alrahlah:vickers_hardness"] = slope_paired(
            np.array([ft["filler_wt_pct"][p] for p in common], float),
            np.log(ae.loc[common, "value"].to_numpy())
            - np.log(aw.loc[common, "value"].to_numpy()))
    return out


def part_b(variant: str, aqueous: bool) -> dict:
    df = pooled.load(filler_variant=variant)
    if aqueous:
        df = pooled.aqueous_family(df)
    y = df["log_hv"].to_numpy(float)
    F = pooled.PROTO_FEATURES
    res = {}
    for held in sorted(df["study"].unique()):
        te = (df["study"] == held).to_numpy(); tr = ~te
        tr_prod = set(df.loc[tr, "product"]); held_prod = set(df.loc[te, "product"])
        anch = sorted(held_prod & tr_prod); test = sorted(held_prod - tr_prod)
        if not anch or not test:
            continue
        mu = df.loc[tr, "filler_wt_pct"].mean(); sd = df.loc[tr, "filler_wt_pct"].std() or 1.0
        D = lambda m: np.hstack([
            ((df.loc[m, "filler_wt_pct"].to_numpy(float) - mu) / sd).reshape(-1, 1),
            df.loc[m, F].to_numpy(float)])
        model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(D(tr), y[tr])
        ph = model.predict(D(te)); yh = y[te]; prod = df.loc[te, "product"].to_numpy()
        is_t = np.isin(prod, test)
        unc = float(np.sqrt(np.mean((ph[is_t] - yh[is_t]) ** 2)))
        rm = []
        for k in range(1, len(anch) + 1):
            for S in itertools.combinations(anch, k):
                m = np.isin(prod, S)
                off = float(np.mean(yh[m] - ph[m]))
                rm.append(float(np.sqrt(np.mean((ph[is_t] + off - yh[is_t]) ** 2))))
        rm = np.array(rm)
        res[held] = dict(n_anchor=len(anch), n_test=len(test),
                         uncal=round(unc, 4), best=round(float(rm.min()), 4),
                         median=round(float(np.median(rm)), 4),
                         worst=round(float(rm.max()), 4),
                         pct_worse=round(float(100 * (rm > unc).mean()), 1),
                         median_better=bool(np.median(rm) < unc))
    return res


def main() -> int:
    out = {}
    print("=" * 96)
    print("A. PRIMARY CLAIM: paired within-material slope of log(HV_eth/HV_water) on filler wt%")
    print("=" * 96)
    keys = None
    for v in ("manufacturer", "measured", "measured_wtotal"):
        a = part_a(v); out.setdefault(v, {})["slope"] = a
        if keys is None:
            keys = list(a)
            print(f"{'variant':17s}" + "".join(f"{k.split(':')[0][:4]}:{k.split(':')[1][:9]:>12s}" for k in keys))
        print(f"{v:17s}" + "".join(
            f"{a[k]['slope']:+.4f}{'*' if a[k]['p']<0.05 else ' '}{'':4s}" for k in keys))
    print("\n  detail (n, r, p, bootstrap 95% CI):")
    for v in ("manufacturer", "measured", "measured_wtotal"):
        print(f"   {v}")
        for k, s in out[v]["slope"].items():
            print(f"     {k:38s} n={s['n']:2d} slope={s['slope']:+.5f} r={s['r']:+.3f} "
                  f"p={s['p']:.4f} CI[{s['ci'][0]:+.4f},{s['ci'][1]:+.4f}]")

    print("\n" + "=" * 96)
    print("B. CALIBRATION SWEEP, aqueous/ethanol family (the well-posed subcorpus)")
    print("=" * 96)
    print(f"{'variant':17s}{'lab':15s}{'uncal':>8s}{'best':>8s}{'median':>8s}"
          f"{'worst':>8s}{'%worse':>8s}  median_better?")
    for v in ("manufacturer", "measured", "measured_wtotal"):
        b = part_b(v, aqueous=True); out[v]["calibration_aqueous"] = b
        for lab, r in b.items():
            print(f"{v:17s}{lab:15s}{r['uncal']:8.3f}{r['best']:8.3f}{r['median']:8.3f}"
                  f"{r['worst']:8.3f}{r['pct_worse']:7.0f}%  "
                  f"{'YES' if r['median_better'] else 'no'}")
        print()
    (ROOT / "output" / "filler_sensitivity.json").write_text(json.dumps(out, indent=2))
    print("wrote output/filler_sensitivity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
