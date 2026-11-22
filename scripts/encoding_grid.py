#!/usr/bin/env python
"""Factorial model-family x encoding grid.

scripts/model_comparison.py confounded two factors: every sklearn model was
given composition descriptors while ProtoCal appeared under one-hot identity,
so "random forest beats ProtoCal" changed family AND encoding at once. This
runs every family under BOTH encodings on identical splits, so the two main
effects separate.
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from model_comparison import (design, rmse_masked, sk_fit_predict,       # noqa: E402
                              protocal_fit_predict, MODELS)
from experiments import build, standardise, PROPS                         # noqa: E402

FAMILIES = {k: v[2] for k, v in MODELS.items() if v[0] == "sk"}
ENCODINGS = ["onehot", "comp"]


def run(D, tr, te, seed):
    Z, _, _ = standardise(D["Y"], D["mask"], tr)
    out = {}
    for enc in ENCODINGS:
        Xm, Pr = design(D, enc)
        X = np.hstack([Xm, Pr])
        for fam, make in FAMILIES.items():
            out[f"{fam}|{enc}"] = rmse_masked(
                sk_fit_predict(make, X[tr], Z[tr], D["mask"][tr], X[te]), Z, D["mask"], te)
        out[f"protocal|{enc}"] = rmse_masked(
            protocal_fit_predict(D, enc, tr, te, Z, seed=seed), Z, D["mask"], te)
    return out


def main():
    D = build()
    uniq = np.unique(D["cell"]); rng = np.random.default_rng(0)
    regimes = {
        "A_held_out_cells": [
            (np.where(~np.isin(D["cell"], h))[0], np.where(np.isin(D["cell"], h))[0])
            for h in (rng.choice(uniq, int(0.25*len(uniq)), replace=False) for _ in range(8))],
        "B_leave_one_material_out": [
            (np.where(D["mcode"] != j)[0], np.where(D["mcode"] == j)[0])
            for j in range(len(D["mats"]))],
    }
    fams = list(FAMILIES) + ["protocal"]
    res = {}
    for reg, folds in regimes.items():
        acc = {}
        for s, (tr, te) in enumerate(folds):
            for k, v in run(D, tr, te, s).items():
                acc.setdefault(k, []).append(v)
        res[reg] = {k: {"mean": round(float(np.mean(v)), 4),
                        "sd": round(float(np.std(v)), 4),
                        "per_fold": [round(float(x), 4) for x in v]} for k, v in acc.items()}

        print(f"=== {reg} ({len(folds)} folds) ===")
        print(f"{'family':<20}{'one-hot':>12}{'composition':>14}{'encoding gain':>15}{'p':>8}")
        frac = 0.25 if reg.startswith("A") else 1/len(D["mats"])
        n = len(folds)
        for f in fams:
            a = np.array(acc[f"{f}|onehot"]); b = np.array(acc[f"{f}|comp"])
            d = a - b
            se = d.std(ddof=1)*np.sqrt(1/n + frac/(1-frac))
            p = 2*stats.t.sf(abs(d.mean()/se), n-1) if se else float("nan")
            print(f"{f:<20}{a.mean():>12.3f}{b.mean():>14.3f}{d.mean():>+15.3f}{p:>8.3f}")
        # family main effect, averaged over encodings
        print(f"\n  family means (averaged over both encodings):")
        for f in sorted(fams, key=lambda f: np.mean(acc[f"{f}|onehot"]+acc[f"{f}|comp"])):
            m = np.mean(acc[f"{f}|onehot"] + acc[f"{f}|comp"])
            print(f"    {f:<20}{m:.3f}")
        print()
    (ROOT/"output"/"encoding_grid.json").write_text(json.dumps(res, indent=2))
    print("wrote output/encoding_grid.json")


if __name__ == "__main__":
    main()
