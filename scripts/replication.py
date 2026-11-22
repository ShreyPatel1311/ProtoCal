#!/usr/bin/env python
"""Independent replication of the material-holdout collapse.

The claim under test: random row-level holdout -- the evaluation protocol used
throughout the ML-for-dental-composites literature -- flatters models that key
on material identity. Hold out a whole material instead and they collapse
toward, or past, the grand mean.

Run identically on two corpora, each restricted to a SINGLE protocol so the
comparison is clean:
  Lodz  (Zenodo 14576242) at its reference protocol: 12 materials
  Domarecka (Zenodo 5569184), an independent study:   9 materials
Both are CC-BY. The second is a different study with different materials, so
agreement between them is replication rather than resampling.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal.dualcure import load as load_dualcure   # noqa: E402
from protocal.stats import canonical_materials        # noqa: E402

MODELS = {
    "ridge":            lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)),
    "random_forest":    lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                                      random_state=0, n_jobs=-1),
    "hist_gbm":         lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                              random_state=0),
    "gaussian_process": lambda: GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(2.0) + WhiteKernel(0.3),
        normalize_y=True, alpha=1e-6, random_state=0),
}


def prepare(df):
    props = sorted(df["property"].unique())
    mats = sorted(df["material"].unique())
    mi = {m: i for i, m in enumerate(mats)}
    df = df.copy()
    df["row"] = df.groupby(["material", "property"]).cumcount()
    wide = df.pivot_table(index=["material", "row"], columns="property",
                          values="value").reset_index()
    Y = np.log(wide[props].to_numpy(float))
    mask = (~np.isnan(Y)).astype(float)
    mcode = np.array([mi[m] for m in wide["material"]])
    return Y, mask, mcode, mats, props


def evaluate(Y, mask, mcode, n_mat, tr, te):
    mu = np.zeros(Y.shape[1]); sd = np.ones(Y.shape[1])
    for k in range(Y.shape[1]):
        v = Y[tr, k][mask[tr, k] > 0]
        if len(v) > 1: mu[k], sd[k] = v.mean(), v.std() or 1.0
    Z = np.nan_to_num((Y - mu) / sd)
    X = np.eye(n_mat)[mcode]
    out = {}

    gm = np.array([Z[tr, k][mask[tr, k] > 0].mean() if (mask[tr, k] > 0).any() else 0.0
                   for k in range(Z.shape[1])])
    rm = lambda p: float(np.sqrt((((p - Z[te]) ** 2) * mask[te]).sum() / max(mask[te].sum(), 1)))
    out["grand_mean"] = rm(np.tile(gm, (len(te), 1)))

    pred = np.tile(gm, (len(te), 1)).copy()
    for k in range(Z.shape[1]):
        for i, r in enumerate(te):
            rows = tr[(mcode[tr] == mcode[r]) & (mask[tr, k] > 0)]
            if len(rows): pred[i, k] = Z[rows, k].mean()
    out["material_mean"] = rm(pred)

    for name, make in MODELS.items():
        pred = np.zeros((len(te), Z.shape[1]))
        for k in range(Z.shape[1]):
            sel = tr[mask[tr, k] > 0]
            if len(sel) < 4: pred[:, k] = gm[k]; continue
            m = make(); m.fit(X[sel], Z[sel, k]); pred[:, k] = m.predict(X[te])
        out[name] = rm(pred)
    return out


def run_corpus(label, df, n_random=8):
    Y, mask, mcode, mats, props = prepare(df)
    n, nm = len(Y), len(mats)
    names = ["grand_mean", "material_mean", *MODELS]
    print(f"\n{'='*74}\n{label}: {n} rows, {nm} materials, {len(props)} properties\n{'='*74}")

    rng = np.random.default_rng(0)
    rand = {k: [] for k in names}
    for _ in range(n_random):
        te = rng.choice(n, int(0.25 * n), replace=False)
        tr = np.setdiff1d(np.arange(n), te)
        for k, v in evaluate(Y, mask, mcode, nm, tr, te).items(): rand[k].append(v)

    lomo = {k: [] for k in names}
    for j in range(nm):
        te = np.where(mcode == j)[0]; tr = np.where(mcode != j)[0]
        for k, v in evaluate(Y, mask, mcode, nm, tr, te).items(): lomo[k].append(v)

    print(f"{'model':<18}{'random holdout':>16}{'leave-one-material':>21}{'collapse':>11}"
          f"{'vs grand mean':>15}")
    res = {}
    gm_lomo = np.mean(lomo["grand_mean"])
    for k in names:
        a, b = np.mean(rand[k]), np.mean(lomo[k])
        flag = "WORSE" if b > gm_lomo + 1e-9 else ""
        print(f"{k:<18}{a:>16.3f}{b:>21.3f}{b - a:>+11.3f}{b - gm_lomo:>+12.3f} {flag}")
        res[k] = {"random_mean": round(a, 4), "random_sd": round(float(np.std(rand[k])), 4),
                  "lomo_mean": round(b, 4), "lomo_sd": round(float(np.std(lomo[k])), 4),
                  "collapse": round(b - a, 4), "vs_grand_mean_lomo": round(b - gm_lomo, 4),
                  "lomo_per_fold": [round(float(x), 4) for x in lomo[k]]}

    # does any identity model still beat the grand mean under material holdout?
    print()
    for k in names:
        if k == "grand_mean": continue
        d = np.array(lomo[k]) - np.array(lomo["grand_mean"])
        nn = len(d); frac = 1 / nm
        se = d.std(ddof=1) * np.sqrt(1 / nn + frac / (1 - frac))
        p = 2 * stats.t.sf(abs(d.mean() / se), nn - 1) if se else float("nan")
        verdict = ("beats mean" if d.mean() < 0 and p < 0.05 else
                   "WORSE than mean" if d.mean() > 0 and p < 0.05 else "no better than mean")
        print(f"   {k:<18}vs grand mean under material holdout: {d.mean():+.3f}  "
              f"p={p:.3f}  -> {verdict}")
        res[k]["p_vs_grand_mean"] = round(float(p), 4)
    return res


def main():
    lodz = canonical_materials(pd.read_csv(ROOT / "output" / "aging_long.csv"))
    ref = lodz[lodz["is_baseline"] == True]  # noqa: E712
    out = {
        "lodz_reference_protocol": run_corpus(
            "CORPUS 1  Lodz @ reference protocol (Zenodo 14576242)", ref),
        "domarecka": run_corpus(
            "CORPUS 2  Domarecka dual-cure, independent study (Zenodo 5569184)",
            load_dualcure()),
    }
    (ROOT / "output" / "replication.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/replication.json")


if __name__ == "__main__":
    main()
