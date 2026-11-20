#!/usr/bin/env python
"""Is b_s information-limited, or estimator-limited?

Finding 3 claimed per-study offsets cannot be recovered from literature-density
data. That conclusion is only sound if the information genuinely is not there.
The decisive test is an ORACLE: hand the estimator the TRUE f_shared and
g_proto, so all that remains is to read b_s off the residuals. If the oracle
recovers b_s well, the information was present and the failure was ours.

Oracle estimator: per-study residual mean, with James-Stein / BLUP shrinkage
    b_hat_s = (n_s * tau^2) / (n_s * tau^2 + sigma^2) * mean_resid_s
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocal.synthetic import SyntheticSpec, make_synthetic

def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))

for label, spec in [
    ("sparse (as published)", SyntheticSpec(seed=0)),
    ("full reporting",        SyntheticSpec(seed=0, report_prob=(1.0,)*4)),
    ("4x rows + full",        SyntheticSpec(seed=0, min_rows_per_study=8,
                                            max_rows_per_study=100, report_prob=(1.0,)*4)),
]:
    data, truth = make_synthetic(spec)
    x, p = data["x"].numpy(), data["p"].numpy()
    y, m = data["y"].numpy(), data["mask"].numpy()
    s = data["study"].numpy()
    b_true = truth["b"].numpy()

    # Oracle residual: remove the TRUE shared and protocol terms.
    resid = y - truth["f_true"](x) - truth["g_true"](p) - truth["inter_true"](x, p)

    K = b_true.shape[1]
    b_mean = np.zeros_like(b_true)
    n_cell = np.zeros_like(b_true)
    for si in range(spec.n_studies):
        sel = s == si
        for k in range(K):
            mk = m[sel, k] > 0
            n_cell[si, k] = mk.sum()
            if mk.sum():
                b_mean[si, k] = resid[sel, k][mk].mean()

    # noise variance from within-study scatter
    sigma2 = np.nanmean([
        np.var(resid[s == si, k][m[s == si, k] > 0], ddof=1)
        for si in range(spec.n_studies) for k in range(K)
        if (m[s == si, k] > 0).sum() > 1
    ])
    tau2 = float(np.var(b_true))
    shrink = (n_cell * tau2) / (n_cell * tau2 + sigma2 + 1e-12)
    b_blup = shrink * b_mean

    obs = m.sum() / (spec.n_studies * K)
    print(f"{label:<24} obs/cell={obs:5.1f}  "
          f"raw-mean r={corr(b_mean, b_true):+.3f}  "
          f"BLUP r={corr(b_blup, b_true):+.3f}  "
          f"(sigma^2={sigma2:.3f}, tau^2={tau2:.3f})")

print("\nIf oracle BLUP r is high, b_s IS recoverable given good f and g,")
print("and Finding 3 is about estimator quality, not information content.")
