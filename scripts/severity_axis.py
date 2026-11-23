#!/usr/bin/env python
"""Keystone test: is the protocol-severity axis's per-material GAIN set by filler?

Two findings have to be reconciled. Protocol effects are nearly one-dimensional
(PC1 ~ 97% of between-protocol variance). Yet materials differ in how hard a
given challenge hits them, and that difference tracks filler loading. Those are
only compatible if one-dimensionality means a single severity DIRECTION with a
material-specific GAIN along it:

    log y_{i,s}  ~  a_i  +  gain_i * severity_s

Row-centre the material x protocol matrix of log values -- which removes a_i
exactly -- and take the SVD. The leading left singular vector is then the
per-material gain and the leading right singular vector the protocol severity
ordering. The test is whether gain correlates with filler wt%.

A sign convention is needed: severity is fixed to point in the direction of
DEGRADATION (so a harsher protocol has a more negative severity score), which
makes a positive gain mean "this material degrades in the usual direction".
Under that convention more filler should mean LESS degradation, i.e. a negative
gain-vs-filler slope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal import alrahlah, kruger, randolph                   # noqa: E402
from protocal.pooled import PRODUCT_ALIASES, _filler_table        # noqa: E402
from protocal.stats import canonical_materials                    # noqa: E402

RNG = np.random.default_rng(0)


def decompose(M: np.ndarray):
    """Row-centred SVD. Returns gain, severity, variance ratios."""
    C = M - M.mean(axis=1, keepdims=True)
    u, s, vt = np.linalg.svd(C, full_matrices=False)
    var = s**2 / (s**2).sum()
    gain, sev = u[:, 0] * s[0], vt[0]
    # orient severity toward degradation: the protocol with the lowest mean
    # value should score most negative
    if np.corrcoef(sev, M.mean(axis=0))[0, 1] < 0:
        gain, sev = -gain, -sev
    return gain, sev, var


def permutation_pc1(M: np.ndarray, n: int = 20000) -> dict:
    """Null for PC1 share: shuffle values independently within each row.

    This destroys any shared protocol direction while preserving each
    material's own marginal distribution, so it answers the only question that
    matters -- whether a PC1 share this large can arise from a matrix of this
    shape with no common axis at all.
    """
    C = M - M.mean(axis=1, keepdims=True)
    obs = (np.linalg.svd(C, compute_uv=False)[0] ** 2) / ((np.linalg.svd(C, compute_uv=False) ** 2).sum())
    null = np.empty(n)
    for k in range(n):
        P = np.array([RNG.permutation(r) for r in M])
        Cp = P - P.mean(axis=1, keepdims=True)
        sv = np.linalg.svd(Cp, compute_uv=False)
        null[k] = sv[0] ** 2 / (sv**2).sum()
    return {"pc1_observed": round(float(obs), 4),
            "null_mean": round(float(null.mean()), 4),
            "null_p95": round(float(np.percentile(null, 95)), 4),
            "p_value": round(float((null >= obs).mean()), 5),
            "n_perm": n}


def corpus_lodz():
    d = canonical_materials(pd.read_csv(ROOT / "output" / "aging_long.csv"))
    d = d[(d["property"] == "vickers_hardness") & (d["value"] > 0)]
    w = d.groupby(["material", "protocol"])["value"].mean().unstack()
    return np.log(w.dropna()), "lodz_2025 (HV)"


def corpus_alrahlah():
    a = alrahlah.load_hardness()
    a = a[a["value"].notna()].copy()
    a["proto"] = np.where(a["days"] == 0, "dry 0d",
                          a["medium"] + " " + a["days"].astype(int).astype(str) + "d")
    w = a.groupby(["material", "proto"])["value"].mean().unstack()
    return np.log(w.dropna()), "alrahlah_2013 (HV)"


def corpus_kruger():
    k = kruger.load()
    k = k[k["value"] > 0].copy()
    k["proto"] = k["medium"] + " " + k["days"].astype(int).astype(str) + "d"
    w = k.groupby(["material", "proto"])["value"].mean().unstack()
    return np.log(w.dropna()), "kruger_2018 (HV)"


def main() -> int:
    filler = _filler_table()
    out = {}
    for fn in (corpus_lodz, corpus_alrahlah, corpus_kruger):
        W, name = fn()
        M = W.to_numpy(float)
        mats = [PRODUCT_ALIASES.get(m, m) for m in W.index]
        gain, sev, var = decompose(M)
        print(f"\n=== {name} :  {M.shape[0]} materials x {M.shape[1]} protocols ===")
        print(f"  PC1 {100*var[0]:.1f}%   PC2 {100*var[1]:.1f}%")
        perm = permutation_pc1(M, n=4000)
        print(f"  permutation null: mean {100*perm['null_mean']:.1f}%  "
              f"p95 {100*perm['null_p95']:.1f}%  ->  p = {perm['p_value']}")

        f = np.array([filler.get(m, np.nan) for m in mats])
        ok = np.isfinite(f)
        rec = {"shape": list(M.shape), "pc1": round(float(var[0]), 4),
               "pc2": round(float(var[1]), 4), "permutation": perm,
               "materials": mats,
               "gain": np.round(gain, 4).tolist(),
               "severity_order": dict(zip(W.columns.astype(str),
                                          np.round(sev, 4).tolist()))}
        if ok.sum() >= 4:
            lr = stats.linregress(f[ok], gain[ok])
            print(f"  GAIN vs filler wt%:  n={int(ok.sum())}  slope={lr.slope:+.5f}  "
                  f"r={lr.rvalue:+.3f}  p={lr.pvalue:.4f}")
            rec["gain_vs_filler"] = {"n": int(ok.sum()), "slope": round(float(lr.slope), 5),
                                     "r": round(float(lr.rvalue), 4),
                                     "p": round(float(lr.pvalue), 5)}
        else:
            print(f"  GAIN vs filler wt%: only {int(ok.sum())} materials with filler -- skipped")
            rec["gain_vs_filler"] = None
        print("  severity ordering (most protective -> most aggressive):")
        for p, v in sorted(zip(W.columns.astype(str), sev), key=lambda t: -t[1])[:10]:
            print(f"     {v:+.3f}  {p}")
        out[name] = rec

    (ROOT / "output" / "severity_axis.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/severity_axis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
