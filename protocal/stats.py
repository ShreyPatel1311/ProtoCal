"""Correct variance decomposition and interaction tests for the aging corpus.

Replaces the invalid `interaction_vs_noise` ratio used in the first audit, which
divided interaction SS by cell COUNT rather than interaction degrees of freedom
and compared cell-mean residuals against single-specimen variance. Its null
expectation is ~0.08-0.11, not 1, so "1.42 means interaction exceeds noise" did
not follow.

Here: Type-II sums of squares by nested model comparison on the log scale, a
proper F test for the interaction, and random-effects variance components by
the method of moments (expected mean squares).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# "Grandio SO Heavy Flow 2" is a filename artifact: it occupies exactly and
# only the baseline-protocol cells that "Grandio SO Heavy Flow" is missing.
# The source Zenodo abstract states twelve materials, not thirteen.
MATERIAL_ALIASES = {"Grandio SO Heavy Flow 2": "Grandio SO Heavy Flow"}


def canonical_materials(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["material"] = out["material"].replace(MATERIAL_ALIASES)
    return out


def _lstsq_ss(y: np.ndarray, X: np.ndarray) -> float:
    """Residual sum of squares of the least-squares fit of y on X."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def _dummies(codes: np.ndarray, n_levels: int) -> np.ndarray:
    """Reference-coded dummies (first level dropped)."""
    D = np.zeros((len(codes), max(n_levels - 1, 0)))
    for i, c in enumerate(codes):
        if c > 0:
            D[i, c - 1] = 1.0
    return D


def two_way_anova(df: pd.DataFrame, prop: str) -> dict:
    """Replicated two-way ANOVA (material x protocol) on log values.

    Uses specimen-level replicates, so the interaction is tested against true
    within-cell error rather than being confounded with it.
    """
    g = df[(df["property"] == prop) & (df["value"] > 0)]
    mats = sorted(g["material"].unique())
    protos = sorted(g["protocol"].unique())
    mi = {m: i for i, m in enumerate(mats)}
    pi = {p: i for i, p in enumerate(protos)}

    y = np.log(g["value"].to_numpy(float))
    mc = np.array([mi[m] for m in g["material"]])
    pc = np.array([pi[p] for p in g["protocol"]])
    n, a, b = len(y), len(mats), len(protos)

    one = np.ones((n, 1))
    Dm, Dp = _dummies(mc, a), _dummies(pc, b)
    # interaction columns = elementwise products of the two dummy blocks
    Di = np.column_stack([Dm[:, i] * Dp[:, j] for i in range(a - 1) for j in range(b - 1)]) \
        if (a > 1 and b > 1) else np.zeros((n, 0))

    X_null = one
    X_m = np.hstack([one, Dm])
    X_mp = np.hstack([one, Dm, Dp])
    X_full = np.hstack([one, Dm, Dp, Di])

    ss_null = _lstsq_ss(y, X_null)
    ss_m = _lstsq_ss(y, X_m)
    ss_mp = _lstsq_ss(y, X_mp)
    ss_full = _lstsq_ss(y, X_full)

    df_full = n - np.linalg.matrix_rank(X_full)
    if df_full <= 0:
        return {"property": prop, "status": "no residual df"}
    ms_err = ss_full / df_full

    df_int = np.linalg.matrix_rank(X_full) - np.linalg.matrix_rank(X_mp)
    df_m = np.linalg.matrix_rank(X_m) - 1
    df_p = np.linalg.matrix_rank(X_mp) - np.linalg.matrix_rank(X_m)

    ss_int = ss_mp - ss_full
    ss_mat = ss_null - ss_m
    ss_pro = ss_m - ss_mp

    F_int = (ss_int / df_int) / ms_err if df_int > 0 else np.nan
    p_int = float(stats.f.sf(F_int, df_int, df_full)) if df_int > 0 else np.nan
    F_pro = (ss_pro / df_p) / ms_err if df_p > 0 else np.nan
    p_pro = float(stats.f.sf(F_pro, df_p, df_full)) if df_p > 0 else np.nan

    # Random-effects variance components by expected mean squares.
    n_bar = n / (a * b)
    ms_m, ms_p, ms_i = ss_mat / df_m, ss_pro / df_p, ss_int / df_int
    v_i = max((ms_i - ms_err) / n_bar, 0.0)
    v_m = max((ms_m - ms_i) / (b * n_bar), 0.0)
    v_p = max((ms_p - ms_i) / (a * n_bar), 0.0)
    v_e = ms_err
    tot = v_m + v_p + v_i + v_e

    return {
        "property": prop, "n": int(n), "materials": a, "protocols": b,
        "F_interaction": round(float(F_int), 3), "df_interaction": int(df_int),
        "df_error": int(df_full), "p_interaction": float(p_int),
        "F_protocol": round(float(F_pro), 3), "p_protocol": float(p_pro),
        "vc_material_pct": round(100 * v_m / tot, 1),
        "vc_protocol_pct": round(100 * v_p / tot, 1),
        "vc_interaction_pct": round(100 * v_i / tot, 1),
        "vc_residual_pct": round(100 * v_e / tot, 1),
    }


def protocol_dof(df: pd.DataFrame) -> dict:
    """How many protocol degrees of freedom does this corpus really have?

    If the aged protocols are near-duplicates of each other, the nominal count
    of protocol levels overstates what any protocol-structured model can learn.
    """
    cells = (canonical_materials(df)
             .assign(logv=lambda d: np.log(d["value"].clip(lower=1e-9)))
             .groupby(["property", "material", "protocol"])["logv"].mean()
             .unstack())
    protos = list(cells.columns)
    M = cells.dropna().to_numpy()
    out = {"protocols": protos, "pairwise_mean_abs_log_diff": {}}
    for i in range(len(protos)):
        for j in range(i + 1, len(protos)):
            d = float(np.abs(M[:, i] - M[:, j]).mean())
            out["pairwise_mean_abs_log_diff"][f"{protos[i]} VS {protos[j]}"] = round(d, 4)
    sv = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    out["singular_values"] = [round(float(s), 3) for s in sv]
    out["variance_explained_by_first_component_pct"] = round(
        100 * float(sv[0] ** 2 / (sv ** 2).sum()), 1)
    return out
