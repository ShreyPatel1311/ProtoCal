#!/usr/bin/env python
"""Headline experiments: baseline ladder, leave-one-protocol-out, and power.

Fixes three defects in the first version of this analysis:
  * standardisation is fit on TRAINING rows only (previously global -> leak);
  * leave-one-protocol-out is run at all (previously never run, although it is
    the only test that speaks to the paper's motivation);
  * the comparison is reported with a Nadeau-Bengio corrected interval, which
    accounts for the train/test overlap that an ordinary paired t ignores.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal.model import ProtoCal, ProtoCalConfig      # noqa: E402
from protocal.stats import canonical_materials           # noqa: E402
from protocal.train import train                         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROPS = ["flexural_strength_MPa", "flexural_modulus_MPa", "diametral_tensile_MPa",
         "vickers_hardness", "fracture_toughness"]


def build():
    df = canonical_materials(pd.read_csv(ROOT / "output" / "aging_long.csv"))
    df = df[df["property"].isin(PROPS) & (df["value"] > 0)].copy()
    mats = sorted(df["material"].unique()); protos = sorted(df["protocol"].unique())
    mi = {m: i for i, m in enumerate(mats)}; pi = {p: i for i, p in enumerate(protos)}

    pf = (df.groupby("protocol").agg(dur=("duration_days", "first"),
                                     temp=("max_temp_C", "first"),
                                     med=("medium", "first"),
                                     base=("is_baseline", "first")).reindex(protos))
    P = np.zeros((len(protos), 5), np.float32)
    P[:, 0] = np.nan_to_num(pf["dur"].to_numpy(float)) / 7.0
    P[:, 1] = (np.nan_to_num(pf["temp"].to_numpy(float), nan=37.0) - 37.0) / 23.0
    P[:, 2] = (pf["med"] == "NaOH").to_numpy(float)
    P[:, 3] = (pf["med"] == "water").to_numpy(float)
    P[:, 4] = (pf["med"] == "NaOH+water").to_numpy(float)
    ref = int(np.argmax(pf["base"].to_numpy(bool)))

    df["row"] = df.groupby(["material", "protocol", "property"]).cumcount()
    wide = df.pivot_table(index=["material", "protocol", "row"],
                          columns="property", values="value").reset_index()
    for c in PROPS:
        if c not in wide: wide[c] = np.nan
    Y = np.log(wide[PROPS].to_numpy(np.float64))     # log scale throughout
    mask = (~np.isnan(Y)).astype(np.float32)
    mcode = np.array([mi[m] for m in wide["material"]])
    pcode = np.array([pi[p] for p in wide["protocol"]])
    return dict(Y=Y, mask=mask, mcode=mcode, pcode=pcode, P=P, ref=ref,
                mats=mats, protos=protos,
                cell=(wide["material"] + " || " + wide["protocol"]).to_numpy())


def standardise(Y, mask, tr):
    """Fit per-property mean/sd on TRAINING rows only."""
    mu = np.zeros(Y.shape[1]); sd = np.ones(Y.shape[1])
    for k in range(Y.shape[1]):
        v = Y[tr, k][mask[tr, k] > 0]
        if len(v) > 1: mu[k], sd[k] = v.mean(), v.std() or 1.0
    Z = np.nan_to_num((Y - mu) / sd)
    return Z, mu, sd


def rmse(pred, Z, mask, idx):
    e = ((pred - Z[idx]) ** 2 * mask[idx]).sum()
    return float(np.sqrt(e / max(mask[idx].sum(), 1)))


def baselines(D, tr, te, Z):
    """Classical estimators, all closed-form."""
    Y, mask, mc, pc = Z, D["mask"], D["mcode"], D["pcode"]
    K = Y.shape[1]
    out = {}

    def wmean(rows, k):
        m = mask[rows, k] > 0
        return Y[rows, k][m].mean() if m.any() else 0.0

    # grand mean
    gm = np.array([wmean(tr, k) for k in range(K)])
    out["grand_mean"] = rmse(np.tile(gm, (len(te), 1)), Y, mask, te)

    # material means
    pred = np.zeros((len(te), K))
    for k in range(K):
        for i, r in enumerate(te):
            rows = tr[mc[tr] == mc[r]]
            pred[i, k] = wmean(rows, k) if len(rows) else gm[k]
    out["material_mean"] = rmse(pred, Y, mask, te)

    # additive two-way: grand + material effect + protocol effect (least squares)
    pred = np.zeros((len(te), K))
    for k in range(K):
        sel = tr[mask[tr, k] > 0]
        if len(sel) < 3: pred[:, k] = gm[k]; continue
        a, b = len(D["mats"]), len(D["protos"])
        X = np.zeros((len(sel), 1 + (a - 1) + (b - 1)))
        X[:, 0] = 1
        for i, r in enumerate(sel):
            if mc[r] > 0: X[i, mc[r]] = 1
            if pc[r] > 0: X[i, a - 1 + pc[r]] = 1
        beta, *_ = np.linalg.lstsq(X, Y[sel, k], rcond=None)
        Xt = np.zeros((len(te), X.shape[1])); Xt[:, 0] = 1
        for i, r in enumerate(te):
            if mc[r] > 0: Xt[i, mc[r]] = 1
            if pc[r] > 0: Xt[i, a - 1 + pc[r]] = 1
        pred[:, k] = Xt @ beta
    out["additive_two_way"] = rmse(pred, Y, mask, te)
    return out


def fit_protocal(D, tr, te, Z, rank=0, epochs=600, seed=0):
    torch.manual_seed(seed)
    X = np.eye(len(D["mats"]), dtype=np.float32)[D["mcode"]]
    Pr = D["P"][D["pcode"]]
    pack = lambda idx: {
        "x": torch.tensor(X[idx]), "p": torch.tensor(Pr[idx]),
        "study": torch.zeros(len(idx), dtype=torch.long),
        "y": torch.tensor(Z[idx], dtype=torch.float32),
        "mask": torch.tensor(D["mask"][idx])}
    cfg = ProtoCalConfig(n_composition=len(D["mats"]), n_protocol=D["P"].shape[1],
                         n_studies=1, n_properties=Z.shape[1], shared_hidden=(64,),
                         proto_hidden=(16,), interaction_rank=rank, dropout=0.0,
                         init_log_tau=-4.0, learn_tau=False)
    m = ProtoCal(cfg); m.set_reference_protocol(torch.tensor(D["P"][D["ref"]:D["ref"]+1]))
    m = m.to(DEV)
    train(m, pack(tr), epochs=epochs, device=DEV, verbose=False)
    m.eval()
    d = pack(te)
    with torch.no_grad():
        pred = m(d["x"].to(DEV), d["p"].to(DEV), d["study"].to(DEV)).cpu().numpy()
    return rmse(pred, Z, D["mask"], te), m


def main():
    D = build()
    N = len(D["cell"])
    print(f"rows {N} | materials {len(D['mats'])} | protocols {len(D['protos'])} | {DEV}")
    res = {}

    # ---------- 1. leave-one-protocol-out ----------
    print("\n=== leave-one-protocol-out (the test the premise requires) ===")
    lopo = []
    for j, proto in enumerate(D["protos"]):
        te = np.where(D["pcode"] == j)[0]; tr = np.where(D["pcode"] != j)[0]
        Z, _, _ = standardise(D["Y"], D["mask"], tr)
        b = baselines(D, tr, te, Z)
        pc_rmse, _ = fit_protocal(D, tr, te, Z)
        row = {"held_out_protocol": proto, "n_test": int(len(te)),
               **{k: round(v, 3) for k, v in b.items()},
               "protocal": round(pc_rmse, 3),
               "is_reference": bool(j == D["ref"])}
        lopo.append(row)
        tag = "  <-- REFERENCE protocol" if j == D["ref"] else ""
        print(f"  {proto[:46]:<46} grand {b['grand_mean']:.3f}  "
              f"addit {b['additive_two_way']:.3f}  ProtoCal {pc_rmse:.3f}{tag}")
    res["lopo"] = lopo

    # ---------- 2. held-out-cell baseline ladder ----------
    print("\n=== held-out cells: baseline ladder (8 x 25% splits) ===")
    uniq = np.unique(D["cell"]); rng = np.random.default_rng(0)
    ladder = {k: [] for k in ["grand_mean", "material_mean", "additive_two_way",
                              "protocal_rank0", "protocal_rank2"]}
    for s in range(8):
        held = rng.choice(uniq, int(0.25 * len(uniq)), replace=False)
        te = np.where(np.isin(D["cell"], held))[0]; tr = np.where(~np.isin(D["cell"], held))[0]
        Z, _, _ = standardise(D["Y"], D["mask"], tr)
        b = baselines(D, tr, te, Z)
        for k, v in b.items(): ladder[k].append(v)
        ladder["protocal_rank0"].append(fit_protocal(D, tr, te, Z, rank=0, seed=s)[0])
        ladder["protocal_rank2"].append(fit_protocal(D, tr, te, Z, rank=2, seed=s)[0])
    for k, v in ladder.items():
        a = np.array(v); print(f"  {k:<20} {a.mean():.3f} +/- {a.std():.3f}")
    res["ladder"] = {k: {"mean": round(float(np.mean(v)), 4),
                         "sd": round(float(np.std(v)), 4),
                         "per_split": [round(float(x), 4) for x in v]}
                     for k, v in ladder.items()}

    # ---------- 3. Nadeau-Bengio corrected comparison ----------
    print("\n=== ProtoCal vs additive, Nadeau-Bengio corrected ===")
    a = np.array(ladder["additive_two_way"]); b = np.array(ladder["protocal_rank0"])
    d = a - b; n = len(d); ratio = 0.25 / 0.75
    se_naive = d.std(ddof=1) / np.sqrt(n)
    se_nb = d.std(ddof=1) * np.sqrt(1 / n + ratio)
    t_nb = d.mean() / se_nb if se_nb else np.nan
    p_nb = float(2 * stats.t.sf(abs(t_nb), n - 1))
    mde = float(stats.t.ppf(0.975, n - 1) * se_nb)
    print(f"  mean gain {d.mean():+.4f} | naive SE {se_naive:.4f} (p={2*stats.t.sf(abs(d.mean()/se_naive), n-1):.3f})")
    print(f"  Nadeau-Bengio SE {se_nb:.4f} -> t={t_nb:.2f}, p={p_nb:.3f}")
    print(f"  minimum detectable effect at n={n}: {mde:.3f}")
    res["comparison"] = {"mean_gain": round(float(d.mean()), 4),
                         "se_naive": round(float(se_naive), 4),
                         "se_nadeau_bengio": round(float(se_nb), 4),
                         "t": round(float(t_nb), 3), "p": round(p_nb, 4),
                         "min_detectable_effect": round(mde, 4)}

    (ROOT / "output" / "experiments.json").write_text(json.dumps(res, indent=2))
    print("\nwrote output/experiments.json")


if __name__ == "__main__":
    main()
