#!/usr/bin/env python
"""Cross-laboratory transfer on the pooled five-laboratory hardness corpus.

Three questions, in the order that decides whether harmonisation is worth doing:

  1. COLD TRANSFER.  Train on four laboratories, predict a fifth never seen,
     with its offset set to zero. This is the honest zero-knowledge case and it
     is what any model trained on scattered literature faces.

  2. ANCHORED TRANSFER.  Same split, but the held-out laboratory's offset is
     estimated from the products it shares with the training laboratories, and
     only the remaining cells are scored. This is the realistic case, and the
     gap between (1) and (2) is the value of harmonisation -- stated as a
     number instead of asserted.

  3. LEAVE-ONE-PRODUCT-OUT.  Can a composite never measured anywhere be
     predicted from its filler loading and the protocol? This is the only
     regime in which a composition model does something a lookup table cannot.

Baselines are deliberately strong: a model that cannot beat "predict this
laboratory's mean" has demonstrated nothing, and on literature-scattered data
that baseline is hard because laboratory offsets dominate.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal import pooled                              # noqa: E402
from protocal.model import ProtoCal, ProtoCalConfig      # noqa: E402
from protocal.train import train                         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PF = pooled.PROTO_FEATURES


def design(df: pd.DataFrame, mu: float, sd: float):
    """Composition block, protocol block, laboratory index."""
    x = ((df["filler_wt_pct"].to_numpy(float) - mu) / sd).reshape(-1, 1)
    p = df[PF].to_numpy(np.float32)
    return x.astype(np.float32), p


def pack(x, p, s, y):
    return {"x": torch.tensor(x), "p": torch.tensor(p),
            "study": torch.tensor(s, dtype=torch.long),
            "y": torch.tensor(y.reshape(-1, 1), dtype=torch.float32),
            "mask": torch.ones(len(y), 1)}


def fit_protocal(xtr, ptr, str_, ytr, n_studies, seed=0, epochs=1200):
    torch.manual_seed(seed)
    cfg = ProtoCalConfig(n_composition=xtr.shape[1], n_protocol=ptr.shape[1],
                         n_studies=n_studies, n_properties=1,
                         shared_hidden=(32, 32), proto_hidden=(16,),
                         dropout=0.0, interaction_rank=1,
                         init_log_tau=-2.0, learn_tau=True)
    m = ProtoCal(cfg)
    m.set_reference_protocol(torch.zeros(1, ptr.shape[1]))
    m = m.to(DEV)
    train(m, pack(xtr, ptr, str_, ytr), epochs=epochs, batch_size=128,
          lr=3e-3, device=DEV, verbose=False)
    return m.eval()


@torch.no_grad()
def predict(m, x, p, s):
    d = pack(x, p, s, np.zeros(len(x)))
    return m(d["x"].to(DEV), d["p"].to(DEV), d["study"].to(DEV)).cpu().numpy().ravel()


@torch.no_grad()
def predict_no_offset(m, x, p):
    """Reference-protocol-free prediction with the study term excluded."""
    d = pack(x, p, np.zeros(len(x), int), np.zeros(len(x)))
    return m(d["x"].to(DEV), d["p"].to(DEV), d["study"].to(DEV),
             include_study_offset=False).cpu().numpy().ravel()


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main() -> int:
    df = pooled.load()
    labs = sorted(df["study"].unique())
    lab_idx = {l: i for i, l in enumerate(labs)}
    df["lab_i"] = df["study"].map(lab_idx)
    y = df["log_hv"].to_numpy(float)
    print(f"cells {len(df)} | products {df['product'].nunique()} | labs {len(labs)} | {DEV}")
    print(f"composition feature: filler_wt_pct | protocol features: {len(PF)}\n")
    res = {"n_cells": int(len(df)), "labs": labs, "proto_features": PF}

    # ---------- 1 & 2. leave-one-laboratory-out ----------
    print("=== leave-one-laboratory-out ===")
    print(f"{'held-out lab':16s} {'n':>4s} {'lab-mean':>9s} {'filler-lm':>9s} "
          f"{'RF':>8s} {'cold':>8s} | {'anchors':>7s} {'n_rest':>6s} {'anchored':>9s}")
    rows = []
    for held in labs:
        te = df["study"] == held
        tr = ~te
        mu = df.loc[tr, "filler_wt_pct"].mean()
        sd = df.loc[tr, "filler_wt_pct"].std() or 1.0
        xtr, ptr = design(df[tr], mu, sd)
        xte, pte = design(df[te], mu, sd)
        ytr, yte = y[tr.to_numpy()], y[te.to_numpy()]

        # baseline: the held-out laboratory's own mean (an oracle it is not
        # fair to beat with, but the number a reader needs to see)
        lab_mean = float(np.mean(yte))
        b_labmean = rmse(np.full_like(yte, lab_mean), yte)
        # baseline: training grand mean
        b_grand = rmse(np.full_like(yte, ytr.mean()), yte)
        # baseline: ridge on [filler, protocol]
        Xtr = np.hstack([xtr, ptr]); Xte = np.hstack([xte, pte])
        lm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xtr, ytr)
        b_lm = rmse(lm.predict(Xte), yte)
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                   random_state=0, n_jobs=-1).fit(Xtr, ytr)
        b_rf = rmse(rf.predict(Xte), yte)

        m = fit_protocal(xtr, ptr, df.loc[tr, "lab_i"].to_numpy(), ytr, len(labs))
        cold = predict_no_offset(m, xte, pte)
        r_cold = rmse(cold, yte)

        # anchored: offset estimated from shared products only
        prod_tr = set(df.loc[tr, "product"])
        is_anchor = df.loc[te, "product"].isin(prod_tr).to_numpy()
        if is_anchor.sum() > 0 and (~is_anchor).sum() > 0:
            off = float(np.mean(yte[is_anchor] - cold[is_anchor]))
            r_anch = rmse(cold[~is_anchor] + off, yte[~is_anchor])
            r_cold_rest = rmse(cold[~is_anchor], yte[~is_anchor])
        else:
            off = np.nan; r_anch = np.nan; r_cold_rest = np.nan

        print(f"{held:16s} {int(te.sum()):4d} {b_labmean:9.3f} {b_lm:9.3f} "
              f"{b_rf:8.3f} {r_cold:8.3f} | {int(is_anchor.sum()):7d} "
              f"{int((~is_anchor).sum()):6d} {r_anch:9.3f}")
        rows.append({"held_out": held, "n_test": int(te.sum()),
                     "rmse_lab_mean_oracle": round(b_labmean, 4),
                     "rmse_train_grand_mean": round(b_grand, 4),
                     "rmse_ridge": round(b_lm, 4), "rmse_rf": round(b_rf, 4),
                     "rmse_protocal_cold": round(r_cold, 4),
                     "n_anchor_cells": int(is_anchor.sum()),
                     "n_rest_cells": int((~is_anchor).sum()),
                     "estimated_offset_log": None if np.isnan(off) else round(off, 4),
                     "rmse_cold_on_rest": None if np.isnan(r_cold_rest) else round(r_cold_rest, 4),
                     "rmse_anchored_on_rest": None if np.isnan(r_anch) else round(r_anch, 4)})
    res["leave_one_lab_out"] = rows

    ok = [r for r in rows if r["rmse_anchored_on_rest"] is not None]
    if ok:
        c = np.mean([r["rmse_cold_on_rest"] for r in ok])
        a = np.mean([r["rmse_anchored_on_rest"] for r in ok])
        print(f"\n  on comparable cells: cold {c:.3f} -> anchored {a:.3f} "
              f"({100*(c-a)/c:+.1f}%)  over {len(ok)} laboratories")
        res["anchoring_gain"] = {"cold": round(float(c), 4), "anchored": round(float(a), 4),
                                 "pct_reduction": round(float(100*(c-a)/c), 2),
                                 "n_labs": len(ok)}

    # ---------- 3. leave-one-product-out ----------
    print("\n=== leave-one-product-out (unseen composite, offsets available) ===")
    prods = sorted(df["product"].unique())
    acc = {"lab_mean": [], "ridge": [], "rf": [], "protocal": [], "n": []}
    for p_out in prods:
        te = df["product"] == p_out
        tr = ~te
        if tr.sum() < 20 or te.sum() == 0:
            continue
        mu = df.loc[tr, "filler_wt_pct"].mean()
        sd = df.loc[tr, "filler_wt_pct"].std() or 1.0
        xtr, ptr = design(df[tr], mu, sd); xte, pte = design(df[te], mu, sd)
        ytr, yte = y[tr.to_numpy()], y[te.to_numpy()]
        str_ = df.loc[tr, "lab_i"].to_numpy(); ste = df.loc[te, "lab_i"].to_numpy()
        # per-laboratory mean of the training products, as the baseline
        labmean = df[tr].groupby("lab_i")["log_hv"].mean()
        base = np.array([labmean.get(s, ytr.mean()) for s in ste])
        Xtr = np.hstack([xtr, ptr]); Xte = np.hstack([xte, pte])
        lm = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=500, min_samples_leaf=2,
                                   random_state=0, n_jobs=-1).fit(Xtr, ytr)
        m = fit_protocal(xtr, ptr, str_, ytr, len(labs))
        acc["lab_mean"].append(rmse(base, yte))
        acc["ridge"].append(rmse(lm.predict(Xte), yte))
        acc["rf"].append(rmse(rf.predict(Xte), yte))
        acc["protocal"].append(rmse(predict(m, xte, pte, ste), yte))
        acc["n"].append(int(te.sum()))
    w = np.array(acc["n"], float)
    print(f"  products held out: {len(w)}  cells: {int(w.sum())}")
    res["leave_one_product_out"] = {}
    for k in ("lab_mean", "ridge", "rf", "protocal"):
        v = np.array(acc[k])
        um, wm = float(v.mean()), float(np.average(v, weights=w))
        print(f"  {k:10s} unweighted {um:.3f}   cell-weighted {wm:.3f}")
        res["leave_one_product_out"][k] = {"unweighted": round(um, 4),
                                           "cell_weighted": round(wm, 4),
                                           "per_product": np.round(v, 4).tolist()}
    res["leave_one_product_out"]["products"] = [
        p for p in prods if (df["product"] == p).sum() and (~(df["product"] == p)).sum() >= 20]

    (ROOT / "output" / "crosslab_train.json").write_text(json.dumps(res, indent=2))
    print("\nwrote output/crosslab_train.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
