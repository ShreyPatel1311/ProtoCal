#!/usr/bin/env python
"""Model-family comparison across three evaluation regimes.

The claim "this corpus limits every model, not just ours" cannot rest on one
architecture. Here linear, kernel, tree-ensemble and neural families are run on
identical splits, under:

  A  held-out CELLS      -- interpolation within seen materials and protocols
  B  leave-one-MATERIAL-out -- can the model say anything about an unseen
                            composite? A one-hot encoding cannot, by
                            construction; composition descriptors can.
  C  leave-one-PROTOCOL-out -- extrapolation along the protocol axis.

All targets are log-transformed and standardised on TRAINING rows only; sklearn
models are fitted per property over observed rows, so no imputation is needed.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from protocal.composition import feature_matrix, FEATURES   # noqa: E402
from protocal.model import ProtoCal, ProtoCalConfig          # noqa: E402
from protocal.train import train                             # noqa: E402
from experiments import build, standardise, PROPS, DEV       # noqa: E402

K = len(PROPS)


def design(D, encoding):
    """Per-row feature block: material encoding ++ protocol features."""
    if encoding == "onehot":
        Xm = np.eye(len(D["mats"]), dtype=np.float32)[D["mcode"]]
    else:
        Xm = feature_matrix(D["mats"])[D["mcode"]]
    return Xm, D["P"][D["pcode"]].astype(np.float32)


def rmse_masked(pred, Z, mask, idx):
    e = ((pred - Z[idx]) ** 2 * mask[idx]).sum()
    return float(np.sqrt(e / max(mask[idx].sum(), 1)))


def sk_fit_predict(make, Xtr, Ytr, Mtr, Xte):
    """Fit one sklearn regressor per property over observed rows."""
    out = np.zeros((len(Xte), K))
    for k in range(K):
        sel = Mtr[:, k] > 0
        if sel.sum() < 4:
            out[:, k] = 0.0
            continue
        m = make()
        m.fit(Xtr[sel], Ytr[sel, k])
        out[:, k] = m.predict(Xte)
    return out


def protocal_fit_predict(D, encoding, tr, te, Z, epochs=600, seed=0):
    torch.manual_seed(seed)
    Xm, Pr = design(D, encoding)
    pack = lambda idx: {
        "x": torch.tensor(Xm[idx]), "p": torch.tensor(Pr[idx]),
        "study": torch.zeros(len(idx), dtype=torch.long),
        "y": torch.tensor(Z[idx], dtype=torch.float32),
        "mask": torch.tensor(D["mask"][idx])}
    cfg = ProtoCalConfig(n_composition=Xm.shape[1], n_protocol=Pr.shape[1],
                         n_studies=1, n_properties=K, shared_hidden=(64,),
                         proto_hidden=(16,), dropout=0.0,
                         init_log_tau=-4.0, learn_tau=False)
    m = ProtoCal(cfg)
    m.set_reference_protocol(torch.tensor(D["P"][D["ref"]:D["ref"] + 1]))
    m = m.to(DEV)
    train(m, pack(tr), epochs=epochs, device=DEV, verbose=False)
    m.eval(); d = pack(te)
    with torch.no_grad():
        return m(d["x"].to(DEV), d["p"].to(DEV), d["study"].to(DEV)).cpu().numpy()


GP_KERNEL = lambda: ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(0.3)

MODELS = {
    "grand_mean":        ("trivial", None),
    "material_mean":     ("trivial", None),
    "additive_two_way":  ("trivial", None),
    "ridge_comp":        ("sk", "comp", lambda: RidgeCV(alphas=np.logspace(-3, 3, 25))),
    "random_forest":     ("sk", "comp", lambda: RandomForestRegressor(
                             n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    "hist_gbm":          ("sk", "comp", lambda: HistGradientBoostingRegressor(
                             max_iter=300, learning_rate=0.06, random_state=0)),
    "gaussian_process":  ("sk", "comp", lambda: GaussianProcessRegressor(
                             kernel=GP_KERNEL(), normalize_y=True, alpha=1e-6, random_state=0)),
    "protocal_onehot":   ("pc", "onehot", None),
    "protocal_comp":     ("pc", "comp", None),
}


def run_fold(D, tr, te, seed=0):
    Z, _, _ = standardise(D["Y"], D["mask"], tr)
    mask = D["mask"]
    res = {}

    gm = np.array([Z[tr, k][mask[tr, k] > 0].mean() if (mask[tr, k] > 0).any() else 0.0
                   for k in range(K)])
    res["grand_mean"] = rmse_masked(np.tile(gm, (len(te), 1)), Z, mask, te)

    pred = np.tile(gm, (len(te), 1)).copy()
    for k in range(K):
        for i, r in enumerate(te):
            rows = tr[(D["mcode"][tr] == D["mcode"][r]) & (mask[tr, k] > 0)]
            if len(rows): pred[i, k] = Z[rows, k].mean()
    res["material_mean"] = rmse_masked(pred, Z, mask, te)

    a, b = len(D["mats"]), len(D["protos"])
    pred = np.tile(gm, (len(te), 1)).copy()
    for k in range(K):
        sel = tr[mask[tr, k] > 0]
        if len(sel) < 3: continue
        def dm(rows):
            X = np.zeros((len(rows), 1 + (a - 1) + (b - 1))); X[:, 0] = 1
            for i, r in enumerate(rows):
                if D["mcode"][r] > 0: X[i, D["mcode"][r]] = 1
                if D["pcode"][r] > 0: X[i, a - 1 + D["pcode"][r]] = 1
            return X
        beta, *_ = np.linalg.lstsq(dm(sel), Z[sel, k], rcond=None)
        pred[:, k] = dm(te) @ beta
    res["additive_two_way"] = rmse_masked(pred, Z, mask, te)

    for name, spec in MODELS.items():
        if spec[0] == "sk":
            _, enc, make = spec
            Xm, Pr = design(D, enc)
            X = np.hstack([Xm, Pr])
            res[name] = rmse_masked(
                sk_fit_predict(make, X[tr], Z[tr], mask[tr], X[te]), Z, mask, te)
        elif spec[0] == "pc":
            _, enc, _ = spec
            res[name] = rmse_masked(
                protocal_fit_predict(D, enc, tr, te, Z, seed=seed), Z, mask, te)
    return res


def main():
    D = build()
    N = len(D["cell"])
    names = list(MODELS)
    out = {}
    print(f"rows {N} | materials {len(D['mats'])} | protocols {len(D['protos'])} | {DEV}")
    print(f"composition features: {FEATURES}\n")

    regimes = {}
    uniq = np.unique(D["cell"]); rng = np.random.default_rng(0)
    regimes["A_held_out_cells"] = [
        (np.where(~np.isin(D["cell"], h))[0], np.where(np.isin(D["cell"], h))[0])
        for h in (rng.choice(uniq, int(0.25 * len(uniq)), replace=False) for _ in range(8))]
    regimes["B_leave_one_material_out"] = [
        (np.where(D["mcode"] != j)[0], np.where(D["mcode"] == j)[0])
        for j in range(len(D["mats"]))]
    regimes["C_leave_one_protocol_out"] = [
        (np.where(D["pcode"] != j)[0], np.where(D["pcode"] == j)[0])
        for j in range(len(D["protos"]))]

    for reg, folds in regimes.items():
        print(f"=== {reg}  ({len(folds)} folds) ===")
        acc = {n: [] for n in names}
        for s, (tr, te) in enumerate(folds):
            r = run_fold(D, tr, te, seed=s)
            for n in names: acc[n].append(r[n])
        out[reg] = {n: {"mean": round(float(np.mean(v)), 4),
                        "sd": round(float(np.std(v)), 4),
                        "per_fold": [round(float(x), 4) for x in v]} for n, v in acc.items()}
        for n in sorted(names, key=lambda n: np.mean(acc[n])):
            print(f"   {n:<20}{np.mean(acc[n]):.3f} +/- {np.std(acc[n]):.3f}")
        print()

    (ROOT / "output" / "model_comparison.json").write_text(json.dumps(out, indent=2))
    print("wrote output/model_comparison.json")


if __name__ == "__main__":
    main()
