#!/usr/bin/env python
"""Mechanistic interpretability: what does g_proto actually learn?

Four probes:
  P1 response curves -- sweep each protocol covariate with the others held at
     the reference, giving the model's learned dose-response per feature;
  P2 fidelity -- learned g_proto at the four observed protocols against the
     empirical protocol main effects estimated directly from cell means;
  P3 dimensionality -- project the learned protocol effect onto the empirical
     singular basis, testing whether the model has also collapsed onto the
     single axis the data contains;
  P4 attribution -- per-prediction decomposition into f_shared, g_proto and
     interaction, showing which term carries the work.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal.model import ProtoCal, ProtoCalConfig    # noqa: E402
from protocal.stats import canonical_materials         # noqa: E402
from protocal.train import train                       # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from experiments import build, standardise, PROPS, DEV  # noqa: E402

FEATURES = ["duration/7d", "(T-37)/23", "medium=NaOH", "medium=water", "medium=NaOH+water"]


def fit_full(D, Z, rank=0, epochs=800, seed=0):
    torch.manual_seed(seed)
    X = np.eye(len(D["mats"]), dtype=np.float32)[D["mcode"]]
    Pr = D["P"][D["pcode"]]
    data = {"x": torch.tensor(X), "p": torch.tensor(Pr),
            "study": torch.zeros(len(X), dtype=torch.long),
            "y": torch.tensor(Z, dtype=torch.float32),
            "mask": torch.tensor(D["mask"])}
    cfg = ProtoCalConfig(n_composition=len(D["mats"]), n_protocol=D["P"].shape[1],
                         n_studies=1, n_properties=Z.shape[1], shared_hidden=(64,),
                         proto_hidden=(16,), interaction_rank=rank, dropout=0.0,
                         init_log_tau=-4.0, learn_tau=False)
    m = ProtoCal(cfg)
    m.set_reference_protocol(torch.tensor(D["P"][D["ref"]:D["ref"] + 1]))
    m = m.to(DEV)
    train(m, data, epochs=epochs, device=DEV, verbose=False)
    return m.eval(), data


def main():
    D = build()
    tr = np.arange(len(D["cell"]))
    Z, mu, sd = standardise(D["Y"], D["mask"], tr)
    model, data = fit_full(D, Z)
    out = {}

    # ---- P1: response curves -------------------------------------------------
    grid = np.linspace(-0.5, 2.0, 40)
    curves = {}
    for fi, fname in enumerate(FEATURES):
        P = np.tile(D["P"][D["ref"]], (len(grid), 1)).astype(np.float32)
        P[:, fi] = grid
        with torch.no_grad():
            g = model.protocol_effect(torch.tensor(P).to(DEV)).cpu().numpy()
        curves[fname] = {"grid": grid.round(3).tolist(),
                         "effect": {p: g[:, k].round(4).tolist() for k, p in enumerate(PROPS)}}
    out["response_curves"] = curves

    # ---- P2: fidelity against empirical protocol main effects ----------------
    df = canonical_materials(pd.read_csv(ROOT / "output" / "aging_long.csv"))
    df = df[df["property"].isin(PROPS) & (df["value"] > 0)]
    emp = {}
    for k, prop in enumerate(PROPS):
        g = df[df["property"] == prop]
        cm = np.log(g.groupby(["material", "protocol"])["value"].mean()).unstack()
        cm = cm.reindex(columns=D["protos"])
        # protocol main effect, centred on the reference protocol, in sd units
        eff = (cm.mean(0) - cm.mean(0).iloc[D["ref"]]) / sd[k]
        emp[prop] = eff.to_numpy()
    with torch.no_grad():
        g_hat = model.protocol_effect(torch.tensor(D["P"]).to(DEV)).cpu().numpy()
    fid = {}
    for k, prop in enumerate(PROPS):
        e, h = emp[prop], g_hat[:, k]
        slope = float(np.dot(e, h) / max(np.dot(e, e), 1e-12))
        fid[prop] = {"empirical": e.round(3).tolist(), "learned": h.round(3).tolist(),
                     "calibration_slope": round(slope, 3),
                     "rmse": round(float(np.sqrt(((e - h) ** 2).mean())), 4)}
    out["fidelity"] = fid
    out["protocols"] = D["protos"]

    # ---- P3: dimensionality of the learned protocol effect -------------------
    E = np.stack([emp[p] for p in PROPS], 1)       # protocols x properties
    H = g_hat
    def spectrum(M):
        s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
        return (s ** 2 / max((s ** 2).sum(), 1e-12)).round(4).tolist()
    out["dimensionality"] = {"empirical_var_share": spectrum(E),
                             "learned_var_share": spectrum(H)}

    # ---- P4: attribution -----------------------------------------------------
    with torch.no_grad():
        x, p = data["x"].to(DEV), data["p"].to(DEV)
        f = model.f_shared(x).cpu().numpy()
        g = model.protocol_effect(p).cpu().numpy()
        i = model.interaction_effect(x, p).cpu().numpy()
    m_ = D["mask"]
    def share(A): return float((np.abs(A) * m_).sum() / max(m_.sum(), 1))
    out["attribution_mean_abs"] = {"f_shared": round(share(f), 4),
                                   "g_proto": round(share(g), 4),
                                   "interaction": round(share(i), 4)}

    (ROOT / "output" / "interpretability.json").write_text(json.dumps(out, indent=2))

    print("=== P2: learned vs empirical protocol effect (sd units) ===")
    print(f"{'property':<24}{'slope':>8}{'rmse':>8}   learned / empirical per protocol")
    for prop in PROPS:
        f_ = fid[prop]
        print(f"{prop:<24}{f_['calibration_slope']:>8.2f}{f_['rmse']:>8.3f}   "
              + " ".join(f"{a:+.2f}/{b:+.2f}" for a, b in zip(f_['learned'], f_['empirical'])))
    print("\n=== P3: variance share by component ===")
    print("  empirical:", out["dimensionality"]["empirical_var_share"])
    print("  learned:  ", out["dimensionality"]["learned_var_share"])
    print("\n=== P4: mean |contribution| ===")
    for k, v in out["attribution_mean_abs"].items(): print(f"  {k:<14}{v:.4f}")
    print("\nwrote output/interpretability.json")


if __name__ == "__main__":
    main()
