#!/usr/bin/env python
"""Does effective protocol dimensionality PREDICT the benefit of conditioning?

The 99.7% figure on the Lodz corpus is one observation. On its own it is a
story fitted to a dataset. The diagnostic earns its place only if it makes a
falsifiable prediction that survives testing:

    as effective protocol dimensionality rises, the benefit of continuous
    protocol features over dummy coding should rise with it.

We generate synthetic corpora whose protocol vectors are confined to a
subspace of known dimension k, measure the diagnostic exactly as we do on real
data, and measure the benefit by leave-one-protocol-out. If the diagnostic is
real, benefit tracks it. If the relationship is flat, the diagnostic is
describing nothing and should be dropped from the paper.

Composition x is continuous here (as real composition descriptors are), so
both models can in principle generalise across materials; the only thing that
varies is the protocol geometry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal.model import ProtoCal, ProtoCalConfig   # noqa: E402
from protocal.train import train                       # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PROTO, N_MAT, N_PROP, PROTO_DIM, N_PER_CELL = 8, 14, 4, 5, 6


def make_corpus(true_k: int, seed: int):
    """Protocol vectors confined to a `true_k`-dimensional subspace."""
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(true_k, PROTO_DIM))
    coords = rng.normal(size=(N_PROTO, true_k))
    P = coords @ basis
    P = (P - P.mean(0)) / (P.std(0) + 1e-9)
    P[0] = 0.0                                    # reference protocol
    X = rng.normal(size=(N_MAT, 6))               # continuous composition

    W1 = rng.normal(0, 1.0, (6, 24)); W2 = rng.normal(0, 0.8, (24, N_PROP))
    G1 = rng.normal(0, 0.9, (PROTO_DIM, 8)); G2 = rng.normal(0, 0.7, (8, N_PROP))
    f = lambda x: np.tanh(x @ W1) @ W2
    g_raw = lambda p: np.tanh(p @ G1) @ G2
    g = lambda p: g_raw(p) - g_raw(P[0:1])

    rows = []
    for mi in range(N_MAT):
        for pi in range(N_PROTO):
            mu = f(X[mi:mi + 1]) + g(P[pi:pi + 1])
            for _ in range(N_PER_CELL):
                rows.append((mi, pi, mu[0] + rng.normal(0, 0.25, N_PROP)))
    mcode = np.array([r[0] for r in rows]); pcode = np.array([r[1] for r in rows])
    Y = np.stack([r[2] for r in rows]).astype(np.float32)
    return X.astype(np.float32), P.astype(np.float32), mcode, pcode, Y


def diagnostic(Y, mcode, pcode):
    """PC1 variance share of the protocol main-effect matrix -- exactly the
    statistic reported for the real corpus."""
    E = np.stack([Y[pcode == j].mean(0) for j in range(N_PROTO)])
    E = E - E.mean(0, keepdims=True)
    s = np.linalg.svd(E, compute_uv=False)
    return float((s ** 2 / (s ** 2).sum())[0])


def lopo(X, P, mcode, pcode, Y, seed):
    """Leave-one-protocol-out: dummy coding vs continuous protocol features."""
    d_err, c_err = [], []
    for held in range(N_PROTO):
        te = pcode == held; tr = ~te
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        Z = (Y - mu) / sd

        # dummy coding: material dummies + protocol dummies (held-out level absent)
        a, b = N_MAT, N_PROTO
        def dm(sel):
            D = np.zeros((sel.sum(), 1 + (a - 1) + (b - 1))); D[:, 0] = 1
            for i, (m_, p_) in enumerate(zip(mcode[sel], pcode[sel])):
                if m_ > 0: D[i, m_] = 1
                if p_ > 0: D[i, a - 1 + p_] = 1
            return D
        beta, *_ = np.linalg.lstsq(dm(tr), Z[tr], rcond=None)
        d_err.append(float(np.sqrt(((dm(te) @ beta - Z[te]) ** 2).mean())))

        # continuous protocol features
        torch.manual_seed(seed)
        cfg = ProtoCalConfig(n_composition=X.shape[1], n_protocol=PROTO_DIM,
                             n_studies=1, n_properties=N_PROP, shared_hidden=(64,),
                             proto_hidden=(16,), dropout=0.0,
                             init_log_tau=-4.0, learn_tau=False)
        m = ProtoCal(cfg); m.set_reference_protocol(torch.tensor(P[0:1])); m = m.to(DEV)
        pack = lambda sel: {
            "x": torch.tensor(X[mcode[sel]]), "p": torch.tensor(P[pcode[sel]]),
            "study": torch.zeros(int(sel.sum()), dtype=torch.long),
            "y": torch.tensor(Z[sel]), "mask": torch.ones(int(sel.sum()), N_PROP)}
        train(m, pack(tr), epochs=400, device=DEV, verbose=False)
        m.eval(); t = pack(te)
        with torch.no_grad():
            pr = m(t["x"].to(DEV), t["p"].to(DEV), t["study"].to(DEV)).cpu().numpy()
        c_err.append(float(np.sqrt(((pr - Z[te]) ** 2).mean())))
    return float(np.mean(d_err)), float(np.mean(c_err))


def main():
    rows = []
    print(f"{'true k':>7}{'diagnostic PC1':>16}{'dummy':>9}{'continuous':>12}{'benefit':>10}")
    for k in [1, 2, 3, 4, 5]:
        for seed in [0, 1, 2]:
            X, P, mc, pc, Y = make_corpus(k, seed)
            diag = diagnostic(Y, mc, pc)
            d, c = lopo(X, P, mc, pc, Y, seed)
            rows.append({"true_k": k, "seed": seed, "pc1_share": round(diag, 4),
                         "dummy_rmse": round(d, 4), "continuous_rmse": round(c, 4),
                         "benefit": round(d - c, 4)})
            print(f"{k:>7}{diag:>16.3f}{d:>9.3f}{c:>12.3f}{d - c:>+10.3f}")

    pc1 = np.array([r["pc1_share"] for r in rows])
    ben = np.array([r["benefit"] for r in rows])
    r_p, p_p = stats.pearsonr(pc1, ben)
    r_s, p_s = stats.spearmanr(pc1, ben)
    print(f"\nPREDICTION: benefit should FALL as PC1 share RISES (lower effective d.o.f.)")
    print(f"  Pearson  r = {r_p:+.3f}  p = {p_p:.4f}")
    print(f"  Spearman r = {r_s:+.3f}  p = {p_s:.4f}")
    print(f"  benefit at k=1: {ben[pc1 > np.median(pc1)].mean():+.3f}   "
          f"at high d.o.f.: {ben[pc1 <= np.median(pc1)].mean():+.3f}")
    verdict = "SUPPORTED" if (r_p < 0 and p_p < 0.05) else "NOT SUPPORTED"
    print(f"  -> diagnostic {verdict}")

    (ROOT / "output" / "diagnostic_validation.json").write_text(json.dumps(
        {"rows": rows, "pearson_r": round(float(r_p), 4), "pearson_p": float(p_p),
         "spearman_r": round(float(r_s), 4), "spearman_p": float(p_s),
         "verdict": verdict}, indent=2))
    print("\nwrote output/diagnostic_validation.json")


if __name__ == "__main__":
    main()
