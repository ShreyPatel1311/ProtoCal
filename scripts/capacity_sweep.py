#!/usr/bin/env python
"""Why does b_s recovery fail? Sweep f_shared capacity and the tau treatment.

Hypothesis: with a high-capacity f_shared and few rows, the composition network
absorbs per-study offsets through x (every study has its own formulations), so
b_s is shrunk to zero and tau collapses. If so, recovery should improve as
f_shared capacity drops or tau is held fixed at a sane value.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocal.model import ProtoCal, ProtoCalConfig
from protocal.synthetic import SyntheticSpec, make_synthetic, recovery_metrics, leakage_diagnostic
from protocal.train import train

dev = "cuda" if torch.cuda.is_available() else "cpu"
spec = SyntheticSpec(seed=0)
data, truth = make_synthetic(spec)

rows = []
for hidden in [(128, 128), (64,), (32,), (16,)]:
    for learn_tau, init_tau in [(True, -1.0), (False, torch.log(torch.tensor(0.35)).item())]:
        torch.manual_seed(0)
        cfg = ProtoCalConfig(
            n_composition=spec.n_composition, n_protocol=spec.n_protocol,
            n_studies=spec.n_studies, n_properties=spec.n_properties,
            shared_hidden=hidden, init_log_tau=init_tau, learn_tau=learn_tau,
        )
        m = ProtoCal(cfg); m.set_reference_protocol(truth["p_reference"]); m = m.to(dev)
        train(m, data, epochs=400, device=dev, verbose=False)
        r = recovery_metrics(m, truth)
        rows.append({
            "f_shared_hidden": str(hidden),
            "tau": "learned" if learn_tau else "fixed@0.35",
            "tau_final": round(float(m.log_tau.exp()), 3),
            "g_corr": round(r["g_corr"], 3), "b_corr": round(r["b_corr"], 3),
            "b_rmse": round(r["b_rmse"], 3),
            "leak_r2": round(leakage_diagnostic(m, truth), 3),
        })
        print(f"{str(hidden):>12} {rows[-1]['tau']:>12}  tau={rows[-1]['tau_final']:<6} "
              f"g={rows[-1]['g_corr']:+.3f}  b={rows[-1]['b_corr']:+.3f}  leak={rows[-1]['leak_r2']:.3f}")

Path("output").mkdir(exist_ok=True)
Path("output/capacity_sweep.json").write_text(json.dumps(rows, indent=2))
print("\nwrote output/capacity_sweep.json")
