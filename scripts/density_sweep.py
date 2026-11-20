#!/usr/bin/env python
"""Is b_s recovery limited by information rather than optimisation?

Sweep observations-per-study (rows per study x reporting density). If b_s
recovery is information-limited, b_corr should climb steadily with the number
of observations per (study, property) cell while g_corr stays flat -- g_proto
pools across all studies, b_s does not.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocal.model import ProtoCal, ProtoCalConfig
from protocal.synthetic import SyntheticSpec, make_synthetic, recovery_metrics
from protocal.train import train

dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
configs = [
    ("sparse (as published)", 2, 25, (0.85, 0.45, 0.30, 0.25)),
    ("2x rows",               4, 50, (0.85, 0.45, 0.30, 0.25)),
    ("4x rows",               8, 100, (0.85, 0.45, 0.30, 0.25)),
    ("full reporting",        2, 25, (1.0, 1.0, 1.0, 1.0)),
    ("4x rows + full",        8, 100, (1.0, 1.0, 1.0, 1.0)),
]
for label, lo, hi, prob in configs:
    spec = SyntheticSpec(seed=0, min_rows_per_study=lo, max_rows_per_study=hi,
                         report_prob=prob)
    data, truth = make_synthetic(spec)
    obs_per_cell = float(data["mask"].sum()) / (spec.n_studies * spec.n_properties)
    torch.manual_seed(0)
    cfg = ProtoCalConfig(n_composition=spec.n_composition, n_protocol=spec.n_protocol,
                         n_studies=spec.n_studies, n_properties=spec.n_properties)
    m = ProtoCal(cfg); m.set_reference_protocol(truth["p_reference"]); m = m.to(dev)
    train(m, data, epochs=400, device=dev, verbose=False)
    r = recovery_metrics(m, truth)
    rows.append({"setting": label, "rows": int(data["x"].shape[0]),
                 "obs_per_study_property": round(obs_per_cell, 1),
                 "g_corr": round(r["g_corr"], 3), "b_corr": round(r["b_corr"], 3),
                 "tau": round(float(m.log_tau.exp()), 3)})
    print(f"{label:<22} rows={rows[-1]['rows']:<5} obs/cell={obs_per_cell:5.1f}  "
          f"g={r['g_corr']:+.3f}  b={r['b_corr']:+.3f}  tau={rows[-1]['tau']:.3f}")

Path("output").mkdir(exist_ok=True)
Path("output/density_sweep.json").write_text(json.dumps(rows, indent=2))
print("\nwrote output/density_sweep.json")
