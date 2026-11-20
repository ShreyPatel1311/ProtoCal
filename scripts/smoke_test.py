#!/usr/bin/env python
"""GPU smoke test: synthetic-recovery precondition check for ProtoCal.

Runs the full pipeline at small scale to shake out errors before scaling up:
  1. generate literature-like synthetic data (sparse, one protocol per study)
  2. fit ProtoCal
  3. check g_proto / b_s recovery against ground truth
  4. run the b_s ~ p_s leakage diagnostic
  5. compare harmonisation against a naive pooled baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocal.model import ProtoCal, ProtoCalConfig          # noqa: E402
from protocal.synthetic import (                              # noqa: E402
    SyntheticSpec, make_synthetic, recovery_metrics, leakage_diagnostic,
)
from protocal.train import train, masked_rmse                 # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--studies", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/smoke_test.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device
    print(f"device: {dev}")
    if dev.startswith("cuda"):
        print(f"gpu:    {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    spec = SyntheticSpec(n_studies=args.studies, seed=args.seed)
    data, truth = make_synthetic(spec)
    print(f"rows:   {data['x'].shape[0]}  studies: {spec.n_studies}  "
          f"observed cells: {int(data['mask'].sum())}/{data['mask'].numel()} "
          f"({100*float(data['mask'].mean()):.0f}% dense)")

    cfg = ProtoCalConfig(
        n_composition=spec.n_composition,
        n_protocol=spec.n_protocol,
        n_studies=spec.n_studies,
        n_properties=spec.n_properties,
    )
    model = ProtoCal(cfg)
    model.set_reference_protocol(truth["p_reference"])
    model = model.to(dev)
    # gauge fix must hold exactly, on device, before training
    with torch.no_grad():
        g0 = model.protocol_effect(truth["p_reference"].to(dev)).abs().max()
    assert float(g0) < 1e-6, f"gauge fixing broken: |g(p_ref)|={float(g0)}"
    print(f"gauge:  |g_proto(p_ref)| = {float(g0):.2e}  (must be 0)")

    t0 = time.time()
    hist = train(model, data, epochs=args.epochs, batch_size=args.batch_size,
                 device=dev, log_every=max(1, args.epochs // 4))
    train_s = time.time() - t0

    rec = recovery_metrics(model, truth)
    leak = leakage_diagnostic(model, truth)

    # harmonisation vs. naive pooling: can we predict the reference-protocol
    # truth for each formulation, without ever seeing it?
    with torch.no_grad():
        f_ref_hat = model.predict_reference(data["x"].to(dev)).cpu()
    import numpy as np
    f_ref_true = torch.tensor(truth["f_true"](data["x"].numpy()), dtype=torch.float32)
    # naive pooling = take reported values at face value
    ones = torch.ones_like(data["mask"])
    harm_rmse = float(masked_rmse(f_ref_hat, f_ref_true, ones).mean())
    naive_rmse = float(masked_rmse(data["y"], f_ref_true, data["mask"]).mean())

    res = {
        "device": dev,
        "gpu": torch.cuda.get_device_name(0) if dev.startswith("cuda") else None,
        "rows": int(data["x"].shape[0]),
        "studies": spec.n_studies,
        "epochs": args.epochs,
        "train_seconds": round(train_s, 1),
        "peak_gpu_mb": (round(torch.cuda.max_memory_allocated() / 1e6, 1)
                        if dev.startswith("cuda") else None),
        "final_loss": hist[-1]["loss"],
        "tau_learned": hist[-1]["tau"],
        "recovery": rec,
        "b_leakage_r2": leak,
        "harmonised_rmse": harm_rmse,
        "naive_pooled_rmse": naive_rmse,
        "variance_reduction_pct": round(100 * (1 - harm_rmse / naive_rmse), 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    print("\n--- recovery (precondition check) ---")
    print(f"  g_proto corr {rec['g_corr']:+.3f}   rmse {rec['g_rmse']:.3f}")
    print(f"  b_s     corr {rec['b_corr']:+.3f}   rmse {rec['b_rmse']:.3f}")
    print(f"  b_s~p_s leakage R^2 {leak:.3f}  (want near 0)")
    print(f"  tau learned {res['tau_learned']:.3f}  (true b scale {spec.b_scale})")
    print("\n--- harmonisation vs naive pooling ---")
    print(f"  ProtoCal reference-protocol RMSE {harm_rmse:.3f}")
    print(f"  naive pooled RMSE                {naive_rmse:.3f}")
    print(f"  improvement                      {res['variance_reduction_pct']}%")
    print(f"\n  train {train_s:.1f}s  peak GPU {res['peak_gpu_mb']} MB")
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
