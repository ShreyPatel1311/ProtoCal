#!/usr/bin/env python
"""Does the low-rank interaction earn its keep on real data?

The audit found a material x protocol interaction exceeding measurement noise
for flexural strength. The practically important question is not whether the
interaction exists but whether modelling it helps PREDICT a material/protocol
combination nobody tested -- which is exactly what harmonising scattered
literature requires.

Design: hold out entire (material, protocol) cells, fit on the rest, predict
the held-out specimens. An additive model can only use main effects. A rank-k
model can additionally borrow interaction structure from other cells. Repeated
over random splits; paired per split so the comparison is like-for-like.

Data: Lodz aging corpus, 1,802 specimen measurements (Zenodo 14576242, CC-BY).
x = material one-hot; p = interpretable protocol features; single laboratory,
so per-study offsets are switched off (n_studies=1).
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
from protocal.model import ProtoCal, ProtoCalConfig   # noqa: E402
from protocal.train import train                       # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROPS = ["flexural_strength_MPa", "flexural_modulus_MPa", "diametral_tensile_MPa",
         "vickers_hardness", "fracture_toughness"]


def build():
    df = pd.read_csv(ROOT / "output" / "aging_long.csv")
    df = df[df["property"].isin(PROPS)].copy()

    mats = sorted(df["material"].unique())
    protos = sorted(df["protocol"].unique())
    m_idx = {m: i for i, m in enumerate(mats)}
    p_idx = {p: i for i, p in enumerate(protos)}

    # Protocol features: interpretable and continuous where possible, so the
    # model can in principle generalise to unseen protocol settings.
    pf = (df.groupby("protocol")
            .agg(duration_days=("duration_days", "first"),
                 max_temp_C=("max_temp_C", "first"),
                 medium=("medium", "first"),
                 is_baseline=("is_baseline", "first"))
            .reindex(protos))
    P = np.zeros((len(protos), 5), dtype=np.float32)
    P[:, 0] = np.nan_to_num(pf["duration_days"].to_numpy(float)) / 7.0
    P[:, 1] = (np.nan_to_num(pf["max_temp_C"].to_numpy(float), nan=37.0) - 37.0) / 23.0
    P[:, 2] = (pf["medium"] == "NaOH").to_numpy(float)
    P[:, 3] = (pf["medium"] == "water").to_numpy(float)
    P[:, 4] = (pf["medium"] == "NaOH+water").to_numpy(float)
    ref_row = int(np.argmax(pf["is_baseline"].to_numpy(bool)))  # water 24h 37C

    # One row per specimen, properties as columns with a mask.
    df["cell"] = df["material"] + " || " + df["protocol"]
    df["row"] = df.groupby(["cell", "property"]).cumcount()
    wide = df.pivot_table(index=["material", "protocol", "cell", "row"],
                          columns="property", values="value").reset_index()
    for c in PROPS:
        if c not in wide:
            wide[c] = np.nan

    Y = wide[PROPS].to_numpy(np.float32)
    mask = (~np.isnan(Y)).astype(np.float32)
    # Standardise per property (scales differ by 3 orders of magnitude).
    mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0)
    Yz = np.nan_to_num((Y - mu) / sd)

    X = np.eye(len(mats), dtype=np.float32)[[m_idx[m] for m in wide["material"]]]
    Pr = P[[p_idx[p] for p in wide["protocol"]]]

    data = {
        "x": torch.tensor(X), "p": torch.tensor(Pr),
        "study": torch.zeros(len(wide), dtype=torch.long),
        "y": torch.tensor(Yz), "mask": torch.tensor(mask),
    }
    return data, wide["cell"].to_numpy(), torch.tensor(P[ref_row:ref_row + 1]), len(mats)


def run_split(data, cells, p_ref, n_mat, rank, held, seed):
    test = np.isin(cells, held)
    tr = {k: v[~test] for k, v in data.items()}
    te = {k: v[test] for k, v in data.items()}
    torch.manual_seed(seed)
    cfg = ProtoCalConfig(
        n_composition=n_mat, n_protocol=data["p"].shape[1], n_studies=1,
        n_properties=len(PROPS), shared_hidden=(64,), proto_hidden=(16,),
        interaction_rank=rank, dropout=0.0,
        init_log_tau=-4.0, learn_tau=False,   # single lab: offsets switched off
    )
    m = ProtoCal(cfg); m.set_reference_protocol(p_ref); m = m.to(DEV)
    train(m, tr, epochs=600, batch_size=256, device=DEV, verbose=False)
    m.eval()
    with torch.no_grad():
        pred = m(te["x"].to(DEV), te["p"].to(DEV), te["study"].to(DEV)).cpu()
    err2 = ((pred - te["y"]) ** 2 * te["mask"]).sum(0)
    n = te["mask"].sum(0).clamp(min=1)
    return (err2 / n).sqrt().numpy(), float((err2.sum() / te["mask"].sum()).sqrt())


def main() -> int:
    data, cells, p_ref, n_mat = build()
    uniq = np.unique(cells)
    print(f"specimens {len(cells)} | cells {len(uniq)} | materials {n_mat} "
          f"| protocol features {data['p'].shape[1]} | device {DEV}")

    ranks = [0, 1, 2, 3]
    n_splits, holdout_frac = 8, 0.25
    rng = np.random.default_rng(0)
    per_prop = {r: [] for r in ranks}
    overall = {r: [] for r in ranks}

    for s in range(n_splits):
        held = rng.choice(uniq, size=int(holdout_frac * len(uniq)), replace=False)
        for r in ranks:
            pp, ov = run_split(data, cells, p_ref, n_mat, r, held, seed=s)
            per_prop[r].append(pp); overall[r].append(ov)
        print(f"  split {s+1}/{n_splits}  " +
              "  ".join(f"rank{r}={overall[r][-1]:.3f}" for r in ranks))

    res = {"n_splits": n_splits, "holdout_frac": holdout_frac,
           "properties": PROPS, "per_property": {}, "overall": {}}
    print("\nheld-out RMSE (standardised units), mean +/- sd over splits")
    print(f"{'rank':<6}" + "".join(f"{p.split('_')[0][:9]:>11}" for p in PROPS) + f"{'OVERALL':>11}")
    base = np.array(overall[0])
    for r in ranks:
        a = np.array(per_prop[r]); o = np.array(overall[r])
        res["per_property"][r] = {"mean": a.mean(0).round(4).tolist(),
                                  "sd": a.std(0).round(4).tolist()}
        res["overall"][r] = {"mean": round(float(o.mean()), 4),
                             "sd": round(float(o.std()), 4)}
        print(f"{r:<6}" + "".join(f"{v:>11.3f}" for v in a.mean(0)) + f"{o.mean():>11.3f}")
    # paired comparison against additive
    print("\nvs additive (paired over splits):")
    for r in ranks[1:]:
        d = base - np.array(overall[r])
        wins = int((d > 0).sum())
        res["overall"][r]["mean_improvement_vs_additive"] = round(float(d.mean()), 4)
        res["overall"][r]["splits_better"] = wins
        print(f"  rank {r}: mean improvement {d.mean():+.4f}  "
              f"better in {wins}/{n_splits} splits")
    # per-property, rank chosen by overall
    best = min(ranks[1:], key=lambda r: np.mean(overall[r]))
    d_pp = np.array(per_prop[0]).mean(0) - np.array(per_prop[best]).mean(0)
    print(f"\nper-property gain, rank {best} vs additive:")
    for p, d in zip(PROPS, d_pp):
        print(f"  {p:<24} {d:+.4f}")
    res["best_rank"] = best
    res["per_property_gain_best_vs_additive"] = dict(zip(PROPS, d_pp.round(4).tolist()))

    (ROOT / "output" / "interaction_experiment.json").write_text(json.dumps(res, indent=2))
    print("\nwrote output/interaction_experiment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
