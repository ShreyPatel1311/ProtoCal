#!/usr/bin/env python
"""Which reference composites make calibration work, and how many are needed?

An average improvement hides the thing a practitioner needs to know. If one
shared composite cuts the error by half and another doubles it, "11.7% better on
average" is not a usable finding -- it is a summary of a bimodal outcome.

Design, held fixed across every comparison so the numbers are commensurable:

  * Hold out laboratory L. Train the model on the others.
  * ANCHOR-ELIGIBLE products are L's products that also appear in training.
    TEST products are L's products that do not. The test set never changes, so
    every anchor subset is scored on identical cells.
  * For every non-empty subset S of the anchor-eligible products, estimate a
    single additive offset from S's cells alone and score on the test cells.
  * Report the uncalibrated error, the full distribution over subsets, and the
    error by subset size -- which is the "how many are needed" curve.

Three reference points, all scored on the same test cells:
  uncalibrated      the model with the offset set to zero
  anchor_mean       predict the mean of L's OWN anchor-cell measurements; the
                    dumbest use of the same information the calibration gets
  oracle_offset     the offset that minimises test error, i.e. the best any
                    single additive correction could do. Not achievable, but it
                    bounds what anchor choice is competing for.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from protocal import pooled                                   # noqa: E402

MAX_EXACT = 12          # enumerate all subsets up to this many anchor products
MAX_SUBSETS = 4000      # otherwise sample this many


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def subsets(items, rng):
    k = len(items)
    if k <= MAX_EXACT:
        for r in range(1, k + 1):
            yield from itertools.combinations(items, r)
    else:
        seen = set()
        while len(seen) < MAX_SUBSETS:
            r = int(rng.integers(1, k + 1))
            s = tuple(sorted(rng.choice(k, size=r, replace=False).tolist()))
            if s in seen:
                continue
            seen.add(s)
            yield tuple(items[i] for i in s)


def run(df: pd.DataFrame, label: str) -> dict:
    rng = np.random.default_rng(0)
    y = df["log_hv"].to_numpy(float)
    feats = pooled.PROTO_FEATURES
    out = {"label": label, "n_cells": int(len(df)), "labs": {}}

    print(f"\n{'='*86}\n{label}: {len(df)} cells, {df['product'].nunique()} products, "
          f"{df['study'].nunique()} labs\n{'='*86}")
    print(f"{'lab':15s}{'anch':>5s}{'test':>5s}{'uncal':>8s}{'anchmean':>9s}"
          f"{'best':>7s}{'median':>8s}{'worst':>7s}{'oracle':>8s}{'%worse':>7s}")

    for held in sorted(df["study"].unique()):
        te_mask = (df["study"] == held).to_numpy()
        tr_mask = ~te_mask
        tr_products = set(df.loc[tr_mask, "product"])
        held_products = sorted(set(df.loc[te_mask, "product"]))
        anchor_products = [p for p in held_products if p in tr_products]
        test_products = [p for p in held_products if p not in tr_products]
        if not anchor_products or not test_products:
            print(f"{held:15s}{len(anchor_products):5d}{len(test_products):5d}"
                  f"   not scorable (needs both anchor and test products)")
            out["labs"][held] = {"scorable": False,
                                 "n_anchor_products": len(anchor_products),
                                 "n_test_products": len(test_products)}
            continue

        mu = df.loc[tr_mask, "filler_wt_pct"].mean()
        sd = df.loc[tr_mask, "filler_wt_pct"].std() or 1.0
        def design(m):
            x = ((df.loc[m, "filler_wt_pct"].to_numpy(float) - mu) / sd).reshape(-1, 1)
            return np.hstack([x, df.loc[m, feats].to_numpy(float)])

        model = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(design(tr_mask), y[tr_mask])
        pred_held = model.predict(design(te_mask))
        y_held = y[te_mask]
        prod_held = df.loc[te_mask, "product"].to_numpy()

        is_test = np.isin(prod_held, test_products)
        r_uncal = rmse(pred_held[is_test], y_held[is_test])
        # baseline: the mean of this laboratory's own anchor measurements
        anch_all = np.isin(prod_held, anchor_products)
        r_anchmean = rmse(np.full(is_test.sum(), y_held[anch_all].mean()),
                          y_held[is_test])
        # oracle single offset
        off_opt = float(np.mean(y_held[is_test] - pred_held[is_test]))
        r_oracle = rmse(pred_held[is_test] + off_opt, y_held[is_test])

        rows = []
        for S in subsets(anchor_products, rng):
            m = np.isin(prod_held, S)
            off = float(np.mean(y_held[m] - pred_held[m]))
            rows.append({"subset": list(S), "size": len(S),
                         "offset": round(off, 4),
                         "rmse": round(rmse(pred_held[is_test] + off,
                                            y_held[is_test]), 4)})
        r = np.array([x["rmse"] for x in rows])
        n_worse = int((r > r_uncal).sum())
        print(f"{held:15s}{len(anchor_products):5d}{len(test_products):5d}"
              f"{r_uncal:8.3f}{r_anchmean:9.3f}{r.min():7.3f}{np.median(r):8.3f}"
              f"{r.max():7.3f}{r_oracle:8.3f}{100*n_worse/len(r):6.0f}%")

        singles = [x for x in rows if x["size"] == 1]
        by_size = {int(k): {"mean": round(float(np.mean([x["rmse"] for x in v])), 4),
                            "sd": round(float(np.std([x["rmse"] for x in v])), 4),
                            "min": round(float(np.min([x["rmse"] for x in v])), 4),
                            "max": round(float(np.max([x["rmse"] for x in v])), 4),
                            "n_subsets": len(v)}
                   for k, v in itertools.groupby(sorted(rows, key=lambda x: x["size"]),
                                                 key=lambda x: x["size"])
                   for v in [list(v)]}
        out["labs"][held] = {
            "scorable": True,
            "anchor_products": anchor_products, "test_products": test_products,
            "n_test_cells": int(is_test.sum()),
            "rmse_uncalibrated": round(r_uncal, 4),
            "rmse_anchor_mean_baseline": round(r_anchmean, 4),
            "rmse_oracle_single_offset": round(r_oracle, 4),
            "subset_rmse": {"min": round(float(r.min()), 4),
                            "median": round(float(np.median(r)), 4),
                            "max": round(float(r.max()), 4),
                            "n_subsets": len(r),
                            "pct_worse_than_uncalibrated": round(100*n_worse/len(r), 1)},
            "by_size": by_size,
            "single_anchor": sorted(
                [{"product": x["subset"][0], "offset": x["offset"], "rmse": x["rmse"]}
                 for x in singles], key=lambda z: z["rmse"]),
        }
    return out


def main() -> int:
    res = {}
    for label, kw in [("all_five_labs", {}),
                      ("excl_kruger_true_vickers_only", {"exclude_kruger": True})]:
        res[label] = run(pooled.load(**kw), label)

    print(f"\n{'='*86}\nSINGLE-ANCHOR DETAIL (all five labs): which product you pick "
          f"decides the outcome\n{'='*86}")
    for lab, r in res["all_five_labs"]["labs"].items():
        if not r.get("scorable"):
            continue
        print(f"\n{lab}  (uncalibrated {r['rmse_uncalibrated']:.3f}, "
              f"oracle {r['rmse_oracle_single_offset']:.3f}, "
              f"test products: {', '.join(r['test_products'][:5])}"
              f"{' ...' if len(r['test_products'])>5 else ''})")
        for s in r["single_anchor"]:
            flag = "  WORSE" if s["rmse"] > r["rmse_uncalibrated"] else ""
            print(f"    {s['product']:26s} offset {s['offset']:+.3f}  "
                  f"rmse {s['rmse']:.3f}{flag}")
        print("    by subset size: " + "  ".join(
            f"n={k}: {v['mean']:.3f}+/-{v['sd']:.3f}" for k, v in sorted(r["by_size"].items())))

    (ROOT / "output" / "anchor_sweep.json").write_text(json.dumps(res, indent=2))
    print("\nwrote output/anchor_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
