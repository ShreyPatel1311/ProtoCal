#!/usr/bin/env python
"""Frozen external validation of shared-composite calibration.

Implements output/PREREG_EXTERNAL_VALIDATION.md exactly. The protocol was
committed (557fde4) before the validation study's tables were read.

Validation study
  Alshabib A, Algamaiah H, Rojas-Rueda S, Jurado CA, Almansour A, AlOtaibi S.
  "Influence of Solvent Challenge on the Hardness and Toughness of
  Viscosity-Modified Composites." Bioengineering 2026;13(7):726.
  doi:10.3390/bioengineering13070726

  Independent of every training laboratory: King Saud University (Riyadh),
  Ponce Health Sciences University, Prince Sattam Bin Abdulaziz University. No
  author or affiliation overlap with Lodz, Rostock, Birmingham/Louvain or
  Manchester. Its Vickers method -- 300 gf, 15 s -- is identical to Alrahlah's,
  which is in the training set.

Two ambiguities the pre-registration did not anticipate are handled by running
BOTH ways and reporting both, rather than by choosing after seeing the result:

  A. Charisma Bulk Flow ONE has two different manufacturer filler figures:
     70 wt% in the validation study, 65 wt% in the Lodz descriptor table. Both
     are manufacturer-reported for the same product.
  B. Filtek Z350 XT is treated as the same product as Filtek Supreme XTE /
     Filtek Ultimate (3M regional naming). Supporting evidence: the validation
     study reports 78 wt% / ~63 vol%, and Randolph's Table 1 gives Filtek
     Supreme XTE as 78.5 / 63.3. Treating them as DIFFERENT products is run as
     a sensitivity, because it reduces the reference set from two products to
     one and that materially changes the test.

Also reported: Table 2 gives Filtek Z350 XT Flowable in water at 1 d as
38.9 (1.2), while Section 3.1 of the same paper gives 38.3 +/- 1.2. The
pre-registration names Tables 1 and 2 as the data source, so 38.9 is primary
and 38.3 is a sensitivity. That product is in the TEST set, so it matters.
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
from protocal import pooled, protocols                       # noqa: E402

VALIDATION = ROOT / "data_external" / "alshabib2026_vickers.csv"

# frozen: model, training set, filler definition
ALPHAS = np.logspace(-3, 3, 25)
TRAIN_KW = dict(filler_variant="manufacturer", exclude_kruger=True)
# frozen: products never used as references
REFERENCE_EXCLUDED = ("Tetric EvoCeram", "Kalore")
# 3M regional naming; see module docstring
Z350_ALIAS = {"Filtek Z350 XT": "Filtek Ultimate"}
N_BOOT = 10_000
SEED = 0


def load_validation(xtef_water_1d: float | None = None) -> pd.DataFrame:
    v = pd.read_csv(VALIDATION)
    if xtef_water_1d is not None:
        m = ((v["material"] == "Filtek Z350 XT Flowable") &
             (v["medium"] == "water") & (v["days"] == 1))
        v.loc[m, "mean_hv"] = xtef_water_1d
    v["log_hv"] = np.log(v["mean_hv"])
    v["protocol_text"] = (v["medium"] + ", " + v["days"].astype(int).astype(str)
                          + " days, 37C")
    feats = pd.DataFrame([protocols.encode(r.protocol_text, days_override=r.days)
                          for r in v.itertuples()], index=v.index)
    return pd.concat([v, feats], axis=1)


def select(v: pd.DataFrame, train_products: set[str], train_filler: pd.Series,
           *, z350_same_product: bool) -> tuple[list[str], list[str], dict]:
    """The frozen reference/test rule, applied mechanically."""
    alias = Z350_ALIAS if z350_same_product else {}
    lo, hi = float(train_filler.min()), float(train_filler.max())
    refs, tests, why = [], [], {}
    for m in sorted(v["material"].unique()):
        p = alias.get(m, m)
        f = float(v.loc[v["material"] == m, "filler_wt_pct_reported"].iloc[0])
        if p not in train_products:
            tests.append(m); why[m] = "test: not in training corpus"
        elif p in REFERENCE_EXCLUDED:
            tests.append(m); why[m] = "test: excluded from reference use (contested filler)"
        elif f <= lo or f >= hi:
            tests.append(m); why[m] = f"test: filler {f} at edge of training range [{lo},{hi}]"
        else:
            refs.append(m); why[m] = f"REFERENCE (as '{p}', filler {f})"
    return refs, tests, why


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def evaluate(v: pd.DataFrame, *, z350_same_product: bool,
             cbf_filler: str) -> dict:
    tr = pooled.aqueous_family(pooled.load(**TRAIN_KW))
    F = pooled.PROTO_FEATURES
    ytr = tr["log_hv"].to_numpy(float)

    vv = v.copy()
    vv["filler_used"] = vv["filler_wt_pct_reported"].astype(float)
    if cbf_filler == "corpus":
        corpus_f = tr.loc[tr["product"] == "Charisma Bulk Flow One",
                          "filler_wt_pct"]
        if len(corpus_f):
            vv.loc[vv["material"] == "Charisma Bulk Flow One", "filler_used"] = \
                float(corpus_f.iloc[0])

    refs, tests, why = select(vv, set(tr["product"]), tr["filler_wt_pct"],
                              z350_same_product=z350_same_product)
    if not refs or not tests:
        return {"estimable": False, "references": refs, "test_products": tests,
                "selection": why}

    mu = tr["filler_wt_pct"].mean(); sd = tr["filler_wt_pct"].std() or 1.0
    Xtr = np.hstack([((tr["filler_wt_pct"].to_numpy(float) - mu) / sd).reshape(-1, 1),
                     tr[F].to_numpy(float)])
    model = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)

    Xv = np.hstack([((vv["filler_used"].to_numpy(float) - mu) / sd).reshape(-1, 1),
                    vv[F].to_numpy(float)])
    pred = model.predict(Xv)
    y = vv["log_hv"].to_numpy(float)
    prod = vv["material"].to_numpy()
    is_ref = np.isin(prod, refs); is_test = np.isin(prod, tests)

    offset = float(np.mean(y[is_ref] - pred[is_ref]))
    r_unc = rmse(pred[is_test], y[is_test])
    r_cal = rmse(pred[is_test] + offset, y[is_test])
    r_ref = rmse(np.full(int(is_test.sum()), y[is_ref].mean()), y[is_test])

    rng = np.random.default_rng(SEED)
    tp = np.array(tests)
    d_cu, d_cr, d_ur = [], [], []
    for _ in range(N_BOOT):
        pick = rng.choice(tp, size=len(tp), replace=True)
        idx = np.concatenate([np.where(prod == p)[0] for p in pick])
        c = rmse(pred[idx] + offset, y[idx]); u = rmse(pred[idx], y[idx])
        r = rmse(np.full(len(idx), y[is_ref].mean()), y[idx])
        d_cu.append(c - u); d_cr.append(c - r); d_ur.append(u - r)

    def ci(a):
        a = np.asarray(a)
        lo, hi = np.percentile(a, [2.5, 97.5])
        return {"mean": round(float(a.mean()), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_zero": bool(lo > 0 or hi < 0)}

    subsets = {}
    for k in range(1, len(refs) + 1):
        for S in itertools.combinations(refs, k):
            m = np.isin(prod, S)
            off = float(np.mean(y[m] - pred[m]))
            subsets[" + ".join(S)] = round(rmse(pred[is_test] + off, y[is_test]), 4)

    # With few test products the cluster bootstrap is degenerate: resampling k
    # clusters with replacement admits only C(2k-1,k) distinct multisets, so the
    # percentile "interval" is the range of a handful of values and must NOT be
    # read as a 95% interval. Record the count so the report cannot hide it.
    n_distinct = len(set(np.round(d_cu, 6)))
    per_product = {}
    rmean = float(y[is_ref].mean())
    for p_ in tests:
        i = np.where(prod == p_)[0]
        per_product[p_] = {
            "n_cells": int(len(i)),
            "uncalibrated": round(rmse(pred[i], y[i]), 4),
            "calibrated": round(rmse(pred[i] + offset, y[i]), 4),
            "reference_mean": round(rmse(np.full(len(i), rmean), y[i]), 4),
            "cal_minus_uncal": round(rmse(pred[i] + offset, y[i])
                                     - rmse(pred[i], y[i]), 4)}

    verdict = ("calibration HARMS" if ci(d_cu)["excludes_zero"] and np.mean(d_cu) > 0
               else "calibration HELPS" if ci(d_cu)["excludes_zero"]
               else "INCONCLUSIVE at this sample size")
    return {
        "estimable": True, "references": refs, "test_products": tests,
        "selection": why, "n_test_cells": int(is_test.sum()),
        "n_ref_cells": int(is_ref.sum()),
        "estimated_offset_log": round(offset, 4),
        "rmse": {"uncalibrated": round(r_unc, 4), "calibrated": round(r_cal, 4),
                 "reference_mean": round(r_ref, 4)},
        "primary_calibrated_minus_uncalibrated": ci(d_cu),
        "secondary_calibrated_minus_reference_mean": ci(d_cr),
        "secondary_uncalibrated_minus_reference_mean": ci(d_ur),
        "reference_subsets": subsets,
        "per_test_product": per_product,
        "bootstrap_distinct_values": n_distinct,
        "bootstrap_interval_trustworthy": bool(len(tests) >= 5),
        "mean_residual_all_validation_cells": round(float(np.mean(y - pred)), 4),
        "mean_residual_reference_cells": round(offset, 4),
        "mean_residual_test_cells": round(float(np.mean(y[is_test] - pred[is_test])), 4),
        "verdict": verdict,
    }


def main() -> int:
    out = {"study": "Alshabib et al., Bioengineering 2026;13(7):726",
           "doi": "10.3390/bioengineering13070726",
           "independence": "no author/affiliation overlap with any training laboratory",
           "validation_method": "Vickers 300 gf / 15 s, 9 indents, n=6 "
                                "(identical load and dwell to Alrahlah)",
           "prereg_commit": "557fde4", "results": {}}

    print("=" * 94)
    print("FROZEN EXTERNAL VALIDATION — Alshabib et al. 2026, Bioengineering 13(7):726")
    print("prereg committed 557fde4, before these tables were read")
    print("=" * 94)

    v_primary = load_validation()
    arms = [("PRIMARY  (study filler, Z350XT=Ultimate, Table 2)", v_primary, True, "study"),
            ("sens A   (corpus filler for CBF: 65 wt%)",          v_primary, True, "corpus"),
            ("sens B   (Z350XT treated as a DIFFERENT product)",  v_primary, False, "study"),
            ("sens C   (XTEF water 1 d = 38.3 from Sec 3.1)",
             load_validation(xtef_water_1d=38.3), True, "study")]

    for label, vdf, same, cbf in arms:
        r = evaluate(vdf, z350_same_product=same, cbf_filler=cbf)
        out["results"][label.split("(")[0].strip()] = r
        print(f"\n--- {label} ---")
        if not r["estimable"]:
            print("  NOT ESTIMABLE:", r["references"], r["test_products"]); continue
        for m, w in r["selection"].items():
            print(f"    {m:28s} {w}")
        print(f"  references={r['references']}  ({r['n_ref_cells']} cells)   "
              f"test={r['test_products']}  ({r['n_test_cells']} cells)")
        print(f"  offset estimated from references: {r['estimated_offset_log']:+.4f}")
        q = r["rmse"]
        print(f"  RMSE  uncalibrated {q['uncalibrated']:.4f}   "
              f"calibrated {q['calibrated']:.4f}   reference_mean {q['reference_mean']:.4f}")
        p = r["primary_calibrated_minus_uncalibrated"]
        print(f"  PRIMARY  cal - uncal = {p['mean']:+.4f}  "
              f"95% CI [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]  -> {r['verdict']}")
        s1 = r["secondary_calibrated_minus_reference_mean"]
        s2 = r["secondary_uncalibrated_minus_reference_mean"]
        print(f"  secondary  cal - refmean = {s1['mean']:+.4f} "
              f"[{s1['ci95'][0]:+.4f},{s1['ci95'][1]:+.4f}]   "
              f"uncal - refmean = {s2['mean']:+.4f} "
              f"[{s2['ci95'][0]:+.4f},{s2['ci95'][1]:+.4f}]")
        print(f"  by reference subset: {r['reference_subsets']}")
        print(f"  per test product:")
        for p_, d_ in r["per_test_product"].items():
            print(f"     {p_:26s} uncal {d_['uncalibrated']:.4f}  cal {d_['calibrated']:.4f}  "
                  f"refmean {d_['reference_mean']:.4f}  delta {d_['cal_minus_uncal']:+.4f}")
        if not r["bootstrap_interval_trustworthy"]:
            print(f"  !! only {len(r['test_products'])} test products: the bootstrap takes "
                  f"{r['bootstrap_distinct_values']} distinct values; the interval above is "
                  f"NOT a usable 95% CI")
        print(f"  model residual (observed - predicted): all cells "
              f"{r['mean_residual_all_validation_cells']:+.4f}, references "
              f"{r['mean_residual_reference_cells']:+.4f}, test "
              f"{r['mean_residual_test_cells']:+.4f}")

    (ROOT / "output" / "external_validation.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/external_validation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
