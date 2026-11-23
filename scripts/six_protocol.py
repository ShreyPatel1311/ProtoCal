#!/usr/bin/env python
"""Protocol geometry on SIX genuinely diverse protocols.

The four protocols in the 2025 corpus share a single NaOH leg, which is why
99.7% of their variation collapsed onto one axis. The 2022 deposit (Zenodo
6583563, CC-BY) tests the SAME three materials under six protocols that do not
share a common leg: artificial saliva, two thermocycling levels, hot water and
NaOH. This is the design needed to ask whether protocol space is genuinely
one-dimensional or whether that was an artefact of protocol selection.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROPS = {"FS": "flexural_strength_MPa", "DTS": "diametral_tensile_MPa",
         "HV": "vickers_hardness"}
# canonical protocol labels, keyed by a normalised fragment of the filename
PROTO = [
    ("24h", "P1 water 24h 37C (control)"),   # also matches "24 h" after space-stripping
    ("artificialsaliva", "P2 artificial saliva 90d 37C"),
    # 7500 MUST be tested before 500: "7500thermo" contains "500thermo",
    # so the looser pattern would swallow the 7500-cycle files.
    ("7500thermo", "P4 water 7500 thermocycles"),
    ("500thermo", "P3 water 500 thermocycles"),
    ("naoh", "P5 0.1M NaOH 7d 60C"),
    ("5days", "P6 water 5d 55C"),
]


def load() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "data_raw" / "lodz_basic_2022").glob("*.xlsx")):
        parts = [p.strip() for p in f.stem.split(",")]
        code = parts[0]
        if code not in PROPS:
            continue
        mat = re.sub(r"\s+", " ", parts[1]).strip()
        # normalise: the deposit spells the control "24 h, 37°C" for FS/HV
        # and "24h, 37C" for DTS, so collapse whitespace before matching
        rest = re.sub(r"\s+", "", ", ".join(parts[2:]).lower())
        label = next((lab for frag, lab in PROTO if frag in rest), None)
        if label is None:
            print("  unmatched protocol:", f.stem); continue
        vals = pd.to_numeric(pd.read_excel(f, header=None).iloc[1:, 0],
                             errors="coerce").dropna()
        for i, v in enumerate(vals):
            rows.append({"material": mat, "property": PROPS[code], "protocol": label,
                         "specimen": i + 1, "value": float(v)})
    return pd.DataFrame(rows)


def main():
    df = load()
    labels = sorted({lab for _, lab in PROTO})  # P1..P6 display order
    print(f"corpus: {len(df)} specimens, {df.material.nunique()} materials, "
          f"{df.protocol.nunique()} protocols, {df.property.nunique()} properties")
    print(f"  materials: {sorted(df.material.unique())}")
    print(f"  cells: {df.groupby(['material','protocol','property']).ngroups} of "
          f"{df.material.nunique()*len(labels)*df.property.nunique()}\n")

    # ---- protocol geometry -------------------------------------------------
    cell = np.log(df.groupby(["property", "material", "protocol"])["value"].mean()).unstack()
    cell = cell.reindex(columns=labels).dropna()
    M = cell.to_numpy()
    s = np.linalg.svd(M - M.mean(0, keepdims=True), compute_uv=False)
    share = (s ** 2 / (s ** 2).sum())
    print("PROTOCOL GEOMETRY (6 protocols, 3 materials x 3 properties)")
    print("  singular values:", np.round(s, 3).tolist())
    print("  variance share: ", np.round(share * 100, 1).tolist(), "%")
    print(f"  PC1 = {share[0]*100:.1f}%   (four-protocol corpus gave 99.7%)")
    n_eff = float(np.exp(-(share[share > 0] * np.log(share[share > 0])).sum()))
    print(f"  effective dimensionality (exp of entropy) = {n_eff:.2f}\n")

    # ---- pairwise distances -------------------------------------------------
    print("PAIRWISE PROTOCOL DISTANCES (mean |delta log| over material x property)")
    D = np.zeros((len(labels), len(labels)))
    for i in range(len(labels)):
        for j in range(len(labels)):
            D[i, j] = np.abs(M[:, i] - M[:, j]).mean()
    hdr = "".join(f"{l.split()[0]:>8}" for l in labels)
    print(f"{'':>30}{hdr}")
    for i, l in enumerate(labels):
        print(f"{l:>30}" + "".join(f"{D[i,j]:>8.3f}" for j in range(len(labels))))

    # ---- equivalence among the NON-control protocols ------------------------
    MARGIN = np.log1p(0.10)
    print(f"\nEQUIVALENCE among the five aged protocols (TOST, 10% margin)")
    res = []
    for pi in range(1, len(labels)):
        for pj in range(pi + 1, len(labels)):
            d = (cell[labels[pi]] - cell[labels[pj]]).dropna()
            n = len(d)
            if n < 4: continue
            m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
            p = max(stats.t.sf((m + MARGIN) / se, n - 1), stats.t.cdf((m - MARGIN) / se, n - 1))
            tag = "EQUIVALENT" if p < 0.05 else ""
            print(f"  {labels[pi][:28]:<30}vs {labels[pj][:26]:<28}"
                  f"{np.expm1(abs(m))*100:>6.1f}%  p={p:.3f}  {tag}")
            res.append({"a": labels[pi], "b": labels[pj], "pct": round(float(np.expm1(abs(m))*100),1),
                        "p_tost": round(float(p), 4), "equivalent": bool(p < 0.05)})
    eq = sum(r["equivalent"] for r in res)
    print(f"\n  equivalent at 10%: {eq}/{len(res)} pairs "
          f"(four-protocol corpus gave 6/15 across more properties)")

    out = {"n_specimens": int(len(df)), "protocols": labels,
           "pc1_share": round(float(share[0]), 4),
           "variance_share": [round(float(x), 4) for x in share],
           "effective_dim": round(n_eff, 3),
           "distances": D.round(4).tolist(), "equivalence": res}
    (ROOT / "output" / "six_protocol.json").write_text(json.dumps(out, indent=2))
    print("\nwrote output/six_protocol.json")


if __name__ == "__main__":
    main()
