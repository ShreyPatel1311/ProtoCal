#!/usr/bin/env python
"""Audit the downloaded open dental-biomaterials datasets.

Main target is the Lodz aging-protocol corpus (Zenodo 14576242), which is a
near-factorial material x aging-protocol design measured at specimen level.
Protocol is encoded in each filename, e.g.

    "FS, Admira Fusion, 0.1M NaOH, 7 days, 60C.xlsx"
     ^property ^material  ^-------- protocol --------^

That makes it the only public dataset found that carries protocol metadata
alongside specimen-level property values -- and therefore a direct empirical
test of ProtoCal's central modelling assumption: that a protocol effect can be
treated as ADDITIVE and shared across materials.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, WORK, OUT = ROOT / "data_raw", ROOT / "data_work", ROOT / "output"

PROP = {"FS": "flexural_strength_MPa", "FM": "flexural_modulus_MPa",
        "DTS": "diametral_tensile_MPa", "HV": "vickers_hardness",
        "FT": "fracture_toughness"}


def extract() -> None:
    dest = WORK / "aging"
    dest.mkdir(parents=True, exist_ok=True)
    for z in (RAW / "aging_protocols_lodz").glob("*.zip"):
        if z.stat().st_size < 5e6:          # skip the SEM image archives
            zipfile.ZipFile(z).extractall(dest)


def parse_name(stem: str) -> dict | None:
    parts = [p.strip() for p in stem.split(",")]
    if len(parts) < 3 or parts[0] not in PROP:
        return None
    protocol = ", ".join(parts[2:])
    norm = protocol.lower()
    days = re.search(r"(\d+)\s*day", norm)
    hours = re.search(r"(\d+)\s*h\b", norm)
    temps = [float(t) for t in re.findall(r"(\d+)\s*°?c", norm)]
    return {
        "property": PROP[parts[0]],
        "material": parts[1],
        "protocol": protocol,
        "medium": ("NaOH+water" if ("naoh" in norm and "water" in norm)
                   else "NaOH" if "naoh" in norm
                   else "water" if "water" in norm else "other"),
        "duration_days": (float(days.group(1)) if days
                          else float(hours.group(1)) / 24 if hours else np.nan),
        "max_temp_C": max(temps) if temps else np.nan,
        "is_baseline": ("water" in norm and "24" in norm and "37" in norm
                        and "naoh" not in norm),
    }


def load_long() -> pd.DataFrame:
    rows = []
    for f in sorted((WORK / "aging").rglob("*.xlsx")):
        meta = parse_name(f.stem)
        if meta is None:
            continue
        df = pd.read_excel(f, header=None)
        vals = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
        for i, v in enumerate(vals):
            rows.append({**meta, "specimen": i + 1, "value": float(v),
                         "source_file": f.name})
    return pd.DataFrame(rows)


def anomalies(df: pd.DataFrame) -> list[dict]:
    found = []
    for (prop, mat, proto), g in df.groupby(["property", "material", "protocol"]):
        v = g["value"].to_numpy()
        n = len(v)
        if n < 3:
            found.append({"type": "tiny_group", "property": prop, "material": mat,
                          "protocol": proto, "n": n,
                          "detail": f"only {n} specimens"})
        if n >= 2 and np.std(v) == 0:
            found.append({"type": "zero_variance", "property": prop, "material": mat,
                          "protocol": proto, "n": n,
                          "detail": f"all specimens identical ({v[0]})"})
        if (v <= 0).any():
            found.append({"type": "non_positive", "property": prop, "material": mat,
                          "protocol": proto, "n": n,
                          "detail": f"{int((v<=0).sum())} non-positive values"})
        if n >= 4:
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            if mad > 0:
                z = 0.6745 * (v - med) / mad
                out = np.abs(z) > 3.5
                if out.any():
                    found.append({"type": "outlier", "property": prop,
                                  "material": mat, "protocol": proto, "n": n,
                                  "detail": f"{int(out.sum())} robust-z>3.5: "
                                            f"{np.round(v[out],1).tolist()} vs median {med:.1f}"})
        cv = np.std(v, ddof=1) / np.mean(v) if n > 1 and np.mean(v) else np.nan
        if np.isfinite(cv) and cv > 0.30:
            found.append({"type": "high_dispersion", "property": prop, "material": mat,
                          "protocol": proto, "n": n, "detail": f"CV={cv:.0%}"})
    return found


def additivity_test(df: pd.DataFrame) -> list[dict]:
    """Does aging shift every material by the same amount (ProtoCal's additive
    g_proto), or is the protocol effect material-specific (an interaction the
    additive model cannot represent)?

    Two-way ANOVA per property: material x protocol, on log values (effects in
    materials science are typically multiplicative, so logs make 'additive'
    the right null). Reports the share of variance taken by the interaction.
    """
    res = []
    for prop, g in df.groupby("property"):
        piv = g.groupby(["material", "protocol"])["value"].agg(["mean", "count"])
        wide = piv["mean"].unstack()
        wide = wide.dropna(axis=0, how="any").dropna(axis=1, how="any")
        if wide.shape[0] < 3 or wide.shape[1] < 3:
            res.append({"property": prop, "status": "insufficient complete cells",
                        "materials": int(wide.shape[0]), "protocols": int(wide.shape[1])})
            continue
        L = np.log(wide.to_numpy())
        grand = L.mean()
        row = L.mean(1, keepdims=True) - grand      # material main effect
        col = L.mean(0, keepdims=True) - grand      # protocol main effect
        inter = L - (grand + row + col)             # residual = interaction
        ss_m = float((row**2).sum() * L.shape[1])
        ss_p = float((col**2).sum() * L.shape[0])
        ss_i = float((inter**2).sum())
        tot = ss_m + ss_p + ss_i
        # within-group noise, for reference
        within = g.groupby(["material", "protocol"])["value"].apply(
            lambda s: np.var(np.log(s), ddof=1) if len(s) > 1 else np.nan).mean()
        res.append({
            "property": prop,
            "materials": int(wide.shape[0]), "protocols": int(wide.shape[1]),
            "pct_var_material": round(100 * ss_m / tot, 1),
            "pct_var_protocol": round(100 * ss_p / tot, 1),
            "pct_var_interaction": round(100 * ss_i / tot, 1),
            "mean_within_group_log_var": round(float(within), 5),
            "interaction_vs_noise": round(
                float(ss_i / max(inter.size, 1) / within) if within and np.isfinite(within) else float("nan"), 2),
        })
    return res


def main() -> int:
    OUT.mkdir(exist_ok=True)
    extract()
    df = load_long()
    df.to_csv(OUT / "aging_long.csv", index=False)

    summary = {
        "n_measurements": int(len(df)),
        "n_materials": int(df["material"].nunique()),
        "materials": sorted(df["material"].unique().tolist()),
        "n_protocols": int(df["protocol"].nunique()),
        "protocols": sorted(df["protocol"].unique().tolist()),
        "properties": sorted(df["property"].unique().tolist()),
        "specimens_per_group": df.groupby(
            ["property", "material", "protocol"]).size().describe().round(2).to_dict(),
        "cells_present": int(df.groupby(["property", "material", "protocol"]).ngroups),
        "cells_if_full_factorial": int(df["property"].nunique() *
                                       df["material"].nunique() *
                                       df["protocol"].nunique()),
    }
    anom = anomalies(df)
    add = additivity_test(df)

    report = {"summary": summary, "anomalies": anom, "additivity": add}
    (OUT / "data_audit.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"measurements: {summary['n_measurements']}")
    print(f"materials:    {summary['n_materials']}")
    print(f"protocols:    {summary['n_protocols']}")
    print(f"properties:   {len(summary['properties'])}")
    print(f"cells:        {summary['cells_present']} present / "
          f"{summary['cells_if_full_factorial']} if full factorial")
    print(f"\nanomalies flagged: {len(anom)}")
    from collections import Counter
    for k, c in Counter(a["type"] for a in anom).most_common():
        print(f"   {k:<18} {c}")
    print("\nadditivity (variance decomposition, log scale):")
    for r in add:
        if "status" in r:
            print(f"   {r['property']}: {r['status']}")
        else:
            print(f"   {r['property']:<24} material {r['pct_var_material']:>5}%  "
                  f"protocol {r['pct_var_protocol']:>5}%  "
                  f"interaction {r['pct_var_interaction']:>5}%  "
                  f"(inter/noise {r['interaction_vs_noise']})")
    print(f"\nwrote {OUT/'data_audit.json'} and {OUT/'aging_long.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
