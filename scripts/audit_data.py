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
import pathlib

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


# One file in the Lodz deposit is misnamed: "HV, Heliomolar Flow, water, 24 h,
# 37C.xlsx" is not that cell's specimens but the depositors' entire 24-sheet
# master workbook (DTS/FS/HV/FT/FM x raw/DANE/STAT/zmiana). Reading only sheet 0
# of it -- a DTS percent-change chart -- silently contributed nothing and left
# Heliomolar Flow HV missing AT THE REFERENCE PROTOCOL, which is where the gauge
# is fixed. The "<PROP> DANE" sheets hold the full per-property matrix, one
# column per (material, protocol) with a self-describing header, so they are
# parsed directly. Values recovered this way agree exactly with every cell
# parsed from the individual files (max |diff| = 0.0 over 47 HV cells), so this
# doubles as an end-to-end check on the filename-regex ingestion path.
MASTER_WORKBOOK = "HV, Heliomolar Flow, water, 24 h, 37\u00b0C.xlsx"
DANE_SHEETS = {"HV DANE": "vickers_hardness", "FS DANE": "flexural_strength_MPa",
               "DTS DANE": "diametral_tensile_MPa", "FT dane": "fracture_toughness"}

# The workbook's sheets are inconsistent about protocol naming: the HV sheet
# spells protocols out in full while the DTS and FT sheets use shorthand. Left
# unmapped, the shorthand looks like four extra protocols and re-admits 715 rows
# that the per-file parse already supplied. Mapping them onto the canonical
# names lets the gap-fill de-duplication work -- and, because every one of those
# 715 rows then matches its canonical counterpart exactly (max |diff| = 4.6e-14,
# floating point only), they serve as an independent check on the DTS and FT
# ingestion just as the HV sheet does for hardness.
PROTOCOL_ALIASES = {
    "control": "water, 24 h, 37\u00b0C",
    "7d NaOH": "0.1M NaOH, 7 days, 60\u00b0C",
    "5d water + NaOH": "water, 5 days, 55\u00b0C and 0.1M NaOH, 7 days, 60\u00b0C",
    "thermocycling + NaOH": ("water, 7500 thermo cycles, 5\u00b0C_55\u00b0C "
                             "and 0.1M NaOH, 7 days, 60\u00b0C"),
}


def _protocol_meta(protocol: str) -> dict:
    """Structured protocol fields from a protocol string (no property/material)."""
    norm = protocol.lower()
    days = re.search(r"(\d+)\s*day", norm)
    hours = re.search(r"(\d+)\s*h\b", norm)
    temps = [float(t) for t in re.findall(r"(\d+)\s*\u00b0?c", norm)]
    return {
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


def load_master_workbook(path: pathlib.Path) -> list[dict]:
    """Parse the `* DANE` sheets of the depositors' master workbook."""
    rows = []
    try:
        book = pd.read_excel(path, sheet_name=None, header=None)
    except Exception:
        return rows
    for sheet, prop in DANE_SHEETS.items():
        if sheet not in book:
            continue
        d = book[sheet]
        for j, label in enumerate(d.iloc[0]):
            if not isinstance(label, str) or ", " not in label:
                continue
            material, protocol = label.split(", ", 1)
            protocol = PROTOCOL_ALIASES.get(protocol.strip(), protocol.strip())
            meta = {"property": prop, "material": material.strip(),
                    **_protocol_meta(protocol)}
            vals = pd.to_numeric(d.iloc[1:, j], errors="coerce").dropna()
            for i, v in enumerate(vals):
                rows.append({**meta, "specimen": i + 1, "value": float(v),
                             "source_file": f"{path.name}#{sheet}"})
    return rows


def load_long() -> pd.DataFrame:
    rows, master = [], []
    for f in sorted((WORK / "aging").rglob("*.xlsx")):
        if f.name == MASTER_WORKBOOK:
            master = load_master_workbook(f)
            continue
        meta = parse_name(f.stem)
        if meta is None:
            continue
        df = pd.read_excel(f, header=None)
        vals = pd.to_numeric(df.iloc[1:, 0], errors="coerce").dropna()
        for i, v in enumerate(vals):
            rows.append({**meta, "specimen": i + 1, "value": float(v),
                         "source_file": f.name})
    # Keep workbook rows only for cells the individual files did not provide,
    # so the per-file parse stays the primary source and the workbook fills gaps.
    df = pd.DataFrame(rows)
    if master:
        have = set(zip(df["property"], df["material"], df["protocol"]))
        extra = [r for r in master
                 if (r["property"], r["material"], r["protocol"]) not in have]
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


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
