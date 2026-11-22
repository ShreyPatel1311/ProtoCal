"""Loader for the Domarecka dual-cure corpus (Zenodo 10.5281/zenodo.5569184).

An INDEPENDENT corpus for replicating the material-holdout result: a different
study, different materials, 9 commercial composites under a single protocol
(24 h water at 37 C -- which is also the Lodz reference condition).

Three traps in this record, all handled here rather than silently:
  * flexural modulus is reported in GPa (2.83-8.1) where the Lodz corpus uses
    MPa (~5190); converted to MPa on load;
  * the same material is spelled "Rebilda" in three sheets and "Rebilda DC" in
    two, and "Charisma classic" here vs "Charisma Classic" in Lodz;
  * Microhardness.xlsx has a different three-sheet layout and is excluded.
A loader keyed naively on column headers would invent materials and a
thousand-fold unit error.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = {
    "flexural_strength_MPa":  ("Flexural strength (FS).xlsx", 1.0),
    "flexural_modulus_MPa":   ("Flexural modulus (FM).xlsx", 1000.0),   # GPa -> MPa
    "diametral_tensile_MPa":  ("Diametral tensile strength (DTS).xlsx", 1.0),
    "vickers_hardness":       ("Hardness (HV).xlsx", 1.0),
    "shrinkage_stress_MPa":   ("Shrinkage stress.xlsx", 1.0),
}

CANON = {
    "rebilda": "Rebilda DC",
    "rebilda dc": "Rebilda DC",
    "charisma classic": "Charisma Classic",
    "multicore flow": "MultiCore Flow",
    "activa bioactive - restorative": "Activa Bioactive-Restorative",
    "filtek bulk fill posterior": "Filtek Bulk Fill Posterior",
    "g-aenial universal flo": "G-aenial Universal Flo",
    "bulk ez": "Bulk EZ",
    "fill-up!": "Fill-Up!",
    "starfill 2b": "StarFill 2B",
}

PROTOCOL = "water, 24 h, 37°C"   # identical to the Lodz reference protocol


def canon(name: str) -> str:
    return CANON.get(str(name).strip().lower(), str(name).strip())


def load(root: Path | None = None) -> pd.DataFrame:
    """Long-format table: material, property, specimen, value."""
    base = (root or Path(__file__).resolve().parent.parent) / "data_raw" / "dualcure_vs_lightcure"
    rows = []
    for prop, (fname, scale) in FILES.items():
        raw = pd.read_excel(base / fname, header=None)
        materials = [canon(c) for c in raw.iloc[0] if pd.notna(c)]
        body = raw.iloc[1:, :len(materials)]
        for j, mat in enumerate(materials):
            vals = pd.to_numeric(body.iloc[:, j], errors="coerce").dropna()
            for i, v in enumerate(vals):
                rows.append({"material": mat, "property": prop, "protocol": PROTOCOL,
                             "specimen": i + 1, "value": float(v) * scale,
                             "source_file": fname})
    return pd.DataFrame(rows)
