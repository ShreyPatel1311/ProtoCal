"""Alrahlah (2013) Manchester thesis: 8 composites, three media, Vickers + K_Ic.

Tables 7.1/7.2 of "Physical, Mechanical and Surface Properties of Dental
Resin-composites", University of Manchester, extracted by hand from the
open-access full text.

Two features make this study disproportionately useful to the pooled corpus:

  * it is the only source measuring an ORGANIC solvent (methyl ethyl ketone)
    alongside water and 75% ethanol, so it extends the solvent-severity axis
    beyond anything in the Lodz, Kruger or Randolph deposits; and
  * its dry baseline is measured three times independently -- once per medium
    block -- which yields a within-laboratory repeatability estimate that is
    separately identified from the per-laboratory offset b_s.

Two Venus Diamond Flow cells (MEK, 1 d and 30 d) are LEFT-CENSORED: the
specimens softened below the indenter's usable range and the thesis reports
only an upper bound of 10 HV. They are returned with value NaN and
`censored_upper` set, never silently dropped: dropping them removes the single
most extreme confirming case for the filler-protection effect and so biases
that estimate toward zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data"

# Vickers indenter load used throughout this study (gf). Recorded because the
# indentation size effect makes HV load-dependent; it is NOT yet known for
# every laboratory in the pooled corpus, so it is carried but not used as a
# protocol covariate. See protocal.pooled.
LOAD_GF = 300.0


def load_hardness() -> pd.DataFrame:
    """Long-format Vickers hardness, one row per (material, medium, day)."""
    df = pd.read_csv(DATA / "alrahlah_2013_hardness.csv")
    out = pd.DataFrame({
        "study": "alrahlah_2013",
        "material": df["material"].str.strip(),
        "property": "vickers_hardness",
        "medium": df["medium"],
        "days": df["storage_days"].astype(float),
        "temp_C": df["storage_temp_c"].fillna(23.0).astype(float),
        "value": df["mean_hv"],
        "sd": df["sd_hv"],
        "n": df["specimens_per_medium_group"],
        "filler_wt_pct": df["filler_wt_pct"],
        "load_gf": df["measurement_load_gf"].astype(float),
        "censored_upper": np.where(df["hardness_status"].eq("not_measured"),
                                   df["reported_upper_bound_hv"], np.nan),
        # which medium block the dry baseline belongs to; lets the three
        # independent dry measurements be told apart for repeatability
        "block": df["assigned_medium"],
    })
    return out


def load_fracture_toughness() -> pd.DataFrame:
    """Long-format K_Ic (MPa.m^0.5), water storage at 1 and 7 days."""
    df = pd.read_csv(DATA / "alrahlah_2013_fracture_toughness.csv")
    return pd.DataFrame({
        "study": "alrahlah_2013",
        "material": df["material"].str.strip(),
        "property": "fracture_toughness",
        "medium": df["medium"],
        "days": df["storage_days"].astype(float),
        "temp_C": df["storage_temp_c"].astype(float),
        "value": df["mean_kic_mpa_sqrt_m"],
        "sd": df["sd_kic_mpa_sqrt_m"],
        "n": df["specimens_per_group"],
        "filler_wt_pct": df["filler_wt_pct"],
        "load_gf": np.nan,
        "censored_upper": np.nan,
        "block": df["medium"],
    })


def load() -> pd.DataFrame:
    return pd.concat([load_hardness(), load_fracture_toughness()], ignore_index=True)


def repeatability() -> dict:
    """Within-laboratory sd of log HV, from the triplicated dry baseline.

    This is a measurement-noise floor for a single operator on a single batch.
    It is the quantity the per-laboratory offset b_s must be compared against:
    if between-laboratory spread is no larger than this, b_s is not needed.
    """
    h = load_hardness()
    dry = h[(h["days"] == 0) & h["value"].notna()].copy()
    dry["log_hv"] = np.log(dry["value"])
    g = dry.groupby("material")["log_hv"].agg(["std", "count"])
    g = g[g["count"] > 1]
    return {
        "pooled_sd_log_hv": float(np.sqrt((g["std"] ** 2).mean())),
        "n_materials": int(len(g)),
        "replicates_per_material": int(g["count"].mode().iloc[0]),
    }
