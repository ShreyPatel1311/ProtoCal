"""Composition descriptors for the 12 composites, from Table 2 of the source study.

Szczesio-Wlodarczyk et al., J Funct Biomater 2025;16(2):49, doi:10.3390/jfb16020049.
Values are the manufacturers' own, as reported in that table ("data given by
manufacturers").

This replaces material IDENTITY (a 12-way one-hot) with material CHEMISTRY. The
distinction is not cosmetic: an MLP over a one-hot can emit only as many
distinct outputs as there are materials, so it is a lookup table and can say
nothing whatever about a composite it has not seen. Continuous descriptors make
leave-one-material-out a well-posed question for the first time.

Missing cells are real gaps in the manufacturers' disclosure, not data entry
errors; the paper notes manufacturers "do not provide detailed compositional
information, with filler content as the only quantitative data available".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#  material                                wt%    vol%   flow  bulk  matrix
_RAW = [
    ("Charisma Classic",                   77.0,  61.0,  0, 0, "dimethacrylate"),
    ("G-ænial Anterior",                   73.0,  64.0,  0, 0, "dimethacrylate"),
    ("Filtek Bulk Fill Flowable Restorative", 64.5, 42.5, 1, 1, "dimethacrylate"),
    ("Charisma Bulk Flow One",             65.0,  41.0,  1, 1, "dimethacrylate"),
    ("Heliomolar Flow",                    59.0,  30.0,  1, 0, "dimethacrylate"),
    ("Revolution Formula 2",               60.0,  np.nan, 1, 0, "dimethacrylate"),
    ("Grandio SO Heavy Flow",              83.0,  np.nan, 1, 0, "dimethacrylate"),
    ("Filtek Ultimate",                    78.5,  63.3,  0, 0, "dimethacrylate"),
    ("Admira Fusion",                      84.0,  np.nan, 0, 0, "ormocer"),
    ("Charisma Topaz",                     np.nan, 59.0, 0, 0, "tcd_di_hea"),
    ("Kalore",                             82.0,  69.0,  0, 0, "dx511"),
    ("NeoSpectra ST",                      79.0,  61.0,  0, 0, "spheretec"),  # 78-80 / 60-62
]

FEATURES = ["filler_wt_pct", "filler_vol_pct", "is_flowable", "is_bulkfill",
            "nonconventional_matrix"]


def composition_table() -> pd.DataFrame:
    """12 x 5 descriptor table, with wt%/vol% gaps imputed by linear regression.

    wt% and vol% are related through filler and matrix density, so the eight
    materials disclosing both support a regression that fills the four gaps.
    The imputation is reported rather than hidden: `imputed_*` flags mark it.
    """
    df = pd.DataFrame(_RAW, columns=["material", "filler_wt_pct", "filler_vol_pct",
                                     "is_flowable", "is_bulkfill", "matrix"])
    df["imputed_wt"] = df["filler_wt_pct"].isna()
    df["imputed_vol"] = df["filler_vol_pct"].isna()

    both = df.dropna(subset=["filler_wt_pct", "filler_vol_pct"])
    w, v = both["filler_wt_pct"].to_numpy(), both["filler_vol_pct"].to_numpy()
    # vol ~ a*wt + b   and the inverse
    a, b = np.polyfit(w, v, 1)
    c, d = np.polyfit(v, w, 1)
    df.loc[df["imputed_vol"], "filler_vol_pct"] = a * df.loc[df["imputed_vol"], "filler_wt_pct"] + b
    df.loc[df["imputed_wt"], "filler_wt_pct"] = c * df.loc[df["imputed_wt"], "filler_vol_pct"] + d

    df["nonconventional_matrix"] = (df["matrix"] != "dimethacrylate").astype(float)
    df["is_flowable"] = df["is_flowable"].astype(float)
    df["is_bulkfill"] = df["is_bulkfill"].astype(float)
    df.attrs["vol_from_wt"] = (round(float(a), 4), round(float(b), 4))
    df.attrs["n_complete_pairs"] = int(len(both))
    return df


def feature_matrix(materials: list[str]) -> np.ndarray:
    """Standardised descriptor matrix in the order of `materials`."""
    t = composition_table().set_index("material")
    X = t.loc[materials, FEATURES].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)
