"""Pooled five-laboratory Vickers-hardness corpus.

Vickers hardness is the only property measured by all five laboratories, so it
is the axis along which cross-laboratory transfer can actually be tested.

The unit of analysis is the (laboratory, material, protocol) CELL MEAN, not the
individual specimen. Two laboratories publish specimen-level data and three
publish only group means, so specimen-level pooling would silently weight Lodz
and Kruger by one to two orders of magnitude more rows and turn a five-
laboratory study into a two-laboratory one. Cell means with the group size `n`
retained keep every laboratory on the same footing; the specimen-level variant
is kept for robustness only.

Materials are addressed by PRODUCT, canonicalised across laboratories, because
the whole point is that a product measured in two places gives an offset
estimate. Protocols are described by continuous, physically interpretable
covariates rather than by identity, so a protocol run in only one laboratory
still informs the others.

Indenter load is recorded but deliberately NOT used as a covariate: it is known
for Alrahlah (300 gf) and Oz (500 gf) and unknown for the other three, so
including it would make it a proxy for laboratory identity rather than a
measured protocol difference. Vickers hardness is load-dependent through the
indentation size effect, so some of what the per-laboratory offset b_s absorbs
here is a real, observable protocol difference we cannot yet observe. This is a
limitation of the corpus, not of the model, and it bounds how much of b_s may
be read as "laboratory idiosyncrasy".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import alrahlah, composition, kruger, oz, randolph
from .stats import canonical_materials

# Same physical product sold under different names in different markets.
PRODUCT_ALIASES = {
    "Filtek Supreme XTE": "Filtek Ultimate",
    "Tetric Evo Ceram": "Tetric EvoCeram",
}

# Protocol covariates, in a vocabulary wide enough to describe all five
# laboratories' challenges. Reference = brief water storage at 37 C, which every
# laboratory runs as its control, so the gauge is anchored in shared practice
# rather than in one deposit's conventions.
PROTO_FEATURES = [
    "log_days",           # log1p(days) / log1p(180)
    "temp_excess",        # (max temp - 37) / 23
    "ethanol_frac",       # volume fraction of ethanol, 0 if none
    "organic_solvent",    # MEK and similar non-aqueous solvents
    "ph_deficit",         # max(0, 7 - pH) / 4, for citric/lactic acid
    "alkaline",           # 0.1 M NaOH
    "surface_treatment",  # prophylaxis paste / in-office bleaching
]

_REFERENCE = {f: 0.0 for f in PROTO_FEATURES}


def _proto_vector(medium: str, days: float, temp_C: float) -> dict:
    """Map a laboratory's protocol description onto the shared covariates."""
    m = str(medium).lower()
    v = dict(_REFERENCE)
    v["log_days"] = float(np.log1p(max(days, 0.0)) / np.log1p(180.0))
    v["temp_excess"] = float((np.nan_to_num(temp_C, nan=37.0) - 37.0) / 23.0)
    if "ethanol" in m:
        if "75" in m:
            v["ethanol_frac"] = 0.75
        elif "60" in m:
            v["ethanol_frac"] = 0.60
        elif "40" in m:
            v["ethanol_frac"] = 0.40
        else:
            v["ethanol_frac"] = 0.75
    if "ketone" in m or "mek" in m:
        v["organic_solvent"] = 1.0
    if "ph 3" in m:
        v["ph_deficit"] = (7.0 - 3.0) / 4.0
    elif "ph 5" in m:
        v["ph_deficit"] = (7.0 - 5.0) / 4.0
    if "naoh" in m:
        v["alkaline"] = 1.0
    if "paste" in m or "bleach" in m:
        v["surface_treatment"] = 1.0
    # thermocycling is thermal stress, not a medium; its temperature swing is
    # already carried by temp_excess via the protocol's stated maximum
    return v


def _filler_table() -> dict[str, float]:
    """Filler wt% per product, preferring the manufacturer tables we trust most.

    Precedence: the Lodz Table 2 descriptors, then Randolph, then Kruger, then
    Alrahlah. Overlaps agree, so precedence only resolves which citation the
    value carries. Nothing is invented: products with no disclosed filler
    loading stay missing and are dropped by callers that need the feature.
    """
    out: dict[str, float] = {}
    ct = composition.composition_table()
    for r in ct.itertuples():
        if np.isfinite(r.filler_wt_pct) and not r.imputed_wt:
            out[PRODUCT_ALIASES.get(r.material, r.material)] = float(r.filler_wt_pct)
    for src in (randolph.load(), kruger.load(), alrahlah.load()):
        if "filler_wt_pct" not in src:
            continue
        for m, g in src.groupby("material"):
            v = g["filler_wt_pct"].dropna()
            key = PRODUCT_ALIASES.get(m, m)
            if len(v) and key not in out:
                out[key] = float(v.iloc[0])
    return out


def _lodz() -> pd.DataFrame:
    df = canonical_materials(pd.read_csv("output/aging_long.csv"))
    df = df[(df["property"] == "vickers_hardness") & (df["value"] > 0)]
    g = df.groupby(["material", "protocol", "medium", "duration_days", "max_temp_C"])
    out = g["value"].agg(["mean", "std", "size"]).reset_index()
    out["study"] = "lodz_2025"
    return out.rename(columns={"duration_days": "days", "max_temp_C": "temp_C"})


def _kruger() -> pd.DataFrame:
    k = kruger.load()
    k = k[k["value"] > 0]
    g = k.groupby(["material", "medium", "days"])
    out = g["value"].agg(["mean", "std", "size"]).reset_index()
    out["temp_C"] = 37.0
    out["protocol"] = out["medium"] + ", " + out["days"].astype(int).astype(str) + " d"
    out["study"] = "kruger_2018"
    return out


def _randolph() -> pd.DataFrame:
    r = randolph.load()
    r = r[(r["property"] == "vickers_hardness") & (r["value"] > 0)].copy()
    r["medium"] = r["medium"].replace({"ethanol75": "ethanol 75 Vol%"})
    r["days"] = 7.0
    r["temp_C"] = 37.0
    return pd.DataFrame({
        "material": r["material"], "medium": r["medium"], "days": r["days"],
        "temp_C": r["temp_C"], "mean": r["value"], "std": r["sd"], "size": 1,
        "protocol": r["protocol"], "study": "randolph_2016",
    })


def _alrahlah() -> pd.DataFrame:
    a = alrahlah.load_hardness()
    a = a[a["value"].notna() & (a["value"] > 0)].copy()
    a["medium"] = a["medium"].replace({
        "distilled_water": "water", "ethanol_75pct_in_water": "ethanol 75 Vol%",
        "methyl_ethyl_ketone": "methyl ethyl ketone", "dry": "dry",
    })
    a["protocol"] = a["medium"] + ", " + a["days"].astype(int).astype(str) + " d"
    return pd.DataFrame({
        "material": a["material"], "medium": a["medium"], "days": a["days"],
        "temp_C": a["temp_C"], "mean": a["value"], "std": a["sd"], "size": a["n"],
        "protocol": a["protocol"], "study": "alrahlah_2013",
    })


def _oz() -> pd.DataFrame:
    o = oz.load()
    o = o[o["value"] > 0].copy()
    o["medium"] = o["medium"].replace({"water_control": "water"})
    o["protocol"] = "oz: " + o["medium"]
    return pd.DataFrame({
        "material": o["material"], "medium": o["medium"], "days": o["days"],
        "temp_C": o["temp_C"], "mean": o["value"], "std": o["sd"], "size": o["n"],
        "protocol": o["protocol"], "study": "oz_2018",
    })


def load(require_filler: bool = True) -> pd.DataFrame:
    """One row per (laboratory, product, protocol) cell, with log-HV target."""
    df = pd.concat([_lodz(), _kruger(), _randolph(), _alrahlah(), _oz()],
                   ignore_index=True)
    df["product"] = df["material"].replace(PRODUCT_ALIASES)
    df["log_hv"] = np.log(df["mean"])

    pf = pd.DataFrame([_proto_vector(r.medium, r.days, r.temp_C)
                       for r in df.itertuples()])
    df = pd.concat([df.reset_index(drop=True), pf], axis=1)

    filler = _filler_table()
    df["filler_wt_pct"] = df["product"].map(filler)
    df["has_filler"] = df["filler_wt_pct"].notna()
    if require_filler:
        df = df[df["has_filler"]].copy()
    return df.reset_index(drop=True)


def anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Products measured by more than one laboratory: the offset-bearing edges."""
    n = df.groupby("product")["study"].nunique()
    return df[df["product"].isin(n[n > 1].index)]
