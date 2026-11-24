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

from . import (alrahlah, composition, filler as filler_mod, kruger, methods,
               oz, protocols, randolph)
from .stats import canonical_materials

# Same physical product sold under different names in different markets.
PRODUCT_ALIASES = {
    "Filtek Supreme XTE": "Filtek Ultimate",
    "Tetric Evo Ceram": "Tetric EvoCeram",
}

# Protocol covariates come from protocal.protocols, which parses a protocol as
# an ordered sequence of steps. The previous inline encoding collapsed 46
# (laboratory, protocol) pairs onto 32 distinct feature vectors, so 14
# protocols were indistinguishable from another protocol -- including a
# single-step alkaline challenge against a thermocycling-then-alkaline
# sequence. See protocal.protocols for the full list and the reasoning.
PROTO_FEATURES = protocols.FEATURES

#: Measurement-method covariates. Constant within a laboratory, hence perfectly
#: collinear with laboratory identity: they CANNOT serve as model features
#: alongside a per-laboratory offset. Carried for the separate question of
#: whether the fitted offsets are explained by method differences -- they are
#: not (offset ~ log load gives r2 = 0.04, p = 0.79 over the four
#: true-Vickers laboratories).
METHOD_COLUMNS = ["load_gf", "dwell_s", "converted_from_martens",
                  "dried_before_test"]


def _encode_protocols(df: pd.DataFrame) -> pd.DataFrame:
    """Feature block for each row's full protocol sequence."""
    recs = [protocols.encode(r.protocol_text, days_override=r.days)
            for r in df.itertuples()]
    return pd.DataFrame(recs, index=df.index)


#: Filler provenance now lives in protocal.filler, which keeps the three
#: distinct quantities apart: the manufacturer figure, Randolph's TGA inorganic
#: residue W_i, and Randolph's acetone-insoluble W_total. An earlier version
#: here recorded Randolph's 82.5 wt% for Tetric EvoCeram as MEASURED; it is the
#: manufacturer figure, and the measured value is 70.5 wt%. The same mistake
#: applied to every Randolph product -- see that module for the full table and
#: for why variant="measured" introduces a definition/laboratory confound of
#: its own.
FILLER_VARIANTS = filler_mod.VARIANTS


def _lodz() -> pd.DataFrame:
    df = canonical_materials(pd.read_csv("output/aging_long.csv"))
    df = df[(df["property"] == "vickers_hardness") & (df["value"] > 0)]
    g = df.groupby(["material", "protocol", "medium", "duration_days", "max_temp_C"])
    out = g["value"].agg(["mean", "std", "size"]).reset_index()
    out["study"] = "lodz_2025"
    out = out.rename(columns={"duration_days": "days", "max_temp_C": "temp_C"})
    # the deposit's protocol string spells out the whole sequence, including the
    # thermocycling stage, so it is the authoritative description
    out["protocol_text"] = out["protocol"]
    out["days"] = [sum(st.days for st in protocols.parse(t))
                   for t in out["protocol_text"]]
    return out


def _kruger() -> pd.DataFrame:
    k = kruger.load()
    k = k[k["value"] > 0]
    g = k.groupby(["material", "medium", "days"])
    out = g["value"].agg(["mean", "std", "size"]).reset_index()
    out["temp_C"] = 37.0
    out["protocol"] = out["medium"] + ", " + out["days"].astype(int).astype(str) + " d"
    out["study"] = "kruger_2018"
    # incubation time is a column here, not part of a label. Day 0 is the shared
    # pre-storage baseline the paper measured once per composite.
    out["protocol_text"] = np.where(
        out["days"] == 0, "dry baseline, 0 days, 37C",
        out["medium"] + ", " + out["days"].astype(int).astype(str) + " days, 37C")
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
        "protocol_text": r["medium"] + ", 7 days, 37C",
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
        "protocol_text": np.where(
            a["medium"] == "dry", "dry, 0 days, 23C",
            a["medium"] + ", " + a["days"].astype(int).astype(str) + " days, 37C"),
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
        # every group had 14 d water storage FIRST; the treatment is a second
        # step, and the combined arm is two treatments genuinely in sequence
        "protocol_text": np.where(
            o["medium"] == "water", "water, 14 days, 37C",
            "water, 14 days, 37C and " + o["medium"].str.replace("_", " ")),
    })


def load(require_filler: bool = True, *, collapse_replicates: bool = True,
         exclude_kruger: bool = False,
         filler_variant: str = "manufacturer"
         ) -> pd.DataFrame:
    """One row per (laboratory, product, protocol) cell, with a log-HV target.

    collapse_replicates
        Alrahlah measured its dry baseline once inside each of three medium
        blocks, so that cell appears three times per material. Those are
        replicate measurements of ONE cell, not three cells: left in place they
        triple that laboratory's weight on the dry protocol. Collapsing them is
        the default; their spread is the within-laboratory repeatability and is
        reported separately by `alrahlah.repeatability()`.
    exclude_kruger
        Kruger did not measure Vickers hardness. A Fischerscope HM 2000 measured
        Martens hardness at 1 N and converted it to a Vickers-equivalent by an
        undisclosed relation, and its specimens were dried under vacuum for 24 h
        before testing -- which reverses the plasticisation that the other
        laboratories were measuring. Every headline result should be reported
        both ways.
    """
    frames = [_lodz(), _randolph(), _alrahlah(), _oz()]
    if not exclude_kruger:
        frames.insert(1, _kruger())
    df = pd.concat(frames, ignore_index=True)
    df["product"] = df["material"].replace(PRODUCT_ALIASES)

    if collapse_replicates:
        keys = ["study", "product", "material", "protocol", "protocol_text",
                "medium", "days", "temp_C"]
        n_before = len(df)
        agg = df.groupby(keys, dropna=False).agg(
            mean=("mean", "mean"), std=("std", "mean"),
            size=("size", "sum"), n_replicate_cells=("mean", "size"))
        df = agg.reset_index()
        df.attrs["collapsed_cells"] = n_before - len(df)

    df["log_hv"] = np.log(df["mean"])
    df = df.reset_index(drop=True)
    df = pd.concat([df, _encode_protocols(df)], axis=1)
    meth = methods.table().set_index("study")
    for c in METHOD_COLUMNS:
        df[c] = df["study"].map(meth[c]).astype(float)

    ft = filler_mod.table(filler_variant, PRODUCT_ALIASES).set_index("product")
    df["filler_wt_pct"] = df["product"].map(ft["filler_wt_pct"])
    df["filler_source"] = df["product"].map(ft["filler_source"])
    df["filler_definition"] = df["product"].map(ft["filler_definition"])
    if "prepolymer_discrepant" in ft:
        df["filler_prepolymer_discrepant"] = (
            df["product"].map(ft["prepolymer_discrepant"]).astype("boolean")
            .fillna(False).astype(bool))
    else:
        df["filler_prepolymer_discrepant"] = False
    df["has_filler"] = df["filler_wt_pct"].notna()
    if require_filler:
        df = df[df["has_filler"]].copy()
    return df.reset_index(drop=True)


#: Protocol features exercised by only ONE laboratory. When that laboratory is
#: held out these features have no training support at all, so the model is
#: extrapolating along a direction it has never seen. This is not a modelling
#: choice to tune -- it is a property of the corpus, and it makes
#: leave-one-laboratory-out ill-posed for the laboratories that own them.
#:
#:   alkaline, log_thermocycles   only lodz_2025     (33 and 11 of its 44 cells)
#:   ph_deficit, chelating_acid,
#:   saliva                       only kruger_2018
#:   organic_solvent              only alrahlah_2013
#:   abrasive, oxidative          only oz_2018
#:
#: The earlier colliding encoding hid this: it collapsed Lodz's alkaline
#: protocols onto vectors shared with other laboratories, so held-out Lodz got a
#: flatteringly low error (0.417) from inputs that were simply wrong. With the
#: protocols correctly distinguished the same split gives 1.163, which is the
#: honest number for extrapolating to an unseen protocol family.
SINGLE_LAB_FEATURES = ("log_thermocycles", "organic_solvent", "ph_deficit",
                       "chelating_acid", "alkaline", "saliva", "abrasive",
                       "oxidative")

#: Features defining the one protocol family every laboratory shares: immersion
#: in water or aqueous ethanol, with no acid, alkali, saliva, organic solvent,
#: thermocycling or surface treatment. 103 cells over 29 products and all five
#: laboratories, retaining all six anchor products -- the only subcorpus on
#: which cross-laboratory protocol transfer is a well-posed question.
AQUEOUS_FAMILY_EXCLUDE = SINGLE_LAB_FEATURES


def aqueous_family(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the shared water / aqueous-ethanol protocol family."""
    keep = np.ones(len(df), bool)
    for f in AQUEOUS_FAMILY_EXCLUDE:
        keep &= (df[f].to_numpy(float) == 0.0)
    return df[keep].copy().reset_index(drop=True)


def anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Products measured by more than one laboratory: the offset-bearing edges."""
    n = df.groupby("product")["study"].nunique()
    return df[df["product"].isin(n[n > 1].index)]
