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

from . import alrahlah, composition, kruger, methods, oz, protocols, randolph
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


#: Filler loading disagrees between sources for exactly one product.
#: Randolph determined inorganic content by thermogravimetric analysis, so that
#: value is an ASH RESIDUE; Alrahlah quotes the manufacturer. Tetric EvoCeram
#: contains prepolymerised filler, which a manufacturer figure may count as
#: filler and a TGA residue will not, so the gap has a known cause rather than
#: being a transcription error. Measured values take precedence, the conflict is
#: flagged on every affected row, and `filler_variant="manufacturer"` refits with
#: the other choice as a sensitivity check.
FILLER_CONFLICTS = {
    "Tetric EvoCeram": {"measured": 82.5, "manufacturer": 76.0,
                        "reason": "prepolymerised filler counted differently by "
                                  "TGA ash residue and manufacturer datasheet"},
}

#: Source precedence for filler loading. Measured (TGA/sedimentation) before
#: manufacturer-reported, and the Lodz descriptor table before the rest only
#: because it carries the citation we lean on elsewhere; the five products
#: reported by more than one source and not listed in FILLER_CONFLICTS agree
#: exactly, so precedence changes the citation, not the number.
FILLER_PRECEDENCE = ("randolph", "lodz_table", "kruger", "alrahlah")


def _filler_table(variant: str = "measured") -> dict[str, tuple[float, str]]:
    """Filler wt% per product as {product: (value, source)}.

    Products with no disclosed filler loading stay missing; nothing is invented.
    Values the Lodz table had to impute from the wt%/vol% regression are
    excluded, because an imputed composition feature would silently become a
    predictor of its own imputation model.
    """
    by_src: dict[str, dict[str, float]] = {k: {} for k in FILLER_PRECEDENCE}
    ct = composition.composition_table()
    for r in ct.itertuples():
        if np.isfinite(r.filler_wt_pct) and not r.imputed_wt:
            by_src["lodz_table"][PRODUCT_ALIASES.get(r.material, r.material)] = \
                float(r.filler_wt_pct)
    for name, src in (("randolph", randolph.load()), ("kruger", kruger.load()),
                      ("alrahlah", alrahlah.load())):
        if "filler_wt_pct" not in src:
            continue
        for m, g in src.groupby("material"):
            v = g["filler_wt_pct"].dropna()
            if len(v):
                by_src[name][PRODUCT_ALIASES.get(m, m)] = float(v.iloc[0])

    out: dict[str, tuple[float, str]] = {}
    for name in FILLER_PRECEDENCE:
        for prod, val in by_src[name].items():
            out.setdefault(prod, (val, name))
    for prod, spec in FILLER_CONFLICTS.items():
        if prod in out:
            out[prod] = (float(spec[variant]), f"conflict:{variant}")
    return out


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
         exclude_kruger: bool = False, filler_variant: str = "measured"
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

    filler = _filler_table(filler_variant)
    df["filler_wt_pct"] = df["product"].map(lambda p: filler.get(p, (np.nan,))[0])
    df["filler_source"] = df["product"].map(
        lambda p: filler.get(p, (np.nan, "none"))[1])
    df["filler_conflict"] = df["product"].isin(FILLER_CONFLICTS)
    df["has_filler"] = df["filler_wt_pct"].notna()
    if require_filler:
        df = df[df["has_filler"]].copy()
    return df.reset_index(drop=True)


def anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Products measured by more than one laboratory: the offset-bearing edges."""
    n = df.groupby("product")["study"].nunique()
    return df[df["product"].isin(n[n > 1].index)]
