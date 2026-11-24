"""Filler loading per product, with explicit provenance and definition.

There is no single number called "filler content". Randolph et al. report three
quantities for the same material and they do not agree:

  W_i       total inorganic content, measured by thermogravimetric analysis.
            An ash residue: prepolymerised filler burns off and is NOT counted.
  W_total   acetone-insoluble fraction, measured by centrifugation. Counts
            prepolymerised filler as filler.
  Table 1   the manufacturer's own figure, which is neither measurement.

For most materials these are close. For materials containing prepolymerised
filler they are far apart, and those materials are disproportionately the
cross-laboratory anchors:

  product                manufacturer   W_i (TGA)   difference
  Tetric EvoCeram               82.5        70.5        -12.0
  Kalore                        82.0        71.1        -10.9
  Filtek Ultimate               78.5        73.0         -5.5
  Venus Diamond Flow            65.0        60.1         -4.9
  Admira Fusion                 84.0        86.3         +2.3

An earlier version of this project recorded Randolph's 82.5 wt% for Tetric
EvoCeram as a MEASURED value. It is the manufacturer figure from Table 1; the
measured inorganic fraction is 70.5 wt% (Table 2). The same error applied to
every Randolph product. It also explains a false reassurance: five of six
multi-source products appeared to "agree exactly" on filler loading, but that
was two copies of the same manufacturer datasheet rather than independent
confirmation.

A FURTHER CAUTION about `variant="measured"`. Measured values exist only for
the 14 Randolph products. Those products are precisely the cross-laboratory
anchors, so using measured values where available gives the ANCHORS one filler
definition and the TEST products another, and the definition then correlates
with laboratory. For calibration analyses that is a confound, not an
improvement. `variant="manufacturer"` is definitionally consistent and
`variant="randolph_only"` is both consistent and measured at the cost of the
other laboratories. Headline results should be reported under all three.

Sources
  Randolph et al., Tables 1 and 2, local copy at
  data_raw/external_papers/randolph2016_filler_characteristics.pdf (pp. 32-34)
  Lodz descriptor table: Szczesio-Wlodarczyk et al., J Funct Biomater 2025;
  16(2):49, Table 2 -- manufacturer-provided
  Kruger: PLoS ONE 2018;13(4):e0195160 -- manufacturer-provided
  Alrahlah: Manchester thesis p.95 -- manufacturer-provided
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Randolph Table 2: measured inorganic content by TGA (W_i) and the
#: acetone-insoluble fraction by centrifugation (W_total), wt%, sd in the paper.
#: Keyed by Randolph's own material names.
RANDOLPH_TABLE2 = {
    #                            density   W_i    W_total
    "Admira Fusion":             (2.18,   86.3,   78.1),
    "Clearfil Majesty ES Flow":  (1.80,   67.2,   66.9),
    "Clearfil Majesty Posterior": (2.39,  88.1,   78.9),
    "ELS":                       (2.07,   72.7,   72.1),
    "ELS Flow":                  (1.68,   52.2,   52.1),
    "Filtek Silorane":           (2.02,   77.8,   74.2),
    "Filtek Supreme XTE":        (1.90,   73.0,   64.9),
    "Grandio":                   (2.06,   85.8,   83.2),
    "Grandio Flow":              (1.91,   78.6,   76.1),
    "Kalore":                    (1.97,   71.1,   79.8),
    "Tetric Evo Ceram":          (2.18,   70.5,   80.9),
    "Venus Diamond":             (2.20,   78.6,   76.4),
    "Venus Diamond Flow":        (1.92,   60.1,   63.3),
    "Venus Pearl":               (2.02,   75.5,   74.7),
}

#: Products whose manufacturer figure and TGA measurement differ by more than
#: 3 wt%. Prepolymerised filler is the known cause: a datasheet may count it,
#: a TGA ash residue cannot.
PREPOLYMER_DISCREPANT = ("Tetric EvoCeram", "Kalore", "Filtek Ultimate",
                         "Venus Diamond Flow", "Clearfil Majesty ES Flow")

VARIANTS = ("manufacturer", "measured", "measured_wtotal", "randolph_only")


def table(variant: str = "manufacturer", product_aliases: dict | None = None
          ) -> pd.DataFrame:
    """Filler loading per product.

    variant
        manufacturer      datasheet figures throughout; one definition for every
                          product, which is what calibration analyses need.
        measured          Randolph's TGA W_i where available, manufacturer
                          elsewhere. Most accurate per product, but the
                          definition then correlates with laboratory.
        measured_wtotal   Randolph's acetone-insoluble W_total where available;
                          counts prepolymerised filler, so it is the measured
                          quantity closest in meaning to a datasheet figure.
        randolph_only     TGA W_i, and ONLY the 14 Randolph products.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    from . import alrahlah, composition, kruger, randolph
    alias = product_aliases or {}

    rows: list[dict] = []

    def add(product, value, source, definition):
        rows.append({"product": alias.get(product, product),
                     "filler_wt_pct": float(value), "filler_source": source,
                     "filler_definition": definition})

    col = 1 if variant in ("measured", "randolph_only") else 2
    definition = ("tga_inorganic" if variant in ("measured", "randolph_only")
                  else "acetone_insoluble")
    if variant != "manufacturer":
        for m, spec in RANDOLPH_TABLE2.items():
            add(m, spec[col], "randolph_table2", definition)
    if variant == "randolph_only":
        return pd.DataFrame(rows).drop_duplicates("product").reset_index(drop=True)

    have = {r["product"] for r in rows}
    # manufacturer figures, in the same precedence as before
    for r in composition.composition_table().itertuples():
        p = alias.get(r.material, r.material)
        if p not in have and np.isfinite(r.filler_wt_pct) and not r.imputed_wt:
            add(r.material, r.filler_wt_pct, "lodz_table_manufacturer",
                "manufacturer"); have.add(p)
    for name, src in (("randolph_table1", randolph.load()),
                      ("kruger_paper", kruger.load()),
                      ("alrahlah_thesis", alrahlah.load())):
        if "filler_wt_pct" not in src:
            continue
        for m, g in src.groupby("material"):
            p = alias.get(m, m)
            v = g["filler_wt_pct"].dropna()
            if p not in have and len(v):
                add(m, float(v.iloc[0]), name, "manufacturer"); have.add(p)

    out = pd.DataFrame(rows).drop_duplicates("product").reset_index(drop=True)
    out["prepolymer_discrepant"] = out["product"].isin(PREPOLYMER_DISCREPANT)
    return out


def comparison(product_aliases: dict | None = None) -> pd.DataFrame:
    """Manufacturer vs measured, for the products where both exist."""
    man = table("manufacturer", product_aliases).set_index("product")
    tga = table("measured", product_aliases).set_index("product")
    wto = table("measured_wtotal", product_aliases).set_index("product")
    idx = [p for p in tga.index if tga.loc[p, "filler_source"] == "randolph_table2"]
    return pd.DataFrame({
        "manufacturer": man.reindex(idx)["filler_wt_pct"],
        "W_i_tga": tga.reindex(idx)["filler_wt_pct"],
        "W_total_acetone": wto.reindex(idx)["filler_wt_pct"],
    }).assign(diff_tga_minus_manuf=lambda d: (d.W_i_tga - d.manufacturer).round(1))
