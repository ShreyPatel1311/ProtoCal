"""Measurement methodology per laboratory, from the source publications.

Pooling hardness across laboratories is only defensible if the measurements are
the same KIND of measurement. They are not. Recorded here because three of these
differences are large enough to account for most of what a per-laboratory offset
would otherwise absorb:

  * TEST LOAD spans a factor of ten, 1 N (~102 gf) to 1000 gf. Vickers hardness
    of a resin composite is load-dependent through the indentation size effect,
    so a laboratory's load partly determines its hardness scale before any
    material or protocol effect.
  * KRUGER DID NOT MEASURE VICKERS HARDNESS. A Fischerscope HM 2000 measured
    Martens hardness by instrumented indentation and converted it to a
    Vickers-equivalent number. The conversion is not given in the paper. These
    values are not interchangeable with an optically-measured diagonal.
  * KRUGER DRIED ITS SPECIMENS for 24 h under vacuum before testing. Softening
    by water and ethanol is largely plasticisation, which drying reverses, so
    this laboratory measured recovered specimens while the others measured them
    as aged. This is the most likely reason protocol effects account for only
    ~2% of variance in that corpus and its leading principal component is
    indistinguishable from its own permutation null.

Sources
  lodz_2025      deposit readme, Zenodo 14576242 ("readme data set.docx")
  kruger_2018    Kruger J, Maletz R, Ottl P, Warkentin M. PLoS ONE 2018;13(4):
                 e0195160, Section 2.4, doi:10.1371/journal.pone.0195160
  randolph_2016  Randolph et al., Section 3, local copy
                 data_raw/external_papers/randolph2016_filler_characteristics.pdf
  alrahlah_2013  Manchester thesis full text, p.93 (method), p.111 (Table 7.2)
  oz_2018        Oz et al., doi:10.17567/ataunidfd.416928, extracted table
"""

from __future__ import annotations

import pandas as pd

GF_PER_NEWTON = 101.9716

METHODS = {
    "lodz_2025": dict(
        instrument="Zwick ZHVum (Zwick-Roell)", technique="vickers_optical",
        load_gf=1000.0, dwell_s=10.0, converted_from_martens=False,
        dried_before_test=False, indents_per_group=9, specimens_per_group=3,
        polished=None,
        note="9 indentations across 3 of the 9 DTS specimens per group",
    ),
    "kruger_2018": dict(
        instrument="Fischerscope HM 2000 (Helmut Fischer)",
        technique="instrumented_indentation",
        load_gf=1.0 * GF_PER_NEWTON, dwell_s=5.0, converted_from_martens=True,
        dried_before_test=True, indents_per_group=10, specimens_per_group=5,
        polished=True,
        note=("Martens hardness converted to a Vickers-equivalent; conversion "
              "not stated in the paper. 1 N at 0.2 N/s, 5 s hold. Specimens "
              "dried 24 h under vacuum over silica gel before measurement. "
              "Media refreshed every 4 weeks; 37 C under continuous agitation."),
    ),
    "randolph_2016": dict(
        instrument="Durimet (Leitz)", technique="vickers_optical",
        load_gf=200.0, dwell_s=30.0, converted_from_martens=False,
        dried_before_test=False, indents_per_group=3, specimens_per_group=3,
        polished=False,
        note=("reported as HV(0.2/30). Cured, UNPOLISHED discs 2 mm x 5 mm dia; "
              "mean of 3 diagonals per specimen"),
    ),
    "alrahlah_2013": dict(
        instrument="not stated in thesis", technique="vickers_optical",
        load_gf=300.0, dwell_s=15.0, converted_from_martens=False,
        dried_before_test=False, indents_per_group=9, specimens_per_group=3,
        polished=None, note="3 specimens x 3 indentations per medium group",
    ),
    "oz_2018": dict(
        instrument="not stated in article", technique="vickers_optical",
        load_gf=500.0, dwell_s=15.0, converted_from_martens=False,
        dried_before_test=False, indents_per_group=30, specimens_per_group=10,
        polished=None, note="10 specimens x 3 indentations per group",
    ),
}

#: Laboratories whose hardness is NOT an optically-measured Vickers diagonal.
NON_VICKERS = tuple(k for k, v in METHODS.items()
                    if v["technique"] != "vickers_optical")


def table() -> pd.DataFrame:
    return pd.DataFrame(METHODS).T.rename_axis("study").reset_index()
