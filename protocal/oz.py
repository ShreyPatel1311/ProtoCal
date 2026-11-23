"""Oz et al. (2018), Ataturk Univ Dent Fac J: 3 composites x 4 surface treatments.

doi:10.17567/ataunidfd.416928

This study matters to the pooled corpus for one structural reason rather than
for its own protocol axis: it is the only source measuring BOTH GrandioSO
(shared with Kruger) and Filtek Ultimate (shared with Lodz), so it is the edge
that connects the Kruger component of the laboratory graph to the Lodz
component. Without it the two largest corpora have no path between them and no
common offset scale.

Its protocol axis -- prophylactic paste and in-office bleaching -- is
mechanical/oxidative rather than solvent-driven, so it also tests whether the
single protocol-severity axis found for solvent and thermal challenges extends
to surface treatments. With only three materials that test is weak, and it is
reported as such.

Filler loading is not disclosed for Filtek Supreme XT Flow. It is left missing;
no value is invented for it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data"

LOAD_GF = 500.0

# Products shared with other corpora, used as cross-laboratory anchors.
ANCHORS = {"GrandioSO": "kruger_2018", "Filtek Ultimate": "lodz_2025"}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "oz_2018_hardness.csv")
    return pd.DataFrame({
        "study": "oz_2018",
        "material": df["material"].str.strip(),
        "property": "vickers_hardness",
        # the treatment IS the protocol here; all groups share 14 d water storage
        "medium": df["treatment_after_water_storage"],
        "days": df["water_storage_days"].astype(float),
        "temp_C": df["water_storage_temp_c"].astype(float),
        "value": df["mean_hv"],
        "sd": df["sd_hv"],
        "n": df["specimens_per_group"],
        "filler_wt_pct": df["filler_wt_pct"],
        "load_gf": df["measurement_load_gf"].astype(float),
        "censored_upper": np.nan,
        "block": df["group_code"].str[0],
        "is_control": df["is_control"].astype(bool),
    })
