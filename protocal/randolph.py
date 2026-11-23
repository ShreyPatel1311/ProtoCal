"""Loader for Randolph et al. 2016, Dental Materials 32(12):1586-1599.

"Filler characteristics of modern dental resin composites and their influence
on physico-mechanical properties." Open-access copy from the University of
Birmingham repository; tables parsed from the PDF rather than retyped, so the
transcription is reproducible and auditable.

A THIRD independent laboratory, and the most useful external corpus found:
17 commercial composites, filler content in wt% and vol%, and flexural
modulus, flexural strength and Vickers hardness measured after one week at
37 C in EITHER water OR 75/25 vol% ethanol/water.

Three materials overlap with the Lodz corpus and act as cross-laboratory
calibration anchors: Admira Fusion, Kalore, and Filtek Supreme XTE -- which is
the European designation for the product Lodz lists as Filtek Ultimate.
Grandio and Grandio Flow here are NOT the GrandioSO of the Kruger corpus;
they are different products and are deliberately not linked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PDF = "data_raw/external_papers/randolph2016_filler_characteristics.pdf"

# Lodz name -> Randolph name, for the shared-material anchors
ANCHORS = {
    "Admira Fusion": "Admira Fusion",
    "Kalore": "Kalore",
    "Filtek Ultimate": "Filtek Supreme XTE",   # same product, EU vs US name
}

# filler content wt%/vol% as given in Table 1 bis ("NS" = not stated)
FILLER = {
    "Admira Fusion": (84.0, None), "Clearfil Majesty Posterior": (92.0, 82.0),
    "Clearfil Majesty ES Flow": (75.0, 59.0), "ELS": (74.0, 49.0),
    "ELS Flow": (53.0, 28.0), "Filtek Silorane": (76.0, 55.0),
    "Filtek Supreme XTE": (78.5, 63.3), "Grandio": (87.0, 71.4),
    "Grandio Flow": (80.2, 65.7), "Kalore": (82.0, None),
    "Tetric Evo Ceram": (82.5, None), "Venus Diamond": (81.0, 64.0),
    "Venus Diamond Flow": (65.0, 41.0), "Venus Pearl": (80.0, 59.0),
}

NUM = r"(\d+\.?\d*)\s*\(([\d.]+)\)"


def _table4_text(root: Path) -> str:
    import pymupdf
    doc = pymupdf.open(root / PDF)
    full = "\n".join(p.get_text() for p in doc)
    i = full.find("Table 4 Mechanical properties")
    j = full.find("Table 5 Degradation", i)
    return full[i:j]


def load(root: Path | None = None) -> pd.DataFrame:
    """Long format: material, property, medium, value, sd, filler_wt_pct."""
    root = root or Path(__file__).resolve().parent.parent
    txt = re.sub(r"\s+", " ", _table4_text(root))
    rows = []
    for mat in FILLER:
        k = txt.find(mat)
        if k < 0:
            continue
        # the six numbers after the name are E_flex, sigma, HV in water then EtOH
        seg = txt[k + len(mat): k + len(mat) + 230]
        vals = re.findall(NUM, seg)
        if len(vals) < 6:
            continue
        order = [("flexural_modulus_GPa", "water"), ("flexural_strength_MPa", "water"),
                 ("vickers_hardness", "water"),
                 ("flexural_modulus_GPa", "ethanol75"), ("flexural_strength_MPa", "ethanol75"),
                 ("vickers_hardness", "ethanol75")]
        for (prop, medium), (v, s) in zip(order, vals[:6]):
            rows.append({"material": mat, "property": prop, "medium": medium,
                         "protocol": f"{medium}, 1 week, 37C",
                         "value": float(v), "sd": float(s),
                         "filler_wt_pct": FILLER[mat][0],
                         "filler_vol_pct": FILLER[mat][1], "study": "randolph2016"})
    return pd.DataFrame(rows)
