"""Protocol encoding: one feature vector per conditioning procedure.

The previous encoding mapped 46 distinct (laboratory, protocol) pairs onto only
32 distinct feature vectors, so 14 protocols were indistinguishable from another
protocol. The collisions were not harmless:

  * "0.1M NaOH, 7 days, 60 C" collided with "water, 7500 thermo cycles,
    5 C_55 C and 0.1M NaOH, 7 days, 60 C" -- a single-step alkaline challenge
    against a two-step thermocycling-then-alkaline sequence. A model given
    identical inputs cannot distinguish the procedures the study is about.
  * Citric and lactic acid collided at every pH and time point, because only pH
    was encoded. They are not equivalent: citric acid chelates calcium from
    glass filler, lactic acid does not, so the expected damage mechanism
    differs.
  * Artificial saliva collided with distilled water, although saliva carries
    mineral content that water does not.
  * All three of Oz's surface treatments collided -- abrasive prophylaxis paste,
    oxidative bleaching, and the two in sequence.

The fix is to parse a protocol as an ORDERED SEQUENCE OF STEPS and to describe
each step by what it chemically and mechanically does, then aggregate. Features
are continuous and physically interpretable wherever possible so that a protocol
run in one laboratory still informs the others.

Reference protocol: brief water storage at 37 C, which every laboratory runs as
its control. All features are zero there, so the gauge is anchored in shared
practice rather than in one deposit's conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

FEATURES = [
    "log_days",         # log1p(cumulative exposure days) / log1p(180)
    "temp_excess",      # (max step temperature - 37) / 23
    "n_steps",          # (number of sequential conditioning steps - 1) / 2
    "log_thermocycles",  # log1p(cycles) / log1p(7500)
    "temp_amplitude",   # thermocycling swing (Tmax - Tmin) / 50
    "ethanol_frac",     # ethanol volume fraction
    "organic_solvent",  # non-aqueous organic solvent (MEK)
    "ph_deficit",       # max(0, 7 - pH) / 4
    "chelating_acid",   # citric (chelates Ca from glass filler) vs lactic
    "alkaline",         # 0.1 M NaOH
    "saliva",           # artificial saliva rather than distilled water
    "abrasive",         # mechanical prophylaxis paste
    "oxidative",        # peroxide bleaching
    "dry_stored",       # stored dry / not immersed
]

DAYS_NORM = float(np.log1p(180.0))
CYCLES_NORM = float(np.log1p(7500.0))


@dataclass
class Step:
    """One conditioning stage."""
    days: float = 0.0
    temp_C: float = 37.0
    ethanol_frac: float = 0.0
    organic_solvent: bool = False
    pH: float | None = None
    chelating: bool = False
    alkaline: bool = False
    saliva: bool = False
    abrasive: bool = False
    oxidative: bool = False
    dry: bool = False
    thermocycles: float = 0.0
    temp_min_C: float | None = None


def _parse_step(text: str, default_temp: float = 37.0) -> Step:
    """Describe one conditioning stage from its free-text description."""
    t = text.lower()
    s = Step(temp_C=default_temp)

    cyc = re.search(r"(\d+)\s*thermo\s*cycles?", t) or re.search(r"(\d+)\s*cycles?", t)
    if cyc:
        s.thermocycles = float(cyc.group(1))
    # a thermocycling range appears as "5°C_55°C" or "5 C - 55 C"
    temps = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*°?\s*c\b", t)]
    if s.thermocycles and len(temps) >= 2:
        s.temp_min_C, s.temp_C = min(temps), max(temps)
    elif temps:
        s.temp_C = max(temps)

    d = re.search(r"(\d+(?:\.\d+)?)\s*day", t)
    h = re.search(r"(\d+(?:\.\d+)?)\s*h\b", t)
    if d:
        s.days = float(d.group(1))
    elif h:
        s.days = float(h.group(1)) / 24.0
    elif s.thermocycles:
        # thermocycling duration is not reported in the deposit; the cycle count
        # carries the dose, so exposure days stay at zero rather than invented
        s.days = 0.0

    if "naoh" in t:
        s.alkaline = True
    if "ethanol" in t or "etoh" in t:
        m = re.search(r"(\d+)\s*(?:vol|%|pct)", t)
        if "75" in t:
            s.ethanol_frac = 0.75
        elif m:
            s.ethanol_frac = float(m.group(1)) / 100.0
        else:
            s.ethanol_frac = 0.75
    if "ketone" in t or re.search(r"\bmek\b", t):
        s.organic_solvent = True
    if "citric" in t:
        s.chelating = True
    ph = re.search(r"ph\s*(\d+(?:\.\d+)?)", t)
    if ph:
        s.pH = float(ph.group(1))
    if "saliva" in t:
        s.saliva = True
    if "paste" in t or "prophyla" in t:
        s.abrasive = True
    if "bleach" in t or "peroxide" in t:
        s.oxidative = True
    if t.strip() in {"dry", "dry 0d"} or t.startswith("dry"):
        s.dry = True
    return s


def parse(protocol: str, default_temp: float = 37.0) -> list[Step]:
    """Split a protocol description into its ordered steps.

    Lodz writes multi-stage protocols as "<stage> and <stage>"; Oz writes a
    two-stage treatment as "<stage>_then_<stage>". Both must survive as two
    steps or the sequence is lost.
    """
    text = str(protocol)
    parts = re.split(r"\s+and\s+|_then_|\s*\+\s*then\s*", text)
    parts = [p for p in (q.strip() for q in parts) if p]
    if not parts:
        parts = [text]
    return [_parse_step(p, default_temp) for p in parts]


def encode(protocol: str, default_temp: float = 37.0,
           days_override: float | None = None) -> dict[str, float]:
    """Feature vector for a protocol description.

    `days_override` is used where a loader already knows the exposure time more
    reliably than the protocol string does (Kruger's incubation times are a
    column, not part of a label).
    """
    steps = parse(protocol, default_temp)
    total_days = sum(s.days for s in steps)
    if days_override is not None:
        total_days = float(days_override)
    max_temp = max(s.temp_C for s in steps)
    cycles = sum(s.thermocycles for s in steps)
    amp = max(((s.temp_C - s.temp_min_C) if s.temp_min_C is not None else 0.0)
              for s in steps)
    ph_values = [s.pH for s in steps if s.pH is not None]

    return {
        "log_days": float(np.log1p(max(total_days, 0.0)) / DAYS_NORM),
        "temp_excess": float((max_temp - 37.0) / 23.0),
        "n_steps": float((len(steps) - 1) / 2.0),
        "log_thermocycles": float(np.log1p(cycles) / CYCLES_NORM),
        "temp_amplitude": float(amp / 50.0),
        "ethanol_frac": float(max(s.ethanol_frac for s in steps)),
        "organic_solvent": float(any(s.organic_solvent for s in steps)),
        "ph_deficit": float(max(0.0, 7.0 - min(ph_values)) / 4.0) if ph_values else 0.0,
        "chelating_acid": float(any(s.chelating for s in steps)),
        "alkaline": float(any(s.alkaline for s in steps)),
        "saliva": float(any(s.saliva for s in steps)),
        "abrasive": float(any(s.abrasive for s in steps)),
        "oxidative": float(any(s.oxidative for s in steps)),
        "dry_stored": float(any(s.dry for s in steps)),
    }


def collisions(labelled: list[tuple[str, str]], **kw) -> list[list[tuple[str, str]]]:
    """Groups of (laboratory, protocol) pairs sharing one feature vector."""
    seen: dict[tuple, list[tuple[str, str]]] = {}
    for lab, proto in labelled:
        v = encode(proto, **kw)
        seen.setdefault(tuple(round(v[f], 6) for f in FEATURES), []).append((lab, proto))
    return [g for g in seen.values() if len(g) > 1]
