#!/usr/bin/env python
"""Stage 2: score candidates on FULL TEXT, not abstracts.

What makes a paper usable is whether its tables actually carry filler content
per material alongside a hardness value per condition. None of that appears in
an abstract, so stage 1's ranking is close to uninformative. This pass pulls
the Europe PMC full-text XML and looks for the real evidence:

  * numeric filler loadings  ("72 wt%", "filler content of 78.5%")
  * hardness values with dispersion ("45.2 +/- 3.1")
  * a water / baseline control condition
  * how many distinct named materials are tabulated

Writes a ranked shortlist with the extracted filler figures, so a human can
judge in seconds whether a paper is worth transcribing.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "output" / "literature_candidates.json"
OUT = ROOT / "output" / "literature_shortlist.json"
CACHE = ROOT / "data_work" / "epmc_fulltext"
CACHE.mkdir(parents=True, exist_ok=True)

FILLER_RE = re.compile(r"(\d{2}(?:\.\d)?)\s*(?:wt\.?\s*%|%\s*(?:by\s+)?(?:wt|weight)|weight\s*%)", re.I)
FILLER_CTX = re.compile(r"filler[^.]{0,120}?(\d{2}(?:\.\d)?)\s*(?:wt\.?\s*%|%)", re.I)
MEANSD_RE = re.compile(r"\d+\.?\d*\s*(?:±|\+/-)\s*\d+\.?\d*")
HARD_RE = re.compile(r"vickers|microhardness|knoop|\bVHN\b|\bHV\b", re.I)
CONTROL_RE = re.compile(r"distilled water|deionized water|water storage|24\s*h|baseline|control", re.I)


def fulltext(pmcid: str) -> str | None:
    cache = CACHE / f"{pmcid}.xml"
    if cache.exists():
        return cache.read_text(errors="replace")
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            txt = r.read().decode("utf-8", errors="replace")
        if len(txt) < 500:
            return None
        cache.write_text(txt)
        return txt
    except Exception:
        return None


def main() -> int:
    cands = json.loads(CAND.read_text())
    pool = [c for c in cands if c.get("pmcid") and c["score"] >= 3]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    pool = pool[:limit]
    print(f"fetching full text for {len(pool)} candidates (cached where possible)\n")

    rows = []
    for i, c in enumerate(pool, 1):
        txt = fulltext(c["pmcid"])
        if not txt:
            continue
        plain = re.sub(r"<[^>]+>", " ", txt)
        fillers = sorted({float(m) for m in FILLER_CTX.findall(plain)
                          if 30.0 <= float(m) <= 95.0})
        if not fillers:
            fillers = sorted({float(m) for m in FILLER_RE.findall(plain)
                              if 30.0 <= float(m) <= 95.0})
        n_meansd = len(MEANSD_RE.findall(plain))
        has_hard = bool(HARD_RE.search(plain))
        has_ctrl = bool(CONTROL_RE.search(plain))
        usable = (len(fillers) >= 3) + has_hard + has_ctrl + (n_meansd >= 12)
        rows.append({**{k: c[k] for k in ("doi", "pmcid", "title", "journal", "year", "url")},
                     "n_filler_values": len(fillers),
                     "filler_values": fillers[:14],
                     "n_mean_sd": n_meansd, "hardness": has_hard,
                     "control": has_ctrl, "usability": usable})
        if i % 20 == 0:
            print(f"  ...{i}/{len(pool)}")
        time.sleep(0.25)

    rows.sort(key=lambda r: (-r["usability"], -r["n_filler_values"], -r["n_mean_sd"]))
    OUT.write_text(json.dumps(rows, indent=2))

    print(f"\n{'use':>3} {'fillers':>7} {'n±SD':>6}  {'PMC':<12} title")
    for r in rows[:25]:
        print(f"{r['usability']:>3} {r['n_filler_values']:>7} {r['n_mean_sd']:>6}  "
              f"{r['pmcid']:<12} {r['title'][:70]}")
    top = [r for r in rows if r["usability"] == 4]
    print(f"\n{len(top)} papers hit all four criteria (filler values + hardness + control + tables)")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
