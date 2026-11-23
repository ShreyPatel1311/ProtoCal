#!/usr/bin/env python
"""Retrieval engine for external dental-composite property corpora.

What we need from a candidate paper is narrow and checkable:
  (a) a mechanical property we already hold -- Vickers hardness above all,
      since both existing laboratories measure it;
  (b) FILLER CONTENT disclosed per material, the only composition descriptor
      that transfers across studies;
  (c) a storage / aging condition, ideally including a water control;
  (d) open access, so the tables can actually be read.

Queries Europe PMC (which indexes full text and marks open access) and
cross-checks OpenAlex for OA location. Scores each hit on how many of the
required elements its full text mentions, so manual reading is spent on the
plausible ones.

Nothing here downloads a paywalled PDF; it only reports what is open.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "literature_candidates.json"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

QUERIES = [
    '(dental composite) AND (Vickers OR microhardness) AND (filler content) AND (storage OR aging OR ageing)',
    '(resin composite) AND (microhardness) AND ("filler loading" OR "filler content" OR "wt%") AND (ethanol OR water storage)',
    '(dental composite) AND (flexural strength) AND ("filler content") AND (thermocycling OR "artificial aging")',
    '(resin-based composite) AND (hardness) AND (degradation OR "food simulating" OR "storage media")',
    '(dental restorative) AND ("Knoop hardness" OR "Vickers hardness") AND (filler) AND (immersion)',
    '(dental composite) AND ("degree of conversion") AND ("filler content") AND (mechanical properties)',
]

# evidence of each required element, searched in title+abstract and full text
SIGNALS = {
    "hardness":   r"vickers|microhardness|knoop|\bHV\b|hardness",
    "filler":     r"filler (content|loading|fraction|wt|weight)|wt\.?%|filler by weight",
    "aging":      r"aging|ageing|storage|thermocycl|immersion|degradation|food.simulat",
    "control":    r"distilled water|water storage|24 h|control group|baseline",
    "tabulated":  r"mean (and|±) standard deviation|\bSD\b|±",
}


def epmc(query: str, page_size: int = 50) -> list[dict]:
    url = (f"{EPMC}?query={urllib.parse.quote(query + ' AND (OPEN_ACCESS:y)')}"
           f"&format=json&pageSize={page_size}&resultType=core")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode()).get("resultList", {}).get("result", [])
    except Exception as e:
        print(f"  query failed: {str(e)[:80]}")
        return []


def score(rec: dict) -> tuple[int, dict]:
    blob = " ".join(str(rec.get(k, "")) for k in
                    ("title", "abstractText", "keywordList")).lower()
    hits = {k: bool(re.search(p, blob, re.I)) for k, p in SIGNALS.items()}
    return sum(hits.values()), hits


def main() -> int:
    seen: dict[str, dict] = {}
    for q in QUERIES:
        print(f"query: {q[:72]}...")
        res = epmc(q)
        print(f"   {len(res)} open-access hits")
        for r in res:
            key = r.get("doi") or r.get("id")
            if not key or key in seen:
                continue
            s, hits = score(r)
            seen[key] = {
                "doi": r.get("doi"), "pmid": r.get("pmid"), "pmcid": r.get("pmcid"),
                "title": r.get("title", "")[:160],
                "journal": (r.get("journalInfo", {}) or {}).get("journal", {}).get("title"),
                "year": r.get("pubYear"), "score": s, "signals": hits,
                "isOpenAccess": r.get("isOpenAccess"),
                "fulltext": bool(r.get("hasTextMinedTerms") == "Y" or r.get("pmcid")),
                "url": (f"https://pmc.ncbi.nlm.nih.gov/articles/{r.get('pmcid')}/"
                        if r.get("pmcid") else
                        (f"https://doi.org/{r.get('doi')}" if r.get("doi") else None)),
            }
        time.sleep(1)

    ranked = sorted(seen.values(), key=lambda d: (-d["score"], -int(d["year"] or 0)))
    OUT.write_text(json.dumps(ranked, indent=2))

    print(f"\n{len(ranked)} unique open-access candidates\n")
    print(f"{'score':>5}  {'year':>4}  {'PMC':<12} title")
    for d in ranked[:30]:
        if d["score"] < 4:
            continue
        print(f"{d['score']:>5}  {d['year']:>4}  {str(d['pmcid'] or '-'):<12} {d['title'][:84]}")
    n5 = sum(1 for d in ranked if d["score"] == 5)
    n4 = sum(1 for d in ranked if d["score"] == 4)
    print(f"\nall five signals: {n5}   four of five: {n4}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
