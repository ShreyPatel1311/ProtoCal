#!/usr/bin/env python
"""Download the open-access dental biomaterials datasets into data_raw/.

Each Zenodo record lands in its own directory (downloads are untrusted input:
never run anything from inside data_raw/). Verifies MD5 and skips files that
are already present and intact.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RECORDS = {
    "23605867": "elokriti_bpa_free_resin",       # computational (MD)
    "14576242": "aging_protocols_lodz",          # experimental, aging protocols
    "5569184": "dualcure_vs_lightcure",          # experimental, 9 composites
    "11186969": "dental_sealants",               # experimental, OriginPro
    "6372935": "ceramic_brackets",               # experimental, brackets
}
ROOT = Path(__file__).resolve().parent.parent / "data_raw"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    skip_big = "--skip-large" in sys.argv
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    for rec_id, name in RECORDS.items():
        out = ROOT / name
        out.mkdir(exist_ok=True)
        with urllib.request.urlopen(
            f"https://zenodo.org/api/records/{rec_id}", timeout=60
        ) as r:
            rec = json.loads(r.read().decode())

        meta = rec.get("metadata", {})
        (out / "_zenodo_metadata.json").write_text(json.dumps(rec, indent=2))

        for f in rec.get("files", []):
            key, size = f["key"], f.get("size", 0)
            want = (f.get("checksum") or "").replace("md5:", "")
            dest = out / key
            if size > 50e6 and skip_big:
                print(f"  skip (large): {key} {size/1e6:.0f}MB")
                continue
            if dest.exists() and want and md5(dest) == want:
                print(f"  ok (cached): {key}")
            else:
                print(f"  downloading: {key} ({size/1e6:.1f}MB)")
                urllib.request.urlretrieve(f["links"]["self"], dest)
                got = md5(dest)
                if want and got != want:
                    print(f"  !! MD5 MISMATCH {key}: want {want} got {got}")
                    return 1
            manifest.append(
                {
                    "record": rec_id,
                    "dir": name,
                    "file": key,
                    "bytes": size,
                    "license": (meta.get("license") or {}).get("id"),
                    "title": meta.get("title"),
                }
            )

    (ROOT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} files in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
