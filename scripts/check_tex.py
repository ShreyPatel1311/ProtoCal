#!/usr/bin/env python
"""Structural validation of the manuscript source.

No LaTeX toolchain is available locally and arXiv compiles server-side, so this
checks what can be checked statically: environment balance, brace balance,
every \\includegraphics target existing, and every \\cite key resolving in the
.bib file.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "paper"
tex = (P / "main.tex").read_text()
bib = (P / "refs.bib").read_text()
errs, warns = [], []

# environments
opens = re.findall(r"\\begin\{(\w+\*?)\}", tex)
closes = re.findall(r"\\end\{(\w+\*?)\}", tex)
for e in set(opens) | set(closes):
    if opens.count(e) != closes.count(e):
        errs.append(f"environment '{e}': {opens.count(e)} begin vs {closes.count(e)} end")

# braces (ignoring escaped)
depth = 0
for i, ch in enumerate(tex):
    if ch == "{" and (i == 0 or tex[i-1] != "\\"): depth += 1
    elif ch == "}" and (i == 0 or tex[i-1] != "\\"): depth -= 1
    if depth < 0: errs.append(f"unbalanced closing brace at char {i}"); break
if depth: errs.append(f"brace depth ends at {depth}, expected 0")

# figures
for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex):
    if not (P / g).exists(): errs.append(f"missing figure: {g}")

# citations
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for c in re.findall(r"\\cite[tp]?\{([^}]+)\}", tex):
    cited |= {k.strip() for k in c.split(",")}
for k in sorted(cited - keys): errs.append(f"\\cite{{{k}}} has no bib entry")
for k in sorted(keys - cited): warns.append(f"bib entry '{k}' never cited")

# labels / refs
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\ref\{([^}]+)\}", tex))
for r in sorted(refs - labels): errs.append(f"\\ref{{{r}}} has no label")

print(f"environments: {len(set(opens))} kinds, balanced" if not any("environment" in e for e in errs) else "")
print(f"figures: {len(re.findall(r'includegraphics', tex))} referenced, all present"
      if not any("figure" in e for e in errs) else "")
print(f"citations: {len(cited)} used, {len(keys)} defined")
print(f"labels: {len(labels)} defined, {len(refs)} referenced")
print(f"words (approx): {len(re.findall(r'[A-Za-z][A-Za-z-]+', re.sub(r'\\\\[a-zA-Z]+', ' ', tex)))}")
for w in warns: print("  warn:", w)
if errs:
    print("\nERRORS:"); [print("  !", e) for e in errs]; sys.exit(1)
print("\nstructural check PASSED")
