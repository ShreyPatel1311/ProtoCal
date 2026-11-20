#!/usr/bin/env python
"""Publication figures for the ProtoCal manuscript."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "font.family": "serif"})
C = {"mat": "#4C72B0", "prot": "#DD8452", "int": "#55A868", "res": "#C44E52",
     "model": "#4C72B0", "base": "#8C8C8C", "acc": "#937860"}
SHORT = {"flexural_strength_MPa": "Flex. strength", "flexural_modulus_MPa": "Flex. modulus",
         "diametral_tensile_MPa": "Diam. tensile", "vickers_hardness": "Vickers",
         "fracture_toughness": "Fracture tough."}
stats_ = json.loads((ROOT / "output" / "corrected_stats.json").read_text())
exp = json.loads((ROOT / "output" / "experiments.json").read_text())
interp = json.loads((ROOT / "output" / "interpretability.json").read_text())

# ---------------- Fig 1: architecture ----------------
fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 4)
def box(x, y, w, h, t, c, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                 fc=c, ec="#333", lw=0.9, alpha=0.9))
    ax.text(x + w/2, y + h/2, t, ha="center", va="center", fontsize=fs, color="white", weight="bold")
def arrow(x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, lw=1.1, color="#333"))
box(0.1, 2.75, 1.5, 0.75, "composition\n$x_i$", "#5B7C99")
box(0.1, 1.5, 1.5, 0.75, "protocol\n$p_s$", "#B07C5B")
box(0.1, 0.25, 1.5, 0.75, "study $s$", "#7C5B99")
box(2.5, 2.75, 2.0, 0.75, "$f_{shared}(x)$\nMLP", C["mat"])
box(2.5, 1.5, 2.0, 0.75, "$g_{proto}(p)-g_{proto}(p_{ref})$", C["prot"], fs=7.2)
box(2.5, 0.25, 2.0, 0.75, "$b_s\\sim\\mathcal{N}(0,\\tau^2)$", C["res"])
box(5.4, 1.5, 1.2, 2.0, "$\\Sigma$", "#444", fs=15)
box(7.4, 1.95, 2.3, 0.75, "$\\hat{y}$ as reported", C["acc"])
box(7.4, 0.7, 2.3, 0.75, "$f_{shared}(x)$:\nharmonised", "#55A868", fs=7.5)
for y in (3.12, 1.87, 0.62): arrow(1.6, y, 2.5, y)
arrow(4.5, 3.12, 5.4, 2.8); arrow(4.5, 1.87, 5.4, 2.5); arrow(4.5, 0.62, 5.4, 2.2)
arrow(6.6, 2.5, 7.4, 2.32); arrow(4.5, 3.12, 7.4, 1.07, style="-|>")
ax.text(6.0, 0.35, "gauge fix: $g_{proto}(p_{ref})\\equiv 0$\n$\\Rightarrow f_{shared}$ IS the\nreference-protocol prediction",
        fontsize=7.2, ha="center", style="italic", color="#333")
fig.savefig(FIG / "fig1_architecture.pdf"); plt.close(fig)

# ---------------- Fig 2: variance components ----------------
rows = stats_["anova"]; labels = [SHORT[r["property"]] for r in rows]
comp = np.array([[r["vc_material_pct"], r["vc_protocol_pct"],
                  r["vc_interaction_pct"], r["vc_residual_pct"]] for r in rows])
order = np.argsort(-comp[:, 1]); comp, labels = comp[order], [labels[i] for i in order]
rows_s = [rows[i] for i in order]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2, 1]})
left = np.zeros(len(labels))
for j, (name, col) in enumerate(zip(["material", "protocol", "interaction", "residual"],
                                    [C["mat"], C["prot"], C["int"], C["res"]])):
    a1.barh(labels, comp[:, j], left=left, color=col, label=name, height=0.62)
    for i, v in enumerate(comp[:, j]):
        if v > 5: a1.text(left[i] + v/2, i, f"{v:.0f}", ha="center", va="center",
                          fontsize=7.5, color="white", weight="bold")
    left += comp[:, j]
a1.set_xlabel("variance component (%)"); a1.set_xlim(0, 100)
a1.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.22), frameon=False)
a1.grid(axis="y", alpha=0)
f = [r["F_interaction"] for r in rows_s]
a2.barh(labels, f, color=C["int"], height=0.62)
for i, r in enumerate(rows_s):
    a2.text(f[i] + 0.3, i, f"p={r['p_interaction']:.0e}", va="center", fontsize=6.5)
a2.set_xlabel("interaction $F$"); a2.set_xlim(0, max(f) * 1.75); a2.set_yticklabels([])
a2.grid(axis="y", alpha=0)
fig.savefig(FIG / "fig2_variance.pdf"); plt.close(fig)

# ---------------- Fig 3: protocol geometry ----------------
emp = interp["dimensionality"]["empirical_var_share"]
lea = interp["dimensionality"]["learned_var_share"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
idx = np.arange(1, len(emp) + 1); w = 0.38
a1.bar(idx - w/2, np.array(emp) * 100, w, label="empirical", color=C["prot"])
a1.bar(idx + w/2, np.array(lea) * 100, w, label="learned $g_{proto}$", color=C["mat"])
a1.set_xlabel("protocol principal component"); a1.set_ylabel("variance explained (%)")
a1.set_xticks(idx); a1.legend(frameon=False, fontsize=7.5)
a1.text(0.5, 0.55, f"PC1 = {emp[0]*100:.1f}%", transform=a1.transAxes, fontsize=9,
        weight="bold", color=C["prot"])
a1.text(0.5, 0.44, "one effective\nprotocol d.o.f.", transform=a1.transAxes, fontsize=7.5, style="italic")
d = stats_["protocol_dof"]["pairwise_mean_abs_log_diff"]
pairs = list(d.items())
vals = [v for _, v in pairs]
cols = [C["res"] if v > 0.25 else C["base"] for v in vals]
a2.barh(range(len(vals)), vals, color=cols, height=0.6)
a2.set_yticks(range(len(vals)))
a2.set_yticklabels(["unaged vs aged" if v > 0.25 else "aged vs aged" for v in vals], fontsize=7.5)
a2.set_xlabel("mean $|\\Delta \\log$ property$|$ between protocols")
a2.grid(axis="y", alpha=0)
fig.savefig(FIG / "fig3_protocol_geometry.pdf"); plt.close(fig)

# ---------------- Fig 4: LOPO ----------------
lopo = exp["lopo"]
names = ["NaOH\n7d 60C", "water 24h 37C\n(REFERENCE)", "water 5d\n+NaOH", "thermocycle\n+NaOH"]
x = np.arange(len(lopo)); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 2.9))
for j, (k, lab, col) in enumerate([("grand_mean", "grand mean", C["base"]),
                                   ("additive_two_way", "additive two-way", C["acc"]),
                                   ("protocal", "ProtoCal", C["model"])]):
    v = [r[k] for r in lopo]
    ax.bar(x + (j - 1) * w, v, w, label=lab, color=col)
    for i, vv in enumerate(v):
        ax.text(x[i] + (j - 1) * w, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
ax.set_ylabel("held-out RMSE (sd units)")
ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right", bbox_to_anchor=(1.0, 1.02))
ax.set_ylim(0, 1.95); ax.grid(axis="x", alpha=0)
ax.annotate("continuous protocol features\nbeat dummy coding", xy=(0.26, 0.70),
            xytext=(0.30, 1.30), fontsize=7, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.8, connectionstyle="arc3,rad=-0.15"))
ax.annotate("extrapolation fails:\nreference is the extreme", xy=(1.26, 1.44),
            xytext=(1.70, 1.80), fontsize=7, ha="center",
            arrowprops=dict(arrowstyle="->", lw=0.8, connectionstyle="arc3,rad=0.15"))
fig.savefig(FIG / "fig4_lopo.pdf"); plt.close(fig)

# ---------------- Fig 5: interpretability ----------------
fid = interp["fidelity"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9))
for prop in fid:
    a1.scatter(fid[prop]["empirical"], fid[prop]["learned"], s=26, label=SHORT[prop], alpha=0.85)
lim = [-1.5, 0.25]
a1.plot(lim, lim, "k--", lw=0.8, zorder=0)
a1.set_xlim(lim); a1.set_ylim(lim)
a1.set_xlabel("empirical protocol effect (sd)"); a1.set_ylabel("learned $g_{proto}$ (sd)")
a1.legend(fontsize=6.3, frameon=False, loc="upper left")
sl = np.mean([fid[p]["calibration_slope"] for p in fid])
a1.text(0.97, 0.06, f"mean slope {sl:.2f}", transform=a1.transAxes, ha="right", fontsize=8, weight="bold")
cur = interp["response_curves"]["(T-37)/23"]
for prop in PROPS_ORDER if (PROPS_ORDER := list(cur["effect"].keys())) else []:
    a2.plot(cur["grid"], cur["effect"][prop], lw=1.4, label=SHORT[prop])
a2.axvline(0, color="k", ls=":", lw=0.8)
a2.text(0.02, -0.02, "reference", rotation=90, fontsize=6.5, va="top")
a2.set_xlabel("normalised temperature $(T-37)/23$"); a2.set_ylabel("learned effect (sd)")
a2.legend(fontsize=6.3, frameon=False)
fig.savefig(FIG / "fig5_interpretability.pdf"); plt.close(fig)

# ---------------- Fig 6: identifiability ----------------
fig, ax = plt.subplots(figsize=(4.2, 2.9))
settings = ["sparse\n(6.2 obs/cell)", "full report\n(13.5)", "4x rows\n(59.4)"]
# oracle: scripts/oracle_blup.py; joint: scripts/density_sweep.py (fixed estimator)
oracle = [0.945, 0.978, 0.996]; joint = [0.352, 0.506, 0.645]
x = np.arange(3); w = 0.35
ax.bar(x - w/2, oracle, w, label="oracle (true $f$, $g$ given)", color=C["int"])
ax.bar(x + w/2, joint, w, label="joint fit", color=C["res"])
for i in range(3):
    ax.text(x[i] - w/2, oracle[i] + 0.015, f"{oracle[i]:.2f}", ha="center", fontsize=7)
    ax.text(x[i] + w/2, joint[i] + 0.015, f"{joint[i]:.2f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=7.5)
ax.set_ylabel("$b_s$ recovery (Pearson $r$)"); ax.set_ylim(0, 1.12)
ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.grid(axis="x", alpha=0)
ax.text(0.5, 0.40, "the gap is joint identifiability,\nnot information content",
        transform=ax.transAxes, ha="center", fontsize=7.2, style="italic")
fig.savefig(FIG / "fig6_identifiability.pdf"); plt.close(fig)

print("figures written:")
for f in sorted(FIG.glob("*.pdf")): print(f"  {f.name}  {f.stat().st_size/1024:.0f}KB")
