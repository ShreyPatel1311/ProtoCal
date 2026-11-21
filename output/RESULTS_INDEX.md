# Results index

Every claim → its number → the file holding it → the script that produced it.
Regenerate anything with `python scripts/<script>`. Raw data is not committed;
`scripts/download_data.py` fetches it (all CC-BY-4.0).

## Status legend
✅ verified and reportable · ⚠️ underpowered, report with caveat · ❌ withdrawn

---

## 1. Corpus

| | |
|---|---|
| **Claim** | 1,802 specimen measurements, 12 materials, 4 protocols, 5 properties, 238/240 cells (99.2% complete) |
| **Status** | ✅ |
| **File** | `output/aging_long.csv` (the tidy table, 325KB) |
| **Script** | `scripts/audit_data.py` |
| **Source** | Zenodo 10.5281/zenodo.14576242 (Szczesio-Wlodarczyk et al., *J Funct Biomater* 16(2):49) |
| **Note** | 12 not 13 — `Grandio SO Heavy Flow 2` is a filename artifact, merged in `protocal/stats.py` |

## 2. Protocol is a first-order effect ✅

Variance components + replicated two-way F-test, log scale, Type-II SS.

| Property | Material | Protocol | Inter. | Resid. | F_inter | p |
|---|---|---|---|---|---|---|
| Flexural strength | 42.6% | 35.6% | 14.1% | 7.7% | 13.79 | 4.3e-42 |
| Diametral tensile | 34.2% | 21.3% | 19.5% | 24.9% | 7.99 | 3.0e-27 |
| Vickers hardness | 49.3% | 21.2% | 15.4% | 14.1% | 10.58 | 1.1e-35 |
| Fracture toughness | 59.3% | 10.4% | 8.2% | 22.2% | 3.21 | 1.2e-7 |
| Flexural modulus | 79.4% | 2.2% | 5.0% | 13.4% | 3.55 | 5.5e-9 |

**File** `output/corrected_stats.json` · **Script** `protocal/stats.py` → inline runner
**Figure** `paper/figures/fig2_variance.pdf`
**Interaction is significant for ALL five properties** — the earlier "only flexural strength" claim was an artifact of an invalid ratio (see `CORRECTIONS.md` §C3).

## 3. The corpus has ~one protocol degree of freedom ✅ ← headline

| | |
|---|---|
| **Claim** | 99.7% of protocol variation on the first principal component; singular values 42.02, 1.80, 0.87, 0.82 |
| **Supporting** | unaged-vs-aged mean \|Δlog\| = 0.333–0.347; aged-vs-aged = 0.118–0.134 |
| **File** | `output/corrected_stats.json` → `protocol_dof` |
| **Script** | `protocal/stats.py::protocol_dof` |
| **Figure** | `paper/figures/fig3_protocol_geometry.pdf` |

## 4. Nine model families, three regimes ✅ (as a sweep) / ⚠️ (individual contrasts)

**Held-out cells** — random forest 0.645, hist GBM 0.645, ProtoCal one-hot 0.684,
additive 0.685, ProtoCal comp 0.695, GP 0.763, ridge 0.851, grand mean 1.016

**Leave-one-material-out** — ProtoCal comp 0.874, random forest 0.896, GP 0.927,
hist GBM 0.938, ProtoCal one-hot 0.993, grand mean 1.060, additive 1.473

**Leave-one-protocol-out** — random forest 0.770, hist GBM 0.784, additive 0.808,
ProtoCal one-hot 0.814, ProtoCal comp 0.844, GP 0.894, grand mean 1.101

**File** `output/model_comparison.json` (per-fold values included) · **Script** `scripts/model_comparison.py`

### Paired Nadeau–Bengio tests — only ONE is significant
| Contrast | diff | p | verdict |
|---|---|---|---|
| **additive vs grand mean (unseen material)** | **+0.412** | **0.016** | ✅ dummy coding is significantly WORSE than the mean |
| composition vs one-hot encoding | −0.118 | 0.345 | ⚠️ directional, 8/12 folds |
| best model vs grand mean | −0.186 | 0.154 | ⚠️ no model beats the mean |
| best neural vs best tree | −0.022 | 0.828 | ⚠️ dead tie |

**Power**: 12 folds, mean sd 0.395 → **MDE = 0.363**. All nulls are underpowered, not equivalences.

## 5. Joint identifiability, not information ✅

| | |
|---|---|
| **Claim** | oracle BLUP recovers `b_s` at r=0.945 (sparse) / 0.978 / 0.996; joint fit 0.352 / 0.506 / 0.645 |
| **Files** | `output/density_sweep.json` (joint), oracle printed by script |
| **Scripts** | `scripts/oracle_blup.py`, `scripts/density_sweep.py` |
| **Figure** | `paper/figures/fig6_identifiability.pdf` |

## 6. Mechanistic interpretability ✅

| Probe | Result |
|---|---|
| Fidelity | learned `g_proto` vs empirical protocol effects: slopes 0.89–1.09, RMSE 0.006–0.116 sd |
| Dimensionality | learned `[0.979, 0.019, 0.003]` vs empirical `[0.980, 0.017, 0.003]` |
| Attribution | mean abs: `f_shared` 0.857, `g_proto` 0.666 |
| Gauge check | `|g(p_ref)|` = 0.00e+00; `|interaction(x,p_ref)|` = 1.09e-08 |

**File** `output/interpretability.json` · **Script** `scripts/interpretability.py`
**Figure** `paper/figures/fig5_interpretability.pdf`

## 7. Distributional assumption ✅ (negative)

Gaussian cannot be rejected: Shapiro rejects in only 4–21% of cells, median skew ≈ 0,
Weibull beats Normal in 50/96 strength cells (coin flip), median Weibull k = 8–18.
**Files** `output/distribution_check.json`, `output/distribution_per_cell.csv` ·
**Script** `scripts/distribution_check.py`

## 8. Withdrawn ❌

| Claim | Why |
|---|---|
| "44 unexplained outliers" | Expected MAD-flags under exact normality at n=6–9 is **51.1**; we saw 44. Data are cleaner than Gaussian. Published about another group's dataset — withdrawn outright. |
| "b_s is information-limited" | Artifact of two estimator bugs. Oracle gets 0.945. See `CORRECTIONS.md` §C1. |
| "interaction only for flexural strength" | Invalid statistic. Significant for all five. See §C3. |
| "cross-study harmonisation demonstrated" | Never tested. One laboratory. |

Full detail: `output/CORRECTIONS.md`

## 9. Superseded / historical

`output/iris_run1_idea.md`, `iris_run1_refined.md` — the original IRIS-generated proposal
`output/DATA_AUDIT_FINDINGS.md` — first audit; **read only alongside CORRECTIONS.md**
`output/interaction_experiment.json`, `capacity_sweep.json`, `smoke_test.json`,
`data_audit.json` — earlier runs, several superseded by §2 and §4
`output/dataset_availability.md` — licence audit of all five candidate datasets

## Manuscript

`paper/main.tex` (1,975 words), `paper/refs.bib`, `paper/figures/*.pdf` (6)
Validate with `python scripts/check_tex.py`. No LaTeX toolchain locally — unbuilt.
