# External validation result — the pre-specified direction was refuted

**Protocol frozen in commit `557fde4`, before the validation study's tables were
read.** Pre-specified direction (H1): calibration by a single additive offset
from shared reference composites **would not reduce** prediction error.

**That prediction was wrong.** Calibration reduced error substantially, in every
arm, and the result is robust to all three ambiguities the pre-registration did
not anticipate. Reported as the pre-registration requires.

## Validation study

Alshabib A, Algamaiah H, Rojas-Rueda S, Jurado CA, Almansour A, AlOtaibi S.
*Influence of Solvent Challenge on the Hardness and Toughness of
Viscosity-Modified Composites.* Bioengineering 2026;13(7):726.
doi:[10.3390/bioengineering13070726](https://doi.org/10.3390/bioengineering13070726)

- **Independent.** King Saud University (Riyadh), Ponce Health Sciences
  University, Prince Sattam Bin Abdulaziz University. No author or affiliation
  overlap with Łódź, Rostock, Birmingham/Louvain or Manchester. Krüger,
  Randolph and Domarecka appear nowhere; Alrahlah appears only as a citation.
- **Method matches a training laboratory exactly.** Vickers 300 gf, 15 s dwell,
  nine indentations in a 3×3 pattern, n = 6 — the same load and dwell as
  Alrahlah. Storage 37 °C in the dark, distilled water or 75 % ethanol/water,
  1 and 30 days. All four protocols fall inside the shared aqueous/ethanol
  family.
- 4 products × 2 media × 2 times = 16 cells, from Tables 1 and 2.

## Reference / test split, from the frozen rule

| product | filler wt% | assignment |
|---|---|---|
| Charisma Bulk Flow ONE | 70 | **reference** (in corpus via Łódź) |
| Filtek Z350 XT | 78 | **reference** (in corpus as Filtek Ultimate) |
| Filtek Z350 XT Flowable | 65 | test — not in corpus |
| G-ænial Universal Flo | 69 | test — not in corpus |

## Primary result

| estimator | RMSE (log HV) |
|---|---|
| uncalibrated | 0.5193 |
| **calibrated** | **0.2431** |
| reference-mean baseline | 0.2762 |

**Primary comparison:** calibrated − uncalibrated = **−0.2773**. Calibration cut
error by 53 %. Both test products improve individually (−0.265 and −0.292), so
this is not one product driving it.

**All four arms agree:**

| arm | cal − uncal |
|---|---|
| primary (study filler, Z350 XT = Ultimate, Table 2) | −0.277 |
| sens A (Łódź's 65 wt% for Charisma Bulk Flow ONE) | −0.342 |
| sens B (Z350 XT treated as a *different* product → 1 reference, 3 test) | −0.261 |
| sens C (XTEF water 1 d = 38.3 from §3.1 rather than 38.9 from Table 2) | −0.276 |

## Why it helps: a large study-level offset

Mean (observed − predicted) across all 16 validation cells is **+0.4038 log,
i.e. +49.8 % on HV**. Observed hardness spans 29.2–68.1 HV; the uncalibrated
model predicts 17.6–53.8. The model is systematically low on this study, so a
single additive correction recovers most of the error.

Notably the offset is **larger over test cells (+0.491) than over reference
cells (+0.317)**, so the single offset *under-corrects* the test products —
consistent with the retrospective finding that the offset depends on the
material rather than only on the study.

## Three qualifications that bound the claim

**1. The reported interval is not a usable 95 % CI.** With two test products the
cluster bootstrap admits only three distinct resample multisets, and takes
exactly three values: −0.2650, −0.2762, −0.2919. The printed interval
[−0.2919, −0.2650] *is* that range. It must not be presented as a 95 %
confidence interval. The per-product breakdown is the honest summary, and it
happens to agree in direction.

**2. Calibration does not beat the trivial baseline.** Calibrated −
reference-mean = **−0.0172**, interval [−0.202, +0.200] — indistinguishable. And
the two test products disagree about which is better:

| test product | uncalibrated | calibrated | reference-mean |
|---|---|---|---|
| Filtek Z350 XT Flowable | 0.4414 | **0.1764** | 0.3788 |
| G-ænial Universal Flo | 0.5869 | 0.2951 | **0.0952** |

For G-ænial Universal Flo, simply predicting the mean hardness of the two
reference products beats the calibrated model threefold. So what this
validation demonstrates is that **a local anchor is necessary**, not that the
composition-and-protocol model is good. The model's own structure earns nothing
over "assume the new product resembles the references".

**3. The uncalibrated model is worse than the trivial baseline.** Uncalibrated −
reference-mean = +0.2601, interval [+0.063, +0.492]. Without an anchor the model
is not merely imprecise on an unseen study — it is worse than ignoring the model
entirely.

## What this does and does not change

**Changed.** The earlier claim — that within a shared protocol family
cross-laboratory prediction works *without* offset calibration, and calibration
usually makes it worse — does not generalise to a genuinely new laboratory. On
this independent study calibration was necessary and clearly beneficial. The
retrospective result appears to have been specific to laboratories already
inside the corpus, whose offsets the pooled fit had partly absorbed.

**Not changed.** The additive offset is still mis-specified: it under-corrects
the test products relative to the references (+0.491 vs +0.317), exactly the
material dependence found retrospectively.

**Still unsupported.** Any claim about *how many* reference products are needed.
Two references here, one in sensitivity B. The subset results (Charisma alone
0.221, Filtek alone 0.267, both 0.243) are descriptive only.

## Honest statement of the revised position

Offset calibration against shared composites is **necessary** for transfer to a
new laboratory and **not sufficient** to make the model useful: after
calibration it only matches a baseline that uses the same reference
measurements and no model at all. The defensible claim for publication is about
*when* calibration is required and what it does not buy, with the negative
retrospective result and this positive external result both reported.
