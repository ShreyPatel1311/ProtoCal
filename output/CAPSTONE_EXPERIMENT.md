# Capstone validation experiment — protocol and pre-registered analysis

**Status: design only. This is a wet-laboratory experiment; it has not been run.**
Everything is specified so it can be handed to the participating laboratories
and executed without further analysis decisions. The analysis plan, the
material panel and the reference/test split are fixed in advance, so the test
is confirmatory rather than exploratory.

**Revision note.** This replaces an earlier single-laboratory design that had
two defects: it computed power for a 12-product panel but specified the primary
slope test on only the 8 held-out products, and a single laboratory joining the
retrospective corpus cannot separate a laboratory effect from the protocol and
measurement differences that accompany it. Both are corrected below.

---

## 1. Why this experiment, and what it must settle

**The effect to confirm, and its newly-discovered fragility.** Filler loading
appears to protect against organic-solvent softening. The paired
within-material slope of `log(HV_ethanol) − log(HV_water)` on filler wt% is
positive in both retrospective laboratories — but **its significance depends on
how filler content is defined**:

| filler definition | Randolph HV (n=14) | Randolph FS | Randolph FM | Alrahlah HV (n=7) |
|---|---|---|---|---|
| manufacturer datasheet | +0.0244 **(p=0.012)** | +0.0370 **(0.032)** | +0.0409 **(0.0095)** | +0.0267 **(0.014)** |
| measured, TGA `W_i` | +0.0176 (0.079) | +0.0326 (0.055) | +0.0411 **(0.0058)** | +0.0249 **(0.0004)** |
| measured, acetone-insoluble `W_total` | +0.0195 (0.099) | +0.0277 (0.181) | +0.0320 (0.098) | +0.0265 **(0.0066)** |

Four of four tests reach significance on manufacturer figures, two of four on
TGA-measured inorganic content, one of four on `W_total`. Randolph's Vickers
result — the one that looked like a clean independent replication — **does not
survive the switch to measured filler** (p = 0.079).

The cause is that manufacturer figures and TGA measurements disagree by up to
12 wt%, and disagree most for exactly the prepolymerised-filler products that
serve as cross-laboratory anchors:

| product | manufacturer | `W_i` (TGA) | difference |
|---|---|---|---|
| Tetric EvoCeram | 82.5 | 70.5 | **−12.0** |
| Kalore | 82.0 | 71.1 | **−10.9** |
| Clearfil Majesty ES Flow | 75.0 | 67.2 | −7.8 |
| Filtek Ultimate | 78.5 | 73.0 | −5.5 |
| Venus Diamond Flow | 65.0 | 60.1 | −4.9 |

**Consequence for this experiment: filler content must be measured in-house by
TGA, and the definition must be declared in advance.** This is no longer
optional. It is the single largest uncontrolled input to the whole analysis.

**The methodological problem to resolve.** The retrospective analysis assumes a
single additive offset per laboratory. That assumption fails on its own data:
estimating one laboratory's offset from one shared product at a time gives
−0.740…+0.204 (Alrahlah) and −0.706…+0.544 (Randolph), against a
within-laboratory repeatability of sd(log HV) = **0.022**. The estimate moves by
more than twenty times the measurement noise depending on which product it
comes from, so the laboratory effect interacts with the material.

Worse, whether calibration helps at all depends on the filler definition too.
On the shared aqueous/ethanol subcorpus, the median anchor subset is worse than
no calibration for every laboratory under manufacturer figures; under measured
`W_i` it becomes *better* for Alrahlah while staying worse for Lodz and
Randolph. So the honest retrospective conclusion is not "calibration hurts" but
**the outcome depends jointly on which reference products are used and on how
filler content is defined** — which is precisely why a controlled experiment is
needed.

**Why two laboratories.** The retrospective corpora differ simultaneously in
indenter load (1 N to 1000 gf), dwell (5–30 s), specimen preparation,
polishing, whether specimens were dried before testing, and which protocol
family was run. No analysis of those data can isolate a laboratory effect. Two
laboratories measuring an **overlapping panel under matched procedures** can,
because everything except laboratory identity is held fixed by construction.

Three questions, in priority order:

1. **Q1 (confirmatory).** Is the filler × solvent slope positive under a
   pre-declared, measured filler definition?
2. **Q2 (decisive).** Is a single additive per-laboratory offset sufficient, or
   does the offset depend on the material?
3. **Q3 (secondary).** Does calibrating from pre-chosen reference products
   predict held-out products better than not calibrating?

---

## 2. Design

**Two laboratories, A and B**, measuring the **same 18 products** under matched
written procedures. One material factor × one two-level protocol factor.
Nothing else varies between laboratories by design; every deviation is logged.

**Conditioning.** Both arms share a 24 h / 37 °C water pre-store so post-cure is
complete and equal before the arms diverge.

| arm | medium | duration | temperature |
|---|---|---|---|
| control | distilled water | 7 days | 37 °C |
| challenge | 75 % v/v ethanol / 25 % water | 7 days | 37 °C |

Seven days at 37 °C in 75/25 ethanol/water matches Randolph exactly and is the
closest common ground with Alrahlah, whose ethanol exposures were 1 and 30 days.
Media are not refreshed, matching both sources. Record actual immersion start
and end times.

**Measurement, identical at both laboratories.** Vickers, **300 gf, 15 s
dwell**, three indentations per specimen averaged, measured **within 15 minutes
of removal from the medium, blotted but not dried**.

> Two of these are not free choices. Specimens must **not** be dried before
> testing: Krüger dried for 24 h under vacuum and is the one corpus in which
> solvent effects essentially vanish (protocol ≈ 2 % of variance; its leading
> principal component is indistinguishable from its own permutation null,
> p = 0.22). Drying reverses plasticisation, which is the effect being measured.
> The load is fixed at 300 gf so hardness is a single-load measurement
> throughout.

**Specimens.** Discs 6 mm diameter × 2 mm, silicone mould, cured per
manufacturer instructions with irradiance recorded and verified at both sites.
Oxygen-inhibition layer removed identically; **all specimens polished to the
same declared final grit** at both laboratories.

**n = 5 specimens per (laboratory × product × arm) cell.** Measurement noise
contributes 0.022/√5 ≈ 0.010 to a cell mean, against a between-material
residual scatter of 0.244 — so n = 5 is already far inside the regime where
between-material variation dominates, and raising it buys nothing for Q1.

**Totals.** 18 products × 2 arms × 5 specimens × 2 laboratories = **360
specimens**, 180 per laboratory.

*Reduced-scope fallback.* If 360 is infeasible, Laboratory B measures the
**control arm only** (18 × 5 = 90 specimens; 270 total). This preserves Q2 and
Q3 in full, since both use control-arm data. It gives up the independent
replication of Q1 in a second laboratory, which is a real loss and should be
stated as such rather than glossed.

**Randomisation and blinding.** Randomise specimen-to-arm assignment and
indentation order. Operators read diagonals blind to product and arm;
specimens are labelled by code. Laboratories do not exchange results until
both have locked their data.

---

## 3. Material panel, filler measurement, and the pre-registered split

**Panel size is driven by power at the conservative effect size.** Using the
*measured*-filler Randolph HV slope of **+0.0176** (not the manufacturer-figure
+0.0263) and the pooled residual sd 0.244, at α = 0.05 one-sided:

| n test products | filler sd 8 wt% | 10 | 12 | 14 |
|---|---|---|---|---|
| 8 | 0.39 | 0.52 | 0.65 | 0.76 |
| 12 | 0.55 | 0.72 | 0.85 | 0.93 |
| **14** | 0.62 | 0.79 | **0.90** | 0.96 |
| 16 | 0.68 | 0.84 | 0.94 | 0.98 |

**Panel: 18 products = 4 reference + 14 test, spanning filler sd ≥ 12 wt%.**
The primary slope test runs on the **14 test products only**, giving **90 %
power** — so the power calculation and the primary analysis now use the same n,
which the previous version did not. If the true effect is the
manufacturer-figure +0.0263, power at n = 14 and sd 12 is 1.00.

Spread matters as much as count: 14 products at sd 12 wt% beats 16 at sd 8.
Span roughly 52–89 wt%, with genuine coverage at both ends rather than a
cluster in the middle.

**Filler content is measured in-house by TGA for all 18 products**, three
replicates each, reported with standard deviations. `W_i` (total inorganic
content, ash residue) is the **pre-declared primary definition**. Manufacturer
figures and, if centrifugation is available, `W_total` are reported as
secondary. Rationale: `W_i` is the quantity that is reproducibly measurable
across laboratories, whereas a datasheet figure is not a measurement at all and
the retrospective analysis shows the two are not interchangeable.

**Pre-registered split — fix before any specimen is made.**

- **Reference products — 4.** Used only to estimate the laboratory offset.
  Draw from products already in the retrospective corpus so the new
  laboratories join the existing graph: *Filtek Ultimate / Supreme XTE*,
  *Admira Fusion*, *Venus Diamond*, *Grandio Flow*, *ELS*, *Filtek Silorane*.
- **Test products — 14.** Never used for calibration. Scored once.

Four references is not arbitrary. In the retrospective sweep error fell
monotonically with reference count and single references were unreliable
(Randolph 0.565 ± 0.158 at one, 0.386 at five; Alrahlah 0.484 ± 0.138 at one,
0.276 at three), with the gain flattening by about four.

Two exclusions, both evidence-based:

- **Do not use an extreme-composition product as a reference.** GrandioSO
  (89 wt%, highest in the corpus) was the *worst* single reference for
  Alrahlah — worse than no calibration at all (0.675 vs 0.495). References
  belong in the interior of the filler range. If GrandioSO is in the panel, it
  goes in the test set.
- **Do not use Tetric EvoCeram or Kalore as references.** They carry the
  largest manufacturer-vs-measured discrepancies (−12.0 and −10.9 wt%). With
  in-house TGA this is less dangerous than it was, but a reference product
  whose composition is contested is the wrong place to spend the risk.

---

## 4. Analysis plan, fixed in advance

**Q1 — the slope.** Primary endpoint, on the **14 test products**, per
laboratory and pooled:

```
Δ_i = log(HV_ethanol,i) − log(HV_water,i)          cell means per product i
Δ_i = β₀ + β₁ · W_i,i + ε_i
```

Primary test: `β₁ > 0`, one-sided α = 0.05, filler as in-house TGA `W_i`.
Report β₁ with a bootstrap 95 % CI, per laboratory and pooled, alongside the
retrospective measured-filler estimates (+0.0176 Randolph, +0.0249 Alrahlah).
Material identity cancels in Δ, so this is immune to between-material baseline
differences — which is why it is primary rather than a model-based comparison.

**Pre-specified sensitivity, not post-hoc:** repeat Q1 with manufacturer figures
and with `W_total`. The retrospective analysis found the conclusion changes
across these, so all three are reported together and the primary remains `W_i`
whatever they show.

**Q2 — is one offset per laboratory enough?** The decisive test. Control arm
only, all 18 products, both laboratories:

```
H₀ (additive):      log HV = f(product) + b_lab
H₁ (interaction):   log HV = f(product) + b_lab + c_lab · W_i
```

Nested F / likelihood-ratio test. Report per-product residuals under H₀ and
their sd against the measured within-laboratory repeatability.

**Pre-specified interpretation.** If H₀ is not rejected and the residual sd is
within about 3× within-laboratory repeatability, single-offset harmonisation is
supported and Q3 is meaningful. If H₀ **is** rejected, the single-offset method
is mis-specified, *that is the paper's finding*, and Q3 must be reported as
describing a mis-specified model rather than a working method. With 18 shared
products this test has real power, unlike the 1–5 shared products available
retrospectively.

**Q3 — does calibration help?** Estimate one additive offset from the 4
reference products' control-arm cells in Laboratory B relative to Laboratory A.
Apply to predictions for the 14 test products. Report all four of:

| | |
|---|---|
| uncalibrated | model with offset 0 |
| calibrated | offset from the 4 reference products |
| reference-mean baseline | predict the mean of the reference products' measured values |
| oracle offset | the offset minimising test error (unachievable upper bound) |

The reference-mean baseline is **mandatory**. Retrospectively it beat every
calibrated variant for Lodz (0.416 vs best 0.528); omitting it would have
hidden that. Also report the distribution over all 15 non-empty subsets of the
4 reference products, not just the mean — the retrospective spread across
reference choices exceeded the mean effect.

**Hardness-loss endpoint.** If a hardness-*loss* claim is made, Δ relative to
the matched water control is a **separate, independently reported outcome** with
its own test, not a re-description of Q1. Absolute HV and Δ have different noise
structure: Δ cancels material baseline but doubles measurement variance.

**Stopping and deviations.** No interim analysis; n fixed at 5 per cell. Record
every deviation. Specimens lost to handling are reported, never silently
replaced.

---

## 5. What this experiment cannot do

- **Two laboratories is two.** It can test whether a single additive offset is
  adequate *between this pair*. It cannot establish additivity in general,
  though it is the first clean test of it in this project.
- **It fixes one protocol contrast.** Water vs 75 % ethanol at 7 days says
  nothing about alkaline or thermocycling challenges, where the retrospective
  corpora disagree in sign on the filler effect (Alrahlah −0.024 vs Lodz
  +0.011). Those act by a different mechanism and need their own experiment.
- **It does not resolve the load question.** All specimens are measured at
  300 gf by design. Whether between-laboratory offsets are driven by indenter
  load needs one panel measured at several loads — worthwhile, separate. The
  retrospective data do not support a load explanation (offset ~ log load gives
  r² = 0.04, p = 0.79 across the four true-Vickers laboratories).
- **It cannot rescue the retrospective calibration claim.** If Q2 rejects
  additivity, the retrospective harmonisation numbers are describing a
  mis-specified model and should be reported as a negative methodological
  result.

## 6. Estimated effort

360 specimens across two laboratories (180 each), or 270 under the reduced
scope. Per laboratory: roughly 2–3 specimen-preparation days, 7 days
conditioning with both arms in parallel, 2 days of indentation. In-house TGA on
18 products adds about 1–2 days. Feasible in four weeks of laboratory time per
site including the pre-registration freeze, with the two sites running in
parallel.
