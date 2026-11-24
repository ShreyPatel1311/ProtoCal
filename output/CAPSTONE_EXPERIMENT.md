# Capstone validation experiment — protocol and pre-registered analysis

**Status: design only. This is a wet-laboratory experiment; it has not been run.**
Everything below is specified so it can be handed to a supervising laboratory
and executed without further analysis decisions. The analysis plan is fixed in
advance, including the material split, so the test is confirmatory rather than
exploratory.

---

## 1. Why this experiment, and what it has to settle

Two results from the five-laboratory retrospective corpus motivate it, and one
of them is a problem rather than a finding.

**The effect to confirm.** Filler loading protects against organic-solvent
softening. Two laboratories agree on the paired within-material slope of
`log(HV_ethanol) − log(HV_water)` on filler wt%:

| laboratory | n | slope (log-HV per wt%) | p |
|---|---|---|---|
| Randolph | 14 | +0.0244 | 0.012 |
| Alrahlah | 7 | +0.0301 | 0.0017 |
| pooled | 21 | **+0.0263** | — |

**The problem to resolve.** The retrospective analysis has to assume a single
additive offset per laboratory. That assumption does not survive its own data.
Estimating one laboratory's offset from one shared product at a time gives:

| laboratory | single-anchor offset range | sd |
|---|---|---|
| Alrahlah | −0.740 … +0.204 | 0.518 |
| Randolph | −0.706 … +0.544 | 0.494 |
| Lodz | −1.121 … −0.992 | 0.065 |

Within-laboratory repeatability is sd(log HV) = **0.022**. So for two of three
laboratories the offset estimate moves by more than *twenty times* the
measurement noise depending on which product it is estimated from. A
per-laboratory additive constant is therefore mis-specified: the laboratory
effect interacts with the material. This is also why calibration is unreliable
— for Randolph, 71% of anchor subsets make prediction *worse* than not
calibrating.

The retrospective corpus cannot separate these explanations, because every
laboratory differs simultaneously in indenter load (1 N to 1000 gf), dwell
(5–30 s), specimen preparation, polishing, and whether specimens were dried
before testing. **A single controlled experiment can.**

Three questions, in priority order:

1. **Q1 (confirmatory).** Is the filler × solvent slope positive, and is it
   consistent with +0.0263?
2. **Q2 (the methodological question).** Is a single additive per-laboratory
   offset sufficient, or does the offset depend on the material? This is the
   one the retrospective data cannot answer and the one the method depends on.
3. **Q3 (secondary).** Does calibrating from pre-chosen reference products
   predict held-out products better than not calibrating?

---

## 2. Design

**Factors.** One material factor (composite) × one protocol factor with two
levels. Nothing else varies.

**Conditioning.** Both arms share a 24 h / 37 °C water pre-store so that
post-cure is complete and equal before the arms diverge.

| arm | medium | duration | temperature |
|---|---|---|---|
| control | distilled water | 7 days | 37 °C |
| challenge | 75 % v/v ethanol / 25 % water | 7 days | 37 °C |

Seven days at 37 °C in 75/25 ethanol/water matches Randolph exactly and is the
closest common ground with Alrahlah, whose ethanol exposures were 1 and 30 days
at 37 °C. **Fixing the duration at 7 days is a deliberate choice to make the
new data commensurable with Randolph rather than with Alrahlah**; Alrahlah's
1-day and 30-day points bracket it, so the comparison stays interpretable
either way. Media are not refreshed, matching both sources. Record the actual
immersion start and end times.

**Measurement.** Vickers, **300 gf, 15 s dwell**, three indentations per
specimen averaged, measured **within 15 minutes of removal from the medium,
blotted but not dried**.

> Two of these are not free choices. Specimens must **not** be dried before
> testing: Krüger dried for 24 h under vacuum and is the one corpus in which
> solvent effects essentially vanish (protocol accounts for ~2 % of variance,
> and its leading principal component is indistinguishable from its own
> permutation null, p = 0.22). Drying reverses plasticisation, which is the
> effect being measured. The load is fixed at 300 gf so that hardness is a
> single-load measurement throughout; 300 gf sits mid-range of the five
> laboratories and matches Alrahlah.

**Specimens.** Discs 6 mm diameter × 2 mm, silicone mould, cured per
manufacturer instructions with irradiance recorded. Remove the
oxygen-inhibition layer identically on every specimen; **polish all specimens
to the same final grit and state it** (Randolph tested unpolished specimens,
and surface state plausibly contributes to its offsets).

**n = 6 specimens per (material × arm) cell.** With 12 materials: 144
specimens.

**Randomisation and blinding.** Randomise specimen-to-arm assignment and
indentation order. The operator reading diagonals should not know the material
or arm; label specimens by code.

---

## 3. Material panel and the pre-registered split

Power drives the panel size. Using the pooled slope +0.0263 and pooled residual
sd 0.244, for the paired test at α = 0.05 two-sided:

| n materials | filler sd 8 wt% | 10 wt% | 12 wt% |
|---|---|---|---|
| 8 | 0.48 | 0.66 | 0.81 |
| 10 | 0.62 | 0.81 | 0.92 |
| **12** | 0.73 | **0.90** | 0.97 |
| 14 | 0.81 | 0.94 | 0.99 |

**Twelve materials spanning filler loadings with sd ≥ 10 wt% gives 90 % power.**
Spread matters as much as count: 12 materials clustered in filler content are
worth less than 10 well spread. Choose the panel to span roughly 58–89 wt%.

**Pre-registered split — fix this before any specimen is made.**

- **Reference (anchor) products — 4.** Used only to estimate the laboratory
  offset. Choose products already measured in ≥ 2 of the five corpus
  laboratories so the new laboratory joins the existing graph: candidates are
  *Filtek Ultimate / Supreme XTE*, *Admira Fusion*, *Kalore*, *Tetric
  EvoCeram*, *Venus Diamond Flow*, *GrandioSO*.
- **Test products — 8.** Never used for calibration. Scored once.

Four anchors is not arbitrary. In the retrospective sweep, error fell
monotonically with anchor count and single anchors were unreliable — Randolph
went from 0.565 ± 0.158 (one anchor) to 0.386 (five), and Alrahlah from
0.484 ± 0.138 to 0.276 (three). One anchor is not enough; the gain flattens by
about four.

**Do not anchor on an extreme-composition product.** GrandioSO (89 wt%, the
highest in the corpus) was the *worst* single anchor for Alrahlah — it made
prediction worse than no calibration at all (0.675 vs 0.495). Anchors should
sit in the interior of the filler range. If GrandioSO is included, put it in
the test set.

**Avoid Tetric EvoCeram as an anchor** unless filler content is measured
in-house: it is the one product whose disclosed loading conflicts across
sources (Randolph 82.5 wt% by TGA ash residue vs Alrahlah 76.0 wt%
manufacturer-reported, a prepolymerised-filler accounting difference).

**Measure filler content in-house by TGA** for all 12 materials if at all
possible. Manufacturer figures are the single largest uncontrolled input to the
composition feature, and one in-house TGA run removes the ambiguity for the
whole panel.

---

## 4. Analysis plan, fixed in advance

**Q1 — the slope.** Primary endpoint, on the 8 test products only:

```
Δ_i = log(HV_ethanol,i) − log(HV_water,i)      per material i, specimen-matched
Δ_i = β₀ + β₁ · filler_wt%_i + ε_i
```

Primary test: `β₁ > 0`, one-sided α = 0.05. Report β₁ with a bootstrap 95 % CI
and alongside the retrospective pooled +0.0263. Material identity cancels in Δ,
so this is immune to between-material baseline differences — which is exactly
why it is the primary endpoint rather than a model-based comparison.

**Q2 — is one offset per laboratory enough?** This is the decisive test.

Fit, on the control arm only, across the new laboratory and the existing corpus:

```
H₀ (additive):     log HV = f(product) + b_lab
H₁ (interaction):  log HV = f(product) + b_lab + c_lab · filler_wt%
```

Compare by likelihood-ratio / nested F test. Report the per-product residuals
under H₀. **Pre-specified interpretation:** if H₀ is not rejected and the
residual sd is within about 3× the within-laboratory repeatability (0.022),
single-offset harmonisation is supported and Q3 is meaningful. If H₀ *is*
rejected, the single-offset method is mis-specified, that is the paper's
finding, and the Q3 numbers must be reported as describing a mis-specified
model rather than as a working method.

**Q3 — does calibration help?** Estimate one additive offset from the 4
reference products' control-arm cells. Apply it to predictions for the 8 test
products. Report all four of:

| | |
|---|---|
| uncalibrated | model with offset 0 |
| calibrated | offset from the 4 reference products |
| anchor-mean baseline | predict the mean of the reference products' measured values |
| oracle offset | the offset minimising test error (unachievable upper bound) |

The anchor-mean baseline is mandatory. In the retrospective sweep it **beat
every calibrated variant for Lodz** (0.416 vs best 0.528), and omitting it
would have hidden that. Also report the full distribution over all 15 non-empty
subsets of the 4 reference products, not just the mean, since the retrospective
spread across anchor choices exceeded the mean effect.

**Hardness-loss endpoint.** If a hardness-*loss* claim is made, Δ relative to
the matched water control is a **separate, independently reported outcome** with
its own test, not a re-description of Q1. Absolute HV and Δ have different
noise structure: Δ cancels material baseline but doubles measurement variance.

**Stopping and deviations.** No interim analysis; n is fixed at 6 per cell.
Record every deviation. Specimens lost to handling are reported, not replaced
silently.

---

## 5. What this experiment cannot do

- **It is one laboratory.** It can test whether the slope replicates and
  whether a single offset is adequate *for this laboratory*. It cannot
  establish that offsets are additive in general.
- **It fixes one protocol contrast.** Water vs 75 % ethanol at 7 days says
  nothing about the alkaline and thermocycling challenges, where the
  retrospective corpora disagree in sign on the filler effect (Alrahlah −0.024
  vs Lodz +0.011). Those appear to act by a different mechanism and need their
  own experiment.
- **It does not resolve the load question.** All specimens are measured at
  300 gf, by design. Whether the between-laboratory offsets are driven by
  indenter load would need one panel measured at several loads — a worthwhile
  but separate experiment. The retrospective data do not support a load
  explanation (offset ~ log load gives r² = 0.04, p = 0.79 across the four
  true-Vickers laboratories).

## 6. Estimated effort

144 specimens; roughly 2 specimen-preparation days, 7 days conditioning in
parallel for both arms, and 1–2 days of indentation. Optional in-house TGA on
12 materials adds about a day. Feasible in three weeks of laboratory time
including the pre-registration freeze.
