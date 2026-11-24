# Pre-registration: external validation of shared-composite calibration

**Frozen before the validation study's measurements were read, extracted or
fitted.** Committed as a standalone change so the git history shows it preceded
the data. Nothing below may be changed after the validation data are loaded; any
deviation is reported as a deviation.

## 0. Why a freeze is necessary here

The retrospective result is not robust to analyst choices. The same Łódź
held-out comparison — calibrated against uncalibrated prediction, all available
reference products — gives:

| filler definition | Krüger in training | Δ RMSE (cal − uncal) | 95% CI |
|---|---|---|---|
| manufacturer | excluded | **+0.055** | [+0.047, +0.062] |
| manufacturer | included | **−0.028** | [−0.038, −0.012] |
| measured `W_i` | excluded | +0.008 | [−0.009, +0.022] |
| measured `W_i` | included | +0.043 | [−0.004, +0.070] |

Calibration significantly *harms* under one configuration and significantly
*helps* under another. With four such configurations, plus the choice of
reference subset (which spans 0.157–0.665 for Łódź alone), an unfrozen analysis
can produce whatever conclusion the analyst prefers. The freeze is therefore not
a formality; it is the only thing that makes the external test informative.

## 1. Hypothesis and its direction

**H1 (pre-specified direction).** Calibrating by a single additive offset
estimated from shared reference composites will **not reduce** prediction error
on held-out composites, relative to uncalibrated prediction.

This is the direction the retrospective analysis suggests. It is stated in
advance so that a result in the opposite direction counts as evidence against
our own prior claim, and it will be reported as such. **The result is reported
whichever way it falls, including if calibration helps.**

## 2. Model — frozen

- Estimator: `sklearn.linear_model.RidgeCV(alphas=np.logspace(-3, 3, 25))`,
  default 
  leave-one-out generalised cross-validation for alpha.
- Target: `log(HV)` of the cell mean.
- Features, in this order: standardised `filler_wt_pct` (one column), then the
  14 protocol features of `protocal.protocols.FEATURES`.
- Standardisation of `filler_wt_pct`: mean and sd computed on **training rows
  only**.
- No hyperparameter search beyond the RidgeCV alpha grid above. No model
  selection against the validation study.

## 3. Training set — frozen

- `protocal.pooled.load(filler_variant="manufacturer", exclude_kruger=True)`,
  then `protocal.pooled.aqueous_family(...)`.
- **Krüger is excluded.** Rationale fixed in advance and independent of
  outcome: Krüger did not measure Vickers hardness. A Fischerscope HM 2000
  measured Martens hardness at 1 N and converted it to a Vickers-equivalent by
  an undisclosed relation, and its specimens were dried 24 h under vacuum
  before testing, which reverses the plasticisation being measured. It is a
  different measurand.
  *We note explicitly that this choice is the one LESS favourable to the
  method:* with Krüger included, calibration appears to help Łódź (−0.028).
  We freeze the unfavourable choice because the measurement argument is prior
  to the result.
- Restriction to the aqueous/ethanol family is likewise prior: 8 of 14 protocol
  features are exercised by only one laboratory, so leave-one-laboratory-out is
  ill-posed outside the family every laboratory shares.

## 4. Filler definition — frozen

- **Primary: `manufacturer`.** Chosen because a newly-identified published study
  will report manufacturer figures and not in-house TGA, so this is the only
  definition applicable consistently to training and validation data.
- **Pre-specified secondary: `measured`** (Randolph TGA `W_i` where available).
  Reported alongside, never substituted for the primary.
- If the validation study reports its own measured filler content, that is
  reported as a third, clearly-labelled variant and does not replace the
  primary.
- No product's filler value may be revised after the validation data are read.

## 5. Reference / test split — frozen rule, not a frozen list

Applied mechanically once the validation study's products are identified:

1. **Reference-eligible** = a validation-study product that appears in the
   training corpus under `protocal.pooled.PRODUCT_ALIASES`, after confirming
   product identity (manufacturer and product name must match; a different
   shade or lot is the same product, a different product line is not).
2. **Excluded from reference use** regardless of overlap:
   - *Tetric EvoCeram* and *Kalore* — largest manufacturer-vs-measured filler
     discrepancies (−12.0 and −10.9 wt%); a reference whose composition is
     contested is the wrong place to carry the correction.
   - any product whose filler wt% is the maximum or minimum of the training
     range — anchoring on an extreme composition was actively harmful
     retrospectively (GrandioSO, 89 wt%, was Alrahlah's worst single
     reference, 0.675 against 0.495 uncalibrated).
3. **Test set** = all remaining validation-study products. A product is never
   both reference and test.
4. **All reference-eligible products are used together** for the primary
   estimate. The retrospective subset sweep is reported as a secondary
   descriptive result only; it is not the primary, because subset choice is
   precisely the instability under examination.
5. If the rule yields zero reference-eligible products, the primary comparison
   is not estimable and that is reported as the outcome. If it yields zero test
   products, likewise.

## 6. Estimators compared — all scored on identical cells

| name | definition |
|---|---|
| `uncalibrated` | model prediction, offset 0 |
| `calibrated` | prediction + one additive offset, offset = mean(`log HV` − prediction) over all reference-product cells |
| `reference_mean` | predict the mean measured `log HV` of the reference-product cells, ignoring the model |

`reference_mean` is mandatory: it uses exactly the same information the
calibration gets, and retrospectively it beat every calibrated variant for Łódź
(0.416 vs best 0.528). Reporting calibration against only the uncalibrated model
would hide that.

## 7. Scoring — frozen

- Metric: RMSE of `log HV` over **test-product cells only**.
- **Primary comparison:** `calibrated` − `uncalibrated`. Positive means
  calibration increased error.
- **Secondary comparisons:** `calibrated` − `reference_mean`; `uncalibrated` −
  `reference_mean`.
- Uncertainty: **cluster bootstrap resampling whole test products** with
  replacement (the product is the unit, not the cell), 10,000 draws,
  `np.random.default_rng(0)`, 95 % percentile interval on the difference.
  Resampling cells would understate uncertainty because cells within a product
  are not independent.
- **Decision rule.** CI excludes 0 and positive → calibration harms. Excludes 0
  and negative → calibration helps. Includes 0 → inconclusive at this sample
  size.
- Conditionality is stated in every report of the interval: it is conditional on
  this validation study and this reference set, and with one reference product
  it cannot support any claim about *how many* references are needed.

## 8. Declared limitations, in advance

- A single validation study with a small number of overlapping products gives a
  test of **whether** calibration helps there, not of how many references are
  required. That claim needs studies with more overlap and is out of scope.
- The validation study is not independent of the training corpus in every
  respect: if it shares a laboratory, a research group or a materials supplier
  with any training study, that is reported.
- Vickers hardness measured at a different indenter load from the training
  laboratories is a known confound; the validation study's load and dwell are
  recorded and reported whether or not they differ.

## 9. Code

The frozen procedure is implemented in `scripts/external_validation.py`. It
accepts the validation data as a CSV in a fixed schema and exposes **no options
that alter the primary analysis**. Both this document and that script are
committed before the validation data.
