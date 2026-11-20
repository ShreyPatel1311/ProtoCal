# Corrections — 2026-11-20

A 17-agent adversarial audit (5 independent assessors + 3 paper designs + 9 judges)
re-derived the numbers in `DATA_AUDIT_FINDINGS.md`. **Four of the six findings are wrong
or mis-stated.** This file is the correction of record; the findings document should be
read only alongside it. Corrected analyses are in progress.

## C1. Finding 3 is WRONG — b_s is recoverable; the estimator was broken

**Published claim:** per-study offsets recover at only r = 0.313 and plateau near 0.558
even at 10× data, therefore `b_s` is *information-limited*.

**Reality:** the information is there. Given the true `f_shared` and `g_proto`, an oracle
BLUP recovers `b_s` at **r = 0.945 at the sparsest setting** (0.978 at full reporting,
0.996 at 4× rows) — `scripts/oracle_blup.py`. The failure was ours.

Two bugs, both fixed:

1. **The τ objective was unbounded below.** `loss()` minimised
   `0.5·b²/τ² + k·log τ` jointly over `b` and `τ` by gradient descent. Taking
   `b = τ^{3/2}` sends this to −∞ as τ → 0, so τ collapsed and `b_s` was crushed to
   ~1e-3 against a true scale of 0.35. Replaced with a closed-form method-of-moments
   update, `update_tau()`.
2. **The prior was applied once per mini-batch** and normalised by the *batch* size
   rather than the training-set size, over-weighting it by the number of batches per
   epoch. Learned τ swung 334× with batch size (0.0007 at bs=64 → 0.2341 full-batch).
   Now scoped to all studies and normalised by `total_n_obs`.

A third issue surfaced while fixing these: `b` initialised at zero makes the
method-of-moments update return τ² = 0 on its first step, pinning τ to its floor and
locking `b` at zero. Fixed by initialising `b` at the prior scale plus a τ warmup.

After the fixes, τ recovers to 0.120 against a true 0.35 and joint-fit `b_s` recovery
rises 0.313 → 0.352. Still far below the oracle's 0.945, so **the honest finding is now
about joint identifiability** — `f_shared` absorbs study offsets when both are fit
together — **not about information content.** The density sweep was measuring our
estimator, not a property of the data.

## C2. Finding 5 is WRONG and must be withdrawn

**Published claim:** 44 robust outliers across 238 cells, "unexplained".

**Reality:** the expected number of MAD-flagged points under *exact* normality at
n = 6–9 is **51.1**. We observed 44. The data are **cleaner than Gaussian**. There is no
anomaly, and the claim pointed the wrong way.

This was published about another research group's CC-BY dataset. It is withdrawn without
qualification. The Lodz corpus shows no evidence of data-quality problems.

## C3. Finding 2's headline statistic is invalid — and the true result is stronger

`interaction_vs_noise` in `scripts/audit_data.py` divided interaction SS by cell *count*
rather than interaction *degrees of freedom*, and omitted n per cell, comparing cell-mean
residuals against single-specimen variance. Its null expectation is 0.075–0.114, not 1,
so "1.42 means interaction exceeds noise" does not follow.

Under correct tests (proper F, heteroscedastic bootstrap, aligned rank transform) the
material × protocol interaction is significant for **all five properties**
(p ≤ 7.8e-09), including flexural modulus, which the findings document explicitly calls
fine. The corrected claim is stronger than the published one — and it makes Findings 2
and 6 mutually consistent for the first time.

## C4. The synthetic numbers did not reproduce from the committed code

Commit `af2628a` inserted the interaction `U`/`V` draws ahead of the data-generating
draws, shifting the RNG stream: HEAD at seed 0 produced 829 rows where the committed
JSONs and paper Table 2 record 780. Fixed by drawing `U`/`V` from a separate generator;
seed 0 again yields 780 rows.

## C5. "13 materials" is a filename artifact — there are 12

`Grandio SO Heavy Flow 2` occupies exactly and only the baseline-protocol cells that
`Grandio SO Heavy Flow` is missing. The Zenodo abstract states twelve materials. The
design is 238/240 = **99.2%** complete, not 238/260 = 91.5%.

## C6. Claims never tested that the paper's motivation depends on

- **Leave-one-protocol-out was never run.** It fails: holding out the reference protocol
  gives held-out RMSE **1.818** against a trivial grand-mean baseline of **1.833**. The
  other three folds look good only because the three aged protocols are near-duplicates
  (they differ by ≤6.5%, while aged-vs-unaged differs by 12–42%). Dropping the unaged
  baseline collapses the headline 31.2% protocol share to **0.3%**. This corpus has
  roughly **one** effective protocol degree of freedom.
- **ProtoCal does not beat a trivial baseline.** Held-out-cell task: grand mean 0.994,
  material means 0.799, additive two-way linear 0.660, ProtoCal 0.649. A 20-line
  closed-form two-way ANOVA ties it to within 0.001. The tie is algebraic: one-hot
  materials make `f_shared` a lookup table and 4 protocol levels make `g_proto`
  saturated.
- **`grouped_cv_splits` has zero callers** while `README.md` advertises
  leave-one-study/lab-out evaluation.
- **No calibration metric exists anywhere** (no NLL, CRPS, coverage, sharpness) despite
  the model fitting per-study heteroscedastic variances.

## C7. Statistical power

The 8 × 25% split design cannot support a head-to-head significance claim at any number
of splits. The Nadeau–Bengio variance inflation term `n_test/n_train = 1/3` does not
shrink with n, flooring the SE at 0.0729 and capping |t| at 1.86. Minimum detectable
effect is 0.162 at n=8 and 0.066 at n=48, against a largest observed gain of 0.037.

## What survives

**Finding 1 survives and strengthens.** After removing the sampling-noise floor, protocol
variance shares are 31.5 / 33.0 / 20.3 / 11.8 / 2.5 % — same ordering, same magnitudes.
Recomputed on the corrected 12-material table: 34.3 / 30.4 / 24.5 / 12.0 / 2.2 %.

**Finding 4's conclusion survives**, though its argument needs replacing: drop the
per-cell AIC tournament (no power at n=5–10) in favour of pooled within-cell standardised
residuals — skew −0.10 to +0.13 (all skewtest p > 0.28), negative excess kurtosis −0.49
to −0.88. Symmetric and short-tailed; Gaussian is fine.

**The gauge fixing is correct**, verified exact for both the shared and interaction terms.
