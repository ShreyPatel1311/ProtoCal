# Data audit and analysis — findings

All five datasets are CC-BY-4.0, downloaded and MD5-verified (`scripts/download_data.py`).
Analysis targets the Lodz aging corpus (Zenodo 14576242), the only public dataset found
that carries **protocol metadata alongside specimen-level values**.

## Corpus

| | |
|---|---|
| Specimen measurements | **1,802** |
| Materials | 13 |
| Aging protocols | 4 |
| Properties | 5 (flexural strength, flexural modulus, diametral tensile, Vickers hardness, fracture toughness) |
| Cells present | 238 of 260 — **91.5% complete factorial** |

Protocol is encoded per file, e.g. `FS, Admira Fusion, 0.1M NaOH, 7 days, 60°C.xlsx`,
parsed into medium / duration / temperature. The four protocols are a 24 h 37 °C water
baseline, NaOH 7 d 60 °C, water 5 d 55 °C, and a combined water-then-NaOH sequence.

## Finding 1 — protocol is a first-order effect, not a nuisance detail

Variance decomposition on log values, material × protocol:

| Property | Material | Protocol | Interaction | Interaction ÷ noise |
|---|---|---|---|---|
| Flexural strength | 55.5% | **31.2%** | 13.3% | **1.42** |
| Vickers hardness | 49.1% | **32.3%** | 18.5% | 0.78 |
| Diametral tensile | 60.4% | 19.8% | 19.8% | 0.51 |
| Fracture toughness | 79.5% | 11.6% | 8.9% | 0.30 |
| Flexural modulus | 93.6% | 2.5% | 3.9% | 0.23 |

Protocol explains up to **a third** of all variance in flexural strength and hardness —
the two most frequently reported properties in the literature. This is direct empirical
support for the premise that pooling multi-study data without protocol conditioning
confounds composition signal with protocol effects.

## Finding 2 — the additive protocol model is mis-specified for flexural strength

This one contradicts the proposal as drafted.

ProtoCal assumes `y = f_shared(x) + g_proto(p) + b_s`: the protocol effect is **additive
and shared across materials**. The interaction column above tests that assumption
directly. For flexural strength the material × protocol interaction is **1.42× the
within-group measurement noise** — i.e. materials genuinely respond *differently* to the
same aging protocol, by more than noise can explain. Vickers hardness is marginal (0.78).

An additive `g_proto(p)` cannot represent this. The remaining three properties are fine
(0.23–0.51, interaction below noise).

**Consequence:** `g_proto` needs a material-dependent term for strength-like properties —
a low-rank interaction `g(p) + <u(x), v(p)>` rather than `g(p)` alone. Keeping the purely
additive form would bake in a known bias on the single most important property.

## Finding 3 — per-study offsets `b_s` are not recoverable at literature density

From the synthetic precondition check (`scripts/smoke_test.py`, `scripts/density_sweep.py`):

| Setting | Obs per (study, property) | g_proto recovery | b_s recovery |
|---|---|---|---|
| Sparse, as published | 6.2 | **+0.951** | +0.313 |
| 2× rows | 14.3 | +0.950 | +0.435 |
| Full reporting | 13.5 | +0.957 | +0.525 |
| 4× rows + full reporting | 59.4 | +0.968 | +0.558 |

`g_proto` is recovered essentially perfectly (r ≈ 0.95) and is **insensitive to density**,
because it pools across all studies. `b_s` is poor at realistic sparsity and plateaus
around 0.55 even with ten times the data — it is information-limited, not an optimisation
failure. A capacity/τ sweep (`scripts/capacity_sweep.py`) confirms this: varying f_shared
from 128×128 down to 16 units, and fixing τ at its true value, moves b_s recovery by less
than 0.08.

**Consequence:** drop the claim that `b_s`/`ε_s` provide an interpretable per-study
"data-quality signal" flagging unreliable studies. The quantity cannot be estimated
reliably from literature-density data. The harmonisation claim, which depends only on
`g_proto`, is unaffected and is the result worth defending.

## Finding 4 — Gaussian likelihood survives (negative result)

Brittle-material strength is classically Weibull, which would make the Gaussian NLL
mis-specified. Tested per cell by AIC against Normal, Weibull and Lognormal:

- Normal is AIC-best in only 4–17% of cells, but
- Shapiro–Wilk rejects normality in only **4–21%** of cells,
- median skew is ≈ 0 (−0.18 to +0.12),
- Weibull beats Normal in **50/96 strength cells (52%)** — a coin flip,
- median Weibull shape k = 8–18, and at high k Weibull ≈ Normal anyway.

At n ≈ 5–10 per cell, AIC differences of this size are not informative and Shapiro has
little power. **The Gaussian assumption cannot be rejected and is retained.**

## Finding 5 — 18.5% of cells contain a robust outlier, unexplained

44 outliers (robust z > 3.5) plus 5 high-dispersion cells (CV > 30%) across 238 cells.
Finding 4 rules out heavy Weibull tails as the explanation, so these are not a
distributional artefact. They warrant inspection before the corpus is used as anchor
data — full list in `output/data_audit.json`.

## Reproduce

```
python scripts/download_data.py
python scripts/audit_data.py
python scripts/distribution_check.py
python scripts/smoke_test.py
python scripts/density_sweep.py
python scripts/capacity_sweep.py
```

---

## Finding 6 — the low-rank interaction fix does NOT work (negative result)

Finding 2 showed the material × protocol interaction is real for flexural strength
(1.42× noise). The obvious correction is a gauge-fixed low-rank interaction term,
`g(p) + <u(x), v(p) − v(p_ref)>`. It is implemented (`ProtoCalConfig.interaction_rank`)
and verified: the gauge fix holds for the interaction too, `|interaction(x, p_ref)| ≈ 1e-8`.

It does not help. Tested by holding out entire (material, protocol) cells — predicting a
combination nobody tested, which is exactly what harmonising scattered literature
requires — over 8 random 25% splits of the Lodz corpus (432 specimens, 48 cells):

| Rank | Held-out RMSE (std. units) | vs additive | Splits better |
|---|---|---|---|
| 0 (additive) | 0.649 ± 0.113 | — | — |
| 1 | 0.685 ± — | −0.037 | 1/8 |
| **2** | **0.649 ± 0.097** | **+0.0001** | **4/8** |
| 3 | 0.681 ± — | −0.033 | 2/8 |

Rank 2 is indistinguishable from additive; ranks 1 and 3 are worse.

Per-property, the gains do track the measured interaction share — diametral tensile
+0.037, Vickers +0.033, flexural strength +0.010, all properties with substantial
interaction — while flexural modulus, which has the *least* interaction (3.9%), gets
**worse** by 0.097, consistent with the extra parameters just adding variance. So there
is a directionally correct signal. But every one of these differences is far inside the
split-to-split standard deviation (0.08–0.23). None is evidence of anything.

**Why it fails.** The corpus has only 4 protocol levels and 48 cells. A low-rank
interaction has to estimate a protocol factor space from that, and there is not enough
of it. The interaction is detectable *in aggregate* by variance decomposition, which
pools all specimens, but it is not *structured enough* to predict a new cell.

This is the same pattern as Finding 3: a quantity that is clearly present in aggregate
is not estimable well enough to use. Two independent instances of it now.

**Consequence.** `interaction_rank` stays in the codebase, defaults to **0**, and is
documented as unsupported by current evidence. Enabling it needs a corpus with many more
protocol levels — which is precisely what a literature-derived dataset would provide,
since published studies span far more than four aging regimes. The honest claim for a
paper is: the additive model is measurably mis-specified for strength properties, and
correcting it requires more protocol diversity than any single-laboratory study contains.
