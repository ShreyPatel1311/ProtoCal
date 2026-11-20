# ProtoCal: Protocol-Aware Hierarchical Meta-Learning for Harmonizing Scattered Literature Data and Closed-Loop Inverse Design of Dental Resin Composites

## Proposed Method

## Material Focus and Justification
We focus on **dimethacrylate-based dental resin composites** (Bis-GMA/UDMA/TEGDMA matrices with glass/ceramic/nanofiller loadings). This class is chosen because (1) it is the most heavily studied dental biomaterial in the literature, with hundreds of papers systematically varying a small number of well-documented, numerically expressible formulation factors (monomer ratio, filler type/wt%/vol%/particle size, silanization, photoinitiator system, cure energy); (2) its core benchtop properties (flexural strength, degree of conversion, polymerization shrinkage, water sorption/solubility) are all governed by ISO 4049 and measurable with equipment present in any standard dental materials lab; and (3) formulations are genuinely combinatorial, making inverse design scientifically meaningful (there is a real, high-dimensional space to search) rather than a toy relabeling problem.

## The Core Problem We Target
The field's real bottleneck is not lack of data volume but **uncontrolled heterogeneity**: published studies differ in specimen geometry, storage medium/duration, thermocycling regime, testing-machine crosshead speed, and especially **curing protocol** (LED vs. QTH, irradiance, distance, exposure time). These nuisance factors are often partially reported and are entangled with the composition signal, so naive pooling of literature data (the standard practice in essentially all existing dental-composite ML papers, which fit a CNN/ANN/random forest directly on concatenated multi-study tables and validate with ordinary k-fold CV) produces models that fit idiosyncratic study-specific noise rather than the true composition→property relationship, and that cannot be trusted for extrapolative, inverse-design use.

## Step 1: Structured Literature Dataset (DentalCompDB)
We construct and publicly release a schema-driven dataset distinguishing two feature blocks for every extracted record:
- **Composition/process vector x** (the variable we want the model to learn from): monomer identity and mass/mol ratios (optionally encoded via RDKit-computed descriptors — molecular weight, C=C double-bond density, computed viscosity proxy — so the model can generalize to monomers not literally seen before, not just memorize one-hot IDs); filler type, wt%/vol%, mean particle size and distribution class (nano/micro/hybrid), silanization (yes/no, type); photoinitiator system and concentration; and a single physically meaningful **cure-dose feature** (irradiance × time = radiant exposure, J/cm²).
- **Protocol/nuisance vector p** (treated as a conditioning context, not as a feature to be creatively exploited): specimen geometry/standard (ISO 4049 vs. ADA vs. custom), storage medium and duration pre-test, thermocycling count, crosshead speed, temperature.
Each record also carries provenance (DOI, table/figure id) for auditability. Extraction uses PDF table-parsing tools plus LLM-assisted structured extraction with mandatory human verification on a 20% random audit sample (we report inter-rater/extraction agreement — e.g., Cohen's κ for categorical fields, mean relative error for numeric fields — as an explicit, rarely-reported data-quality metric for this field).

## Step 2: Hierarchical Protocol-Conditioned Meta-Model (ProtoCal)
For a property value y observed for formulation i in study s, we model:

y_{i,s} = f_shared(x_i; θ) + g_proto(p_s; φ) + b_s + ε_s

- **f_shared** is a shared composition→property function (we compare a multi-output neural network — jointly predicting flexural strength, DC%, shrinkage, and water sorption/solubility, since these are mechanistically correlated and joint training improves sample efficiency on small per-study n — against a multi-output Gaussian Process with a physically structured kernel over x, which gives calibrated epistemic uncertainty needed for the design loop).
- **g_proto** is a learned, *structured* correction term that takes the protocol vector p_s (not just a study ID) through a small embedding network. Unlike standard random-intercept/one-hot mixed-effects models used in classic meta-analysis (which cannot generalize beyond protocol combinations already observed), g_proto generalizes zero-shot to new protocol combinations by composing learned effects of geometry, aging, and cure dose — enabling us to ask the model to predict what any literature formulation's property *would be* under one fixed reference protocol (ISO 4049, no aging), i.e., to **harmonize** the literature onto a common footing.
- **b_s** is a per-study random residual offset and **ε_s** a per-study heteroscedastic noise term, both partially pooled (empirical-Bayes shrinkage) toward zero/global variance — this lets small or likely-unreliable studies be automatically downweighted rather than contributing full-strength noise, and the fitted b_s/ε_s values become an interpretable by-product: a data-quality signal correlating with sample size and standard-compliance that flags outlier/likely-erroneous studies in the field.

Training jointly maximizes the marginal likelihood (GP variant) or ELBO/NLL (variational NN variant) over all literature records.

## Step 3: Anchor Calibration with In-House Gold-Standard Data
A designed set of ~50 formulations (Latin hypercube over filler vol%, monomer ratio, and fixed, single reference cure protocol) is fabricated and tested in-house under one controlled protocol. These **anchor points** are added to the training objective with p_s fixed to the reference protocol; they pin down the absolute scale of f_shared (correcting literature-wide publication-bias/optimistic-reporting drift) and let us validate that g_proto-corrected literature predictions agree with real reference-protocol measurements — directly testing the harmonization hypothesis rather than assuming it.

## Step 4: Closed-Loop Inverse Design
Using the calibrated model's predictive mean and uncertainty, we run multi-objective Bayesian optimization (noisy expected hypervolume improvement, qNEHVI) over the formulation space x to propose Pareto-optimal candidates jointly maximizing flexural strength and degree of conversion while minimizing shrinkage and water sorption/solubility. Top-ranked candidates are fabricated and benchtop-tested (new specimens, same reference protocol), closing the loop with genuine prospective validation — the form of evidence almost entirely absent from existing dental-materials ML papers, which validate only by resampling the same literature table.

**Why this should work:** the method directly targets the stated bottleneck (heterogeneous, inconsistently reported data) with a mechanism — structured protocol conditioning plus partial pooling plus a small controlled anchor set — rather than ignoring it (pooling) or only removing it adversarially (domain-invariant learning, which would also destroy real composition-aging interactions we may want to predict, e.g., shrinkage under different cure doses).

## Experiment Plan

## Datasets
1. **DentalCompDB (new, literature-derived):** Systematic search (PubMed, Scopus, Web of Science) 2000–2025 for studies on experimental or commercial dimethacrylate dental composites reporting flexural strength (ISO 4049 3-/4-point bending), degree of conversion (FTIR/Raman), polymerization shrinkage (strain gauge, bonded-disk, AccuVol, or density-based methods), and/or water sorption/solubility (ISO 4049 gravimetric), with explicit composition tables (own-formulated composites, which report filler/monomer ratios explicitly, are prioritized). Target: 150–300 papers, 800–1500 formulation×property records, each annotated with the composition vector x and protocol vector p defined above, plus DOI provenance. Released publicly (Zenodo/Figshare) with schema documentation — this is itself a community benchmark contribution.
2. **In-house anchor/validation set (new, benchtop):** ~50 formulations from a Latin hypercube design over filler vol% (40–70%, silanized barium-aluminosilicate glass), Bis-GMA:TEGDMA(:UDMA) ratio, and camphorquinone/amine concentration, all cured under one fixed reference protocol (standard LED, 1200 mW/cm², 20 s). Each formulation: n=5 bars (25×2×2 mm) for 3-point flexure on a universal testing machine; FTIR-ATR real-time cure monitoring for DC%; bonded-disk/strain-gauge shrinkage measurement; ISO 4049 gravimetric disks (15×1 mm) for water sorption/solubility. Split into (a) training anchors used for calibration and (b) a held-out prospective test subset of formulations never seen in DentalCompDB.
3. **Round-2 BO validation batch:** 10–15 freshly fabricated formulations proposed by the trained model's Bayesian-optimization loop, tested identically, used only for final prospective evaluation (never touched during model fitting).

## Baselines
- **Pooled-ML baseline:** multi-output ANN/random forest trained on all literature records with composition features only, ignoring study/protocol identity — representative of current published practice in dental-composite ML papers.
- **Standard mixed-effects model:** linear/ANN model with per-study one-hot random intercepts (classic meta-analytic approach; cannot generalize to unseen protocol combinations).
- **Domain-adversarial NN (DANN):** representation trained to be invariant to study identity, as a contrasting domain-generalization strategy.
- **Naive transfer learning:** pretrain on pooled literature, fine-tune only on anchor set.
- **Single global GP:** one Matérn-kernel GP over composition features with no hierarchical/protocol structure.
- **Random-search and single-objective greedy** baselines for the inverse-design/BO stage.

## Metrics
- **Predictive accuracy** (RMSE/MAE per property: MPa for flexural strength, % for DC and shrinkage, µg/mm³ for sorption/solubility) under (i) random k-fold split, (ii) leave-one-study-out CV (harder, realistic generalization test), and (iii) the held-out in-house prospective test set (the most important number, since it is true generalization beyond anything used in fitting).
- **Uncertainty calibration:** negative log-likelihood and prediction-interval coverage probability (PICP) at 90%/95% on the prospective test set.
- **Harmonization quality:** for near-duplicate compositions tested across ≥2 different studies, measure variance of raw reported values vs. variance after g_proto-based protocol correction (report % variance reduction).
- **Data-quality signal validity:** correlation between fitted per-study residual/noise terms (b_s, ε_s) and independent proxies of study reliability (reported sample size, standard compliance, journal metrics).
- **Inverse-design performance:** Pareto hypervolume achieved vs. number of physical benchtop experiments consumed (sample efficiency) relative to random/greedy search; measured-vs-predicted agreement (RMSE, calibration) for the physically fabricated Round-2 BO candidates — the ultimate real-world test of the whole pipeline.

## Ablation Studies
1. **Remove protocol embedding** (plain per-study random intercept instead of structured g_proto(p_s)) — isolates the benefit of generalizable protocol conditioning versus classic mixed-effects modeling.
2. **Remove anchor calibration** (train on literature only, zero-shot predict the reference protocol) — quantifies the value of the small controlled in-house dataset.
3. **Anchor-set size sweep** (10/20/40/60 formulations) — produces a sample-efficiency curve answering "how much benchtop data is actually needed," directly relevant to adoption by other small labs.
4. **Multi-task vs. single-task heads** (joint 4-property prediction vs. four independent models) — tests whether exploiting cross-property correlation helps under data scarcity.
5. **GP vs. neural-network f_shared** — accuracy/calibration trade-off, and which backbone is more reliable inside the BO loop.
6. **Chemistry-informed (RDKit) vs. categorical monomer encoding** — tests extrapolation to monomer combinations absent from the literature set.
7. **With vs. without per-study heteroscedastic noise weighting** — shows whether automatic downweighting of unreliable studies improves held-out accuracy versus equal-weighting all literature records.

