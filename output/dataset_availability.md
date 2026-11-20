# Dental biomaterials ML datasets — public availability audit

Checked 2026-11-20. Verdict column = can you download and reuse it today.

## Summary

| Dataset | Type | Open? | License | Verdict |
|---|---|---|---|---|
| EloKriti BPA-free dental resin (Zenodo 23605867) | Computational (MD) | **Yes** | CC-BY-4.0 | **Reusable** |
| Szczesio-Wlodarczyk aging protocols (Zenodo 14576242) | Experimental | **Yes** | CC-BY-4.0 | **Reusable** |
| Domarecka dual-cure vs light-cure (Zenodo 5569184) | Experimental | **Yes** | CC-BY-4.0 | **Reusable** |
| Dental sealant formulations (Zenodo 11186969) | Experimental | Yes | see record | Reusable (OriginPro files) |
| Ceramic brackets (Zenodo 6372935) | Experimental | Yes | CC-BY-4.0 | Reusable (not a composite) |
| Paniagua et al. 2025, *J Dent Res* | Literature-curated | **No** | — | Not released |
| ICCSMT '25, 347 formulations | Literature + proprietary | **Unverified** | — | Page 403; partly proprietary |
| Dental adhesive active-learning BO | Proprietary | **No** | — | Code only, data proprietary |

---

## 1. EloKriti — BPA-free dental resin candidates
**https://zenodo.org/records/23605867** · DOI 10.5281/zenodo.23605867
Balantrapu, Feitosa Sochacki, Ganesh (University at Buffalo SUNY). Published 2026-10-24, v1.0. **A newer version exists.**

- License: **"Creative Commons Attribution 4.0 International"** — open, direct download, no request needed.
- Files: `Data_Zenodo.zip` (8.2 MB) + `README.txt`.
- Contents: `candidate_pool.csv` (~1000-candidate screening pool), `evaluated_clean_no_expansion.csv` (descriptors + MD-derived objectives), `gaff_ok.csv`, plus `candidate_pool_audit/` (7 CSVs) and `reviewer_validation/` (~17 CSVs), and `stage2_validation/` with cured-network structures, Tg source data and mechanical outputs.
- Objectives: elastic modulus (max), volumetric shrinkage (min), fractional free volume (min).
- **Computational, not experimental** — GROMACS + HTPolyNet virtual curing. No lab measurements.
- Code: **https://github.com/hbalantr/Elokriti**, MIT licensed. Study CSVs are *not* in the repo (expects a local `data/`).
- Caveat: manuscript not yet published ("final journal citation and DOI will be added after publication").
- Bundled third-party file `PI1070.csv` comes from RadonPy and must be cited separately; its license is not stated.

## 2. Artificial aging protocols for dental composites
**https://zenodo.org/records/14576242** · DOI 10.5281/zenodo.14576242
Szczesio-Wlodarczyk et al., Medical University of Lodz. Published 2025-02-04. Paper: *J Funct Biomater* 16(2):49, doi 10.3390/jfb16020049.

- License: **"Creative Commons Attribution 4.0 International"** — open, no login.
- Files (263.2 MB): flexural strength, flexural modulus, diametral tensile strength, fracture toughness, Vickers hardness (zipped), a README `.docx`, plus SEM micrographs for Filtek Ultimate (163 MB) and Heliomolar Flow (98 MB).
- **Experimental.** 12 materials across 5 types, measured before and after aging. Aging uses 0.1 M NaOH hydrolytic degradation; three protocols from the authors' earlier work are compared.
- **Most relevant record found** — it is explicitly about protocol standardisation, i.e. the exact heterogeneity problem ProtoCal targets.
- Unresolved: whether protocol parameters are stored per measurement. The README is a `.docx` that couldn't be read remotely; download and check before relying on it.
- Only 2 of the 12 materials are named on the record page (from SEM folder names). Specimens per group not stated on the page.

## 3. Dual-curing vs light-curing composites
**https://zenodo.org/records/5569184** · DOI 10.5281/zenodo.5569184
Domarecka, Szczesio-Wlodarczyk, Krasowski, Fronczek, Gozdek, Sokolowski, Bociong. Published 2021-10-17. Paper: *Coatings* 2021.

- License: **"Creative Commons Attribution 4.0 International"** — open, no login.
- Files (155.4 kB, all `.xlsx`): DTS, flexural modulus, flexural strength, hardness, microhardness, shrinkage stress.
- **Experimental.** 9 named commercial materials — 6 dual-cure (Bulk EZ, Fill-Up!, StarFill 2B, Rebilda DC, MultiCore Flow, Activa Bioactive-Restorative), 3 light-cured (Filtek Bulk Fill Posterior, Charisma Classic, G-aenial Universal Flo).
- n: 9 per group for hardness and DTS; 5 per group for flexural strength, modulus and shrinkage stress.
- Conditioning: 24 h in water at 37 °C. No aging protocol.
- README notes data is "Processed, manually entered data" — summary values, not raw instrument output.

## 4. Paniagua et al. 2025 — NOT released
**https://pmc.ncbi.nlm.nih.gov/articles/PMC12000627/** · doi 10.1177/00220345241311888 · *J Dent Res* 104(5):513–521
Preprint: https://pmc.ncbi.nlm.nih.gov/articles/PMC11482990/

- **No data availability statement exists.** The only related sentence is a future intention:
  > "A future goal is to create an open-source dataset, where researchers can contribute their research data in exchange for the use of the dataset for their research."
- Single supplement: `sj-docx-1-jdr-10.1177_00220345241311888.docx` (405.2 kB) containing Appendix Tables 1–14. Appendix Table 1 covers the 17 attributes for the 233 composites and *may* be a per-composite listing — unverified, worth opening.
- Dataset: "From 200+ publications, 321 composite formulations with 28 CAs and 17 POs were included", reduced to **233 commercial composites, 17 attributes, 7 outcomes**. Sources: "Google Scholar and company brochures".
- **Commercial products only** — experimental formulations were deliberately excluded.
- Predicted: flexural modulus, flexural strength, volumetric shrinkage, shrinkage stress. Degree of conversion is an *input*, not an output.
- Very sparse: flexural strength 54.9% of entries, fracture work 3.9% (9 entries).

## 5. ICCSMT '25, 347 formulations — UNVERIFIED
**https://dl.acm.org/doi/full/10.1145/3802927.3802965**

- Page returns **HTTP 403**; could not read the data availability statement, licensing or any release location.
- From search metadata only: compiled from peer-reviewed literature 2010–2024, manufacturer technical documentation, **and proprietary experimental records**. The proprietary component makes full public release unlikely.
- Treat as unresolved until someone with ACM DL access reads it.

## 6. Dental adhesive active learning + BO — NOT released
**https://www.researchgate.net/publication/380682933**

- 91 dental adhesive formulations described as **proprietary**.
- Code stated to be on GitHub; dataset release not evidenced.

---

## What this means for the proposal

**The DentalCompDB contribution survives, and is stronger than I first judged.** I had flagged the 347-formulation set as making it redundant; having checked, that set is unverifiable and partly proprietary, and the one comparable literature compilation (Paniagua) is unreleased, commercial-products-only, and sparse. **There is no public, reusable, literature-extracted dental composite dataset with protocol metadata.** That gap is real.

**But two things should change in the plan:**

1. **The ~50-formulation in-house anchor set is partly pre-purchasable.** Records 2 and 3 are CC-BY, experimental, and overlap the target properties (flexural strength, modulus, DTS, hardness, shrinkage stress). They won't replace the anchors — different protocols, no per-formulation composition for commercial products — but they are free validation material and cut risk on Step 3.

2. **Record 2 is a natural collaborator and a protocol-effect testbed.** The Lodz group has published on standardising aging protocols and released the data CC-BY. ProtoCal's `g_proto` term is a model of exactly what they measured empirically. Their 12-material before/after-aging data is the cleanest available probe of whether a protocol-correction term recovers known effects.

**Open question to resolve before committing:** download the Paniagua supplement `.docx` and check whether Appendix Table 1 is a per-composite table. If it is, 233 commercial composites with 17 attributes are effectively already extracted — which would cut the Step 1 extraction burden substantially, with attribution.
