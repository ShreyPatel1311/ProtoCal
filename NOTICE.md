# Attribution

This repository began as a working copy of **IRIS — Interactive Research Ideation System**

- Upstream: https://github.com/Anikethh/IRIS-Interactive-Research-Ideation-System
- Paper: Garikaparthi, Patwardhan, Vig, Cohan. *IRIS: Interactive Research Ideation
  System for Accelerating Scientific Discovery.* ACL 2025 System Demonstrations,
  pp. 592–603. https://aclanthology.org/2025.acl-demo.57/

The original IRIS README is preserved as `docs_IRIS_README.md`. Upstream history is
retained in this repository and upstream remains configured as the `upstream` remote.

**Licence status:** at the time of copying, the upstream repository carried **no LICENSE
file** and GitHub reported no licence. Absent a licence, default copyright applies and
all rights are reserved by the original authors. This copy was made at the explicit
direction of the repository owner. Anyone reusing the IRIS-derived portions of this
repository should seek permission from the upstream authors.

Files original to this project (not derived from IRIS):

```
protocal/           ProtoCal model, synthetic validation, training
scripts/            data download, audit, smoke test, sweeps
paper/              manuscript
output/             generated results and reports
run_iris.py         launcher wiring IRIS to the Claude Code backend
src/models/claude_code_llm.py
research_goal_dental.txt
```

The patch to `app.py` (MCTS review-feedback plumbing) is a bug fix against upstream;
see `docs/iris_review_feedback_fix.patch`.

## Datasets

All redistributed analysis derives from CC-BY-4.0 datasets. Data files themselves are
**not** committed — `scripts/download_data.py` fetches them. Attribution:

- Balantrapu, Feitosa Sochacki, Ganesh. *Data for "Active-Learning Discovery of BPA-Free
  Dental Resin Candidates…"* Zenodo. doi:10.5281/zenodo.23605867 (CC-BY-4.0)
- Szczesio-Wlodarczyk, Kopacz, Ranoszek-Soliwoda, Sokołowski, Bociong. *Towards the
  Standardization of Artificial Aging Protocols for Dental Composites.* Zenodo.
  doi:10.5281/zenodo.14576242 (CC-BY-4.0). Article: J Funct Biomater 16(2):49,
  doi:10.3390/jfb16020049
- Domarecka et al. *A Comparative Study of the Mechanical Properties of Selected Dental
  Composites with a Dual-Curing System…* Zenodo. doi:10.5281/zenodo.5569184 (CC-BY-4.0)
- *Study of Physical-Chemical and Mechanical Properties of New Dental Sealants
  Formulations.* Zenodo. doi:10.5281/zenodo.11186969 (CC-BY-4.0)
- *Comparative analysis of mechanical properties of alumina and in-office 3D-printed
  zirconia ceramic brackets.* Zenodo. doi:10.5281/zenodo.6372935 (CC-BY-4.0)
