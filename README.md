# ProtoCal

**Protocol-aware hierarchical modelling for literature-scattered dental composite data.**

The bottleneck in ML for dental biomaterials is not data volume but *uncontrolled
heterogeneity*: published studies differ in specimen geometry, storage medium, aging
regime, crosshead speed and curing protocol. Pooling them naively confounds composition
signal with protocol effects. ProtoCal models the protocol explicitly:

```
y_{i,s} = f_shared(x_i) + g_proto(p_s) + b_s + eps_s
```

with `g_proto` **gauge-fixed** to zero at a reference protocol, which is what makes the
decomposition identifiable when each study reports under a single protocol.

## Status

Early. The model runs, the synthetic precondition check passes for `g_proto`, and a real
open dataset has been audited. Two results already constrain the design — see
[`output/DATA_AUDIT_FINDINGS.md`](output/DATA_AUDIT_FINDINGS.md):

- **Protocol explains up to 32%** of variance in flexural strength and Vickers hardness
  across 1,802 specimen measurements — the premise holds.
- **The additive `g_proto` is mis-specified for flexural strength**: the material ×
  protocol interaction is 1.42× measurement noise. A low-rank interaction term is needed.
- **`b_s` is not recoverable** at literature data density (r ≈ 0.31); the per-study
  "data-quality signal" idea does not survive contact with the numbers.

## Quick start

```bash
uv sync --python 3.12
python scripts/download_data.py     # CC-BY datasets, MD5-verified
python scripts/audit_data.py        # corpus audit + additivity test
python scripts/smoke_test.py        # synthetic recovery, GPU
```

## Layout

| Path | What |
|---|---|
| `protocal/model.py` | gauge-fixed hierarchical model |
| `protocal/synthetic.py` | synthetic-recovery precondition check + leakage diagnostic |
| `protocal/train.py` | training, grouped CV (leave-one-study/lab-out) |
| `scripts/` | download, audit, distribution check, sweeps |
| `output/` | generated reports and results |
| `paper/` | manuscript |

## Provenance

This repository is derived from IRIS (ACL 2025 demo), which was used to generate the
research idea. See [`NOTICE.md`](NOTICE.md) for attribution and licence status, and
[`output/iris_run1_refined.md`](output/iris_run1_refined.md) for the generated proposal.
