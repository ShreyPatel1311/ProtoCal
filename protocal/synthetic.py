"""Synthetic-recovery validation for ProtoCal.

This is a *precondition check*, not an optional diagnostic. Before any claim is
made from a real-data decomposition, we generate data from a known f_shared,
g_proto and b_s -- with the sparsity, study-size imbalance and single-protocol-
per-study structure of real literature data -- refit, and measure how well the
three terms are recovered. If recovery fails here it cannot be trusted there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SyntheticSpec:
    n_studies: int = 60
    n_properties: int = 4
    n_composition: int = 8
    n_protocol: int = 5
    min_rows_per_study: int = 2
    max_rows_per_study: int = 25
    # Probability that any given property is reported for a given formulation.
    # Real literature is far sparser for some properties than others.
    report_prob: tuple[float, ...] = (0.85, 0.45, 0.30, 0.25)
    b_scale: float = 0.35
    noise_scale: float = 0.18
    # Fraction of studies that reuse the exact reference protocol.
    frac_reference_protocol: float = 0.12
    # Strength of a material x protocol interaction in the generating process.
    # 0.0 reproduces the purely additive world the original model assumed; the
    # real Lodz corpus shows a non-zero interaction for flexural strength, so
    # the recovery check should be run at both settings.
    interaction_scale: float = 0.0
    interaction_rank: int = 2
    seed: int = 0


def make_synthetic(spec: SyntheticSpec):
    """Return tensors plus the ground-truth functions used to generate them."""
    rng = np.random.default_rng(spec.seed)

    # --- ground-truth generators -------------------------------------------
    W1 = rng.normal(0, 1.0, (spec.n_composition, 32))
    W2 = rng.normal(0, 0.8, (32, spec.n_properties))

    def f_true(x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ W1) @ W2

    # g_proto is deliberately smooth and low-rank in the protocol covariates:
    # this is the structure ProtoCal assumes and must be able to find.
    G1 = rng.normal(0, 0.9, (spec.n_protocol, 8))
    G2 = rng.normal(0, 0.6, (8, spec.n_properties))

    def g_raw(p: np.ndarray) -> np.ndarray:
        return np.tanh(p @ G1) @ G2

    p_ref = np.zeros((1, spec.n_protocol))

    def g_true(p: np.ndarray) -> np.ndarray:
        return g_raw(p) - g_raw(p_ref)  # gauge-fixed, same as the model

    # Material x protocol interaction, also gauge-fixed so that the entire
    # protocol contribution vanishes at the reference protocol.
    r = spec.interaction_rank
    U = rng.normal(0, 1.0, (spec.n_composition, r * spec.n_properties))
    V = rng.normal(0, 1.0, (spec.n_protocol, r * spec.n_properties))

    def inter_true(x: np.ndarray, p: np.ndarray) -> np.ndarray:
        if spec.interaction_scale == 0.0:
            return np.zeros((x.shape[0], spec.n_properties))
        u = np.tanh(x @ U).reshape(-1, r, spec.n_properties)
        v = (np.tanh(p @ V) - np.tanh(p_ref @ V)).reshape(-1, r, spec.n_properties)
        return spec.interaction_scale * (u * v).sum(1)

    # --- per-study protocols and offsets -----------------------------------
    protocols = rng.normal(0, 1.0, (spec.n_studies, spec.n_protocol))
    n_ref = max(1, int(spec.frac_reference_protocol * spec.n_studies))
    protocols[:n_ref] = 0.0  # these studies used the reference protocol

    b_true = rng.normal(0, spec.b_scale, (spec.n_studies, spec.n_properties))
    noise_sd = np.abs(
        rng.normal(spec.noise_scale, spec.noise_scale * 0.4, (spec.n_studies, 1))
    ).clip(0.03)

    xs, ps, ss, ys, ms = [], [], [], [], []
    for s in range(spec.n_studies):
        n_rows = rng.integers(spec.min_rows_per_study, spec.max_rows_per_study + 1)
        x = rng.normal(0, 1.0, (n_rows, spec.n_composition))
        p = np.repeat(protocols[s : s + 1], n_rows, axis=0)  # one protocol per study
        y = f_true(x) + g_true(p) + inter_true(x, p) + b_true[s]
        y = y + rng.normal(0, noise_sd[s], y.shape)
        m = (rng.random(y.shape) < np.array(spec.report_prob)).astype(np.float32)
        # keep rows that report at least one property
        keep = m.sum(1) > 0
        xs.append(x[keep]); ps.append(p[keep]); ys.append(y[keep]); ms.append(m[keep])
        ss.append(np.full(keep.sum(), s))

    data = {
        "x": torch.tensor(np.concatenate(xs), dtype=torch.float32),
        "p": torch.tensor(np.concatenate(ps), dtype=torch.float32),
        "study": torch.tensor(np.concatenate(ss), dtype=torch.long),
        "y": torch.tensor(np.concatenate(ys), dtype=torch.float32),
        "mask": torch.tensor(np.concatenate(ms), dtype=torch.float32),
    }
    truth = {
        "b": torch.tensor(b_true, dtype=torch.float32),
        "protocols": torch.tensor(protocols, dtype=torch.float32),
        "g_at_protocols": torch.tensor(g_true(protocols), dtype=torch.float32),
        "p_reference": torch.tensor(p_ref, dtype=torch.float32),
        "f_true": f_true,
        "g_true": g_true,
        "inter_true": inter_true,
    }
    return data, truth


def recovery_metrics(model, truth) -> dict[str, float]:
    """Correlation and RMSE between fitted and true g_proto / b_s."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        g_hat = model.protocol_effect(truth["protocols"].to(dev)).cpu()
    g_true = truth["g_at_protocols"]
    b_hat = model.b.detach().cpu()
    b_true = truth["b"]

    def corr(a: torch.Tensor, b: torch.Tensor) -> float:
        a, b = a.flatten(), b.flatten()
        a = a - a.mean(); b = b - b.mean()
        denom = (a.norm() * b.norm()).clamp(min=1e-12)
        return float((a @ b) / denom)

    return {
        "g_corr": corr(g_hat, g_true),
        "g_rmse": float(((g_hat - g_true) ** 2).mean().sqrt()),
        "b_corr": corr(b_hat, b_true),
        "b_rmse": float(((b_hat - b_true) ** 2).mean().sqrt()),
    }


def leakage_diagnostic(model, truth) -> float:
    """Post-hoc check: regress fitted b_s on the protocol covariates.

    If g_proto has absorbed the explainable protocol signal, b_s should carry
    essentially none of it and the R^2 here should be near zero. A high value
    means protocol effects are leaking into the per-study offsets -- which
    would also invalidate reading b_s as a data-quality signal.
    """
    P = truth["protocols"].cpu().numpy()
    B = model.b.detach().cpu().numpy()
    P = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(P, B, rcond=None)
    resid = B - P @ coef
    ss_res = float((resid**2).sum())
    ss_tot = float(((B - B.mean(0)) ** 2).sum()) or 1.0
    return 1.0 - ss_res / ss_tot
