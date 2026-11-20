"""ProtoCal: protocol-aware hierarchical model for literature-scattered
dental-composite property data.

For a property value y of formulation i reported by study s:

    y_{i,s} = f_shared(x_i) + g_proto(p_s) + b_s + eps_s

  f_shared  shared composition -> property function (multi-output)
  g_proto   structured protocol correction, *gauge-fixed* so that
            g_proto(p_reference) == 0 exactly
  b_s       per-study free offset under a learned N(0, tau^2) shrinkage prior
  eps_s     per-study heteroscedastic noise (learned log-variance)

The gauge fixing is what makes the decomposition identifiable: because every
study typically reports under a single protocol, g_proto(p_s) and b_s are both
per-study additive constants and would otherwise trade off freely. Pinning
g_proto to zero at a reference protocol removes that additive ambiguity and
makes f_shared directly interpretable as the reference-protocol prediction.

Identification further relies on an asymmetry of capacity, enforced here:
g_proto is a small, weight-decayed function shared across all studies, while
b_s is an independent scalar per study under strong shrinkage. Any between-
study variation explainable by observed protocol covariates is therefore
cheaper to put in g_proto than in b_s.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ProtoCalConfig:
    n_composition: int
    n_protocol: int
    n_studies: int
    n_properties: int
    shared_hidden: tuple[int, ...] = (128, 128)
    proto_hidden: tuple[int, ...] = (16,)
    dropout: float = 0.1
    # Strength of the zero-mean prior on b_s. Learned (empirical Bayes) when
    # learn_tau is True; otherwise fixed at this value.
    init_log_tau: float = -1.0
    learn_tau: bool = True
    init_log_noise: float = -1.0


def _mlp(sizes: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        if idx < len(sizes) - 2:
            layers.append(nn.SiLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ProtoCal(nn.Module):
    """Hierarchical protocol-conditioned regressor with per-study effects."""

    def __init__(self, cfg: ProtoCalConfig):
        super().__init__()
        self.cfg = cfg

        self.f_shared = _mlp(
            [cfg.n_composition, *cfg.shared_hidden, cfg.n_properties], cfg.dropout
        )
        # Deliberately low capacity, and no dropout: g_proto must stay a smooth,
        # strongly-regularised function of observed protocol covariates.
        self.g_proto = _mlp(
            [cfg.n_protocol, *cfg.proto_hidden, cfg.n_properties], dropout=0.0
        )

        # Per-study offsets, one scalar per (study, property).
        self.b = nn.Parameter(torch.zeros(cfg.n_studies, cfg.n_properties))
        # Per-study heteroscedastic observation noise.
        self.log_noise = nn.Parameter(
            torch.full((cfg.n_studies, cfg.n_properties), cfg.init_log_noise)
        )
        # Prior scale on b_s (empirical Bayes when learn_tau).
        self.log_tau = nn.Parameter(
            torch.tensor(float(cfg.init_log_tau)), requires_grad=cfg.learn_tau
        )

        # Reference protocol vector; g_proto is gauge-fixed to zero here.
        self.register_buffer("p_reference", torch.zeros(1, cfg.n_protocol))

    def set_reference_protocol(self, p_ref: torch.Tensor) -> None:
        """Define the protocol at which g_proto is pinned to zero."""
        self.p_reference = p_ref.reshape(1, -1).to(self.p_reference.device).float()

    def protocol_effect(self, p: torch.Tensor) -> torch.Tensor:
        """Gauge-fixed protocol correction: exactly zero at p_reference."""
        return self.g_proto(p) - self.g_proto(self.p_reference)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        study: torch.Tensor,
        *,
        include_study_offset: bool = True,
    ) -> torch.Tensor:
        out = self.f_shared(x) + self.protocol_effect(p)
        if include_study_offset:
            out = out + self.b[study]
        return out

    def predict_reference(self, x: torch.Tensor) -> torch.Tensor:
        """Harmonised prediction: what this formulation would measure under the
        reference protocol, with study idiosyncrasy removed."""
        return self.f_shared(x)

    def loss(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        study: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Negative log marginal likelihood with the b_s shrinkage prior.

        `mask` is 1 where a property was actually reported. Literature records
        are sparse and missingness is informative, so the loss is computed over
        observed entries only -- no imputation.
        """
        pred = self.forward(x, p, study)
        log_var = self.log_noise[study]
        inv_var = torch.exp(-log_var)

        sq = (y - pred) ** 2
        nll = 0.5 * (log_var + sq * inv_var) * mask
        n_obs = mask.sum().clamp(min=1.0)
        nll = nll.sum() / n_obs

        # Zero-mean Gaussian prior on the per-study offsets. Studies that appear
        # in this batch only; tau is shared.
        tau2 = torch.exp(2.0 * self.log_tau)
        used = torch.unique(study)
        b_used = self.b[used]
        prior = 0.5 * (b_used**2 / tau2).sum() / n_obs
        prior = prior + 0.5 * used.numel() * self.cfg.n_properties * (
            2.0 * self.log_tau
        ) / n_obs

        return {"loss": nll + prior, "nll": nll, "prior": prior}
