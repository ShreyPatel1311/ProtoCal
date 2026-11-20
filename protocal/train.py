"""Training and evaluation utilities for ProtoCal."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import ProtoCal, ProtoCalConfig


def make_loader(data: dict, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(data["x"], data["p"], data["study"], data["y"], data["mask"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train(
    model: ProtoCal,
    data: dict,
    *,
    epochs: int = 300,
    batch_size: int = 256,
    lr: float = 3e-3,
    weight_decay_proto: float = 1e-2,
    tau_warmup_frac: float = 0.3,
    device: str = "cpu",
    log_every: int = 50,
    verbose: bool = True,
) -> list[dict]:
    """Fit by maximising the penalised marginal likelihood.

    g_proto gets a heavier weight decay than f_shared: that asymmetry is part
    of the identification strategy, not a tuning detail.
    """
    model = model.to(device)
    # The interaction factors are part of the protocol side of the model and
    # carry the same identification burden, so they share g_proto's heavier
    # weight decay rather than f_shared's.
    proto_params = list(model.g_proto.parameters())
    if getattr(model, "rank", 0) > 0:
        proto_params += list(model.u_net.parameters())
        proto_params += list(model.v_net.parameters())
    proto_ids = {id(p) for p in proto_params}
    other = [p for p in model.parameters() if id(p) not in proto_ids]

    opt = torch.optim.AdamW(
        [
            {"params": other, "weight_decay": 1e-4},
            {"params": proto_params, "weight_decay": weight_decay_proto},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = make_loader(data, batch_size)
    # The b_s prior is normalised by the training-set size, not the batch size.
    model.set_total_n_obs(float(data["mask"].sum()))

    history = []
    for epoch in range(epochs):
        model.train()
        tot = n = 0.0
        for x, p, s, y, m in loader:
            x, p, s, y, m = (t.to(device) for t in (x, p, s, y, m))
            opt.zero_grad(set_to_none=True)
            out = model.loss(x, p, s, y, m)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(out["loss"]) * x.shape[0]
            n += x.shape[0]
        sched.step()
        # Empirical-Bayes update for tau in closed form (method of moments).
        # Doing this by gradient descent alongside b_s is what collapsed tau.
        # Held back for a warmup period: updating from the initial b, before
        # the data has moved it, drags tau toward its floor and the shrinkage
        # then prevents b from ever escaping.
        if model.cfg.learn_tau and epoch >= int(tau_warmup_frac * epochs):
            model.update_tau()
        rec = {"epoch": epoch, "loss": tot / max(n, 1), "tau": float(model.log_tau.exp())}
        history.append(rec)
        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  loss {rec['loss']:.4f}  tau {rec['tau']:.3f}")
    return history


@torch.no_grad()
def predict(model: ProtoCal, data: dict, device: str = "cpu", reference: bool = False):
    """Predict either as-reported (with protocol + study terms) or harmonised
    to the reference protocol (composition effect only)."""
    model = model.to(device).eval()
    x = data["x"].to(device)
    if reference:
        return model.predict_reference(x).cpu()
    return model(x, data["p"].to(device), data["study"].to(device)).cpu()


def masked_rmse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-property RMSE over observed entries only."""
    err = ((pred - y) ** 2 * mask).sum(0) / mask.sum(0).clamp(min=1)
    return err.sqrt().numpy()


def grouped_cv_splits(study: torch.Tensor, groups: np.ndarray | None = None, n_folds: int | None = None):
    """Leave-one-study-out (or leave-one-lab-out when `groups` maps study->lab).

    Random k-fold over rows is the wrong test here: it leaks study identity,
    so a model can memorise per-study offsets and look far better than it is.
    """
    s = study.numpy()
    key = groups[s] if groups is not None else s
    uniq = np.unique(key)
    if n_folds is not None and n_folds < len(uniq):
        rng = np.random.default_rng(0)
        buckets = np.array_split(rng.permutation(uniq), n_folds)
    else:
        buckets = [np.array([u]) for u in uniq]
    for held in buckets:
        test = np.isin(key, held)
        yield np.where(~test)[0], np.where(test)[0]
