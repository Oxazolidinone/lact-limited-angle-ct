"""Effective receptive field (ERF) of a regularizer: |d R(x)[centre] / d x| averaged over several images."""
from __future__ import annotations

import numpy as np
import torch


def effective_receptive_field(reg: torch.nn.Module, images: torch.Tensor, device="cpu") -> np.ndarray:
    """images: (N,1,H,W). Returns the ERF map (H,W) normalised to unit sum."""
    reg = reg.to(device).eval()
    N, _, H, W = images.shape
    acc = torch.zeros(H, W, dtype=torch.float64)
    for i in range(N):
        x = images[i: i + 1].to(device).clone().requires_grad_(True)
        out = reg(x)
        out[0, 0, H // 2, W // 2].backward()
        assert x.grad is not None
        acc += x.grad[0, 0].abs().double().cpu()
    erf = acc.numpy()
    s = erf.sum()
    return erf / s if s > 0 else erf


def erf_radius(erf: np.ndarray, mass: float = 0.9) -> float:
    """Radius (pixels) around the centre that holds a fraction ``mass`` of the total ERF mass."""
    H, W = erf.shape
    yy, xx = np.mgrid[:H, :W]
    r = np.hypot(yy - H // 2, xx - W // 2).ravel()
    order = np.argsort(r)
    cum = np.cumsum(erf.ravel()[order])
    k = np.searchsorted(cum, mass * cum[-1])
    return float(r[order][min(k, len(r) - 1)])
