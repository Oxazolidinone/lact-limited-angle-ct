"""Evaluation masks: the whole image (primary) and the body region (secondary)."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def body_mask_np(x_ref: np.ndarray, mu_ref: float, mu_water: float, hu_thresh: float = -500.0,
                 min_size: int = 200) -> np.ndarray:
    """Body mask of one reference image (H, W): HU above a threshold, holes filled, largest component kept."""
    x_thr = mu_water * (1.0 + hu_thresh / 1000.0) / mu_ref
    m = x_ref > x_thr
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0:
        return np.ones_like(x_ref, dtype=bool)
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.argmax(sizes) + 1
    m = lab == keep
    m = ndimage.binary_fill_holes(m)
    if m.sum() < min_size:
        return np.ones_like(x_ref, dtype=bool)
    return m


def body_mask(x_ref: torch.Tensor, mu_ref: float, mu_water: float, hu_thresh: float = -500.0) -> torch.Tensor:
    """Batch (B,1,H,W) -> boolean mask (B,1,H,W)."""
    arr = x_ref.detach().cpu().numpy()
    out = np.stack([body_mask_np(a[0], mu_ref, mu_water, hu_thresh)[None] for a in arr])
    return torch.from_numpy(out).to(x_ref.device)


def fov_circle_mask(n: int, radius_frac: float = 1.0) -> np.ndarray:
    c = (np.arange(n) + 0.5 - n / 2) / (n / 2)
    X, Y = np.meshgrid(c, c)
    return radius_frac ** 2 >= (X ** 2 + Y ** 2)
