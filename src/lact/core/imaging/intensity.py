"""CT intensity conversions (HU <-> mu <-> normalised [0, 1] scale) and image resizing. Pure functions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hu_to_mu(hu, mu_water: float):
    return mu_water * (1.0 + hu / 1000.0)


def mu_to_hu(mu, mu_water: float):
    return (mu / mu_water - 1.0) * 1000.0


def x_to_hu(x, mu_ref: float, mu_water: float):
    """Normalised image -> HU (for display windows and HU-scale RMSE)."""
    return (x * mu_ref / mu_water - 1.0) * 1000.0


def hu_to_x(hu, mu_ref: float, mu_water: float):
    return mu_water * (1.0 + hu / 1000.0) / mu_ref


def normalize_mu(mu: np.ndarray, mu_ref: float) -> np.ndarray:
    return np.clip(mu / mu_ref, 0.0, 1.0).astype(np.float32)


def resize_square(img: np.ndarray, n: int) -> np.ndarray:
    """Resize an (N,H,W) or (H,W) image to n x n (area when shrinking, bilinear when enlarging)."""
    squeeze = img.ndim == 2
    t = torch.from_numpy(img[None] if squeeze else img).float().unsqueeze(1)
    if t.shape[-1] == n and t.shape[-2] == n:
        out = t
    elif t.shape[-1] > n:
        out = F.interpolate(t, size=(n, n), mode="area")
    else:
        out = F.interpolate(t, size=(n, n), mode="bilinear", align_corners=False)
    arr = out[:, 0].numpy()
    return arr[0] if squeeze else arr


def block_mean(img: np.ndarray, factor: int) -> np.ndarray:
    """Mean over factor x factor blocks of the last two axes (preserves the integral)."""
    *lead, h, w = img.shape
    return img.reshape(*lead, h // factor, factor, w // factor, factor).mean(axis=(-3, -1))
