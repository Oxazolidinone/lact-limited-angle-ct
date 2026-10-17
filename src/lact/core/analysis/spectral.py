"""Frequency-domain analysis: fully / partially / un-measured regions and the error energy in each.

Frequency coverage is estimated DIRECTLY from the discrete operator (independent of angle conventions):
the PSF of A^T A at the image centre has spectrum |H(k)| ~ w(phi) / |k| (w = angular coverage). Hence
c(k) = |k| * |H(k)|, normalised to [0, 1], is the coverage map of the geometry.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def coverage_map(projector, n_points: int = 5, spread: int = 4) -> np.ndarray:
    """Coverage map c(k) in [0, 1] (fftshifted frequencies), shape (n, n)."""
    n = projector.img_size
    c0 = n // 2
    offs = [(0, 0), (spread, 0), (-spread, 0), (0, spread), (0, -spread)][:n_points]
    acc = np.zeros((n, n))
    for dr, dc in offs:
        d = torch.zeros(1, 1, n, n, device=projector.device)
        d[0, 0, c0 + dr, c0 + dc] = 1.0
        psf = projector.normal(d)[0, 0].cpu().double().numpy()
        acc += np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psf))))
    acc /= len(offs)
    k = np.fft.fftshift(np.fft.fftfreq(n))
    KX, KY = np.meshgrid(k, k)
    rad = np.hypot(KX, KY)
    c = acc * rad
    band = (rad > 0.05) & (rad < 0.4)
    ref = np.percentile(c[band], 99) if band.any() else c.max()
    return np.clip(c / max(ref, 1e-12), 0.0, 1.0)


def region_masks(cov: np.ndarray, lo: float = 0.1, hi: float = 0.9, k_min: float = 0.01) -> dict[str, np.ndarray]:
    n = cov.shape[0]
    k = np.fft.fftshift(np.fft.fftfreq(n))
    KX, KY = np.meshgrid(k, k)
    valid = np.hypot(KX, KY) >= k_min
    return {"measured": valid & (cov >= hi), "partial": valid & (cov > lo) & (cov < hi),
            "missing": valid & (cov <= lo)}


def error_energy_by_region(pred: np.ndarray, ref: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """Fraction of the error (pred - ref) spectral energy in each region, and the absolute energy."""
    E = np.abs(np.fft.fftshift(np.fft.fft2(pred - ref))) ** 2
    tot = E.sum()
    out = {}
    for name, m in masks.items():
        out[f"frac_{name}"] = float(E[m].sum() / tot) if tot > 0 else float("nan")
        out[f"energy_{name}"] = float(E[m].sum() / E.size)
    return out
