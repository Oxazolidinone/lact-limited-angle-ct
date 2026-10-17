"""Filter design of the fan-beam FBP: ramp filters in the frequency domain and Parker short-scan weights."""
from __future__ import annotations

import math

import numpy as np

from ...imaging.geometry import FanBeamGeometry

FILTERS = ("ram-lak", "shepp-logan", "cosine", "hann")


def ramp_filter_freq(n_det: int, tau: float, filt: str = "ram-lak", cutoff: float = 1.0) -> np.ndarray:
    """Frequency response (rfft, length P//2+1) of the discrete ramp filter, multiplied by tau."""
    if filt not in FILTERS:
        raise ValueError(f"filter must be one of {FILTERS}")
    P = int(2 ** math.ceil(math.log2(2 * n_det)))
    k = np.arange(P)
    n = np.where(k < P // 2, k, k - P).astype(np.float64)
    h = np.zeros(P, dtype=np.float64)
    h[n == 0] = 1.0 / (4.0 * tau ** 2)
    odd = (np.abs(n) % 2) == 1
    h[odd] = -1.0 / (np.pi * n[odd] * tau) ** 2
    H = np.real(np.fft.rfft(h)) * tau
    f = np.fft.rfftfreq(P)  # 0 .. 0.5 (cycles / sample)
    fc = 0.5 * cutoff
    if filt == "shepp-logan":
        H *= np.sinc(f / (2.0 * fc))
    elif filt == "cosine":
        H *= np.cos(np.pi * f / (2.0 * fc))
    elif filt == "hann":
        H *= 0.5 * (1.0 + np.cos(np.pi * f / fc))
    H[f > fc] = 0.0
    return H


def parker_weights(geom: FanBeamGeometry, sign: float = 1.0) -> np.ndarray:
    """Parker weights (V, n_det) for short scans; delta = (span - 180°)/2."""
    D = geom.dso_mm
    s = geom.det_coords_mm() * D / geom.dsd_mm
    gamma = sign * np.arctan(s / D)[None, :]
    delta = 0.5 * (math.radians(geom.span_deg) - math.pi)
    if delta <= 0:
        raise ValueError("Parker weights apply only when span > 180°")
    bp = ((np.arange(geom.n_views) + 0.5) * math.radians(geom.step_deg))[:, None]
    w = np.ones((geom.n_views, geom.n_det))
    r1 = bp < (2 * delta - 2 * gamma)
    r3 = bp > (math.pi - 2 * gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        w1 = np.sin(0.25 * math.pi * bp / (delta - gamma)) ** 2
        w3 = np.sin(0.25 * math.pi * (math.pi + 2 * delta - bp) / (delta + gamma)) ** 2
    w = np.where(r1, w1, w)
    w = np.where(r3, w3, w)
    w = np.where(bp > math.pi + 2 * delta, 0.0, w)
    return np.nan_to_num(w, nan=0.0).astype(np.float32)
