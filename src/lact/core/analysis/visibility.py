"""Local microlocal visibility operator: splits the error into measured / unmeasured directions.

Basis (Quinto 1993; Frikel & Quinto 2013): a singularity (edge) at point p with normal direction xi is reconstructed
stably from X-ray data if and only if the line through p perpendicular to xi is a MEASURED ray. In fan-beam geometry
the rays through p have directions psi(beta, p) = direction of (p - s(beta)), beta in the measured angle set Theta.
The set of visible frequency directions at p is therefore

    V(p) = { phi : phi = psi(beta, p) + pi/2 (mod pi), beta in Theta }.

V(p) depends on position (fan-beam is not shift-invariant). We build:
  * a Parseval directional frame T: T x = {F^-1 H_j F x}_j (j = 0..J) with sum_j H_j^2 = 1, so T^H T = I;
  * a gate M_V = diag(m_j(p)), m_j(p) = |V(p) cap B_j| / |B_j| in [0, 1] (B_j: the j-th direction bin);
  * P_vis = T^H M_V T (self-adjoint, 0 <= P_vis <= I), P_inv = I - P_vis.
P_vis is NOT idempotent (redundant frame, per-pixel gate), hence "visibility operator" rather than projection.
The low-frequency band (|k| < k0, no defined direction) counts as visible (DC lies on every Fourier slice).

Used in the result analysis: <e, P_vis e> and <e, P_inv e> are the error in measured / unmeasured directions
(summing to ||e||^2), see experiments/figures/export_results.py.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from ..imaging.geometry import FanBeamGeometry

DC_RADIUS = 1.5     # radius (in frequency samples) of the low band treated as visible
FINE = 8            # fine bins per direction bin when computing coverage fractions


# ---------------------------------------------------------------------------- geometry -> gate
def visibility_gates(geom: FanBeamGeometry, img_size: int, n_dirs: int, fine: int = FINE) -> np.ndarray:
    """Gate m_j(p) in [0, 1], shape (n_dirs, n, n), computed analytically from the source positions.

    Frequency directions use physical coordinates (x to the right, y up) with the ASTRA/FBP image grid convention:
    column j -> x = (j + 0.5 - n/2) px, row i -> y = (n/2 - i - 0.5) px.
    """
    n = int(img_size)
    px = geom.pixel_size_mm(n)
    c = (np.arange(n) + 0.5 - n / 2) * px
    Y, X = np.meshgrid(-c, c, indexing="ij")
    nb = n_dirs * fine
    hit = np.zeros((nb, n, n), dtype=bool)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cos_fan = math.cos(math.radians(geom.fan_half_angle_deg))

    def bins_of(sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
        dx, dy = X - sx, Y - sy
        # the source -> pixel ray must lie inside the fan (reach the detector): angle to the central ray (-s) <= gamma_m
        inside = (dx * -sx + dy * -sy) >= cos_fan * np.hypot(dx, dy) * math.hypot(sx, sy)
        phi = np.mod(np.arctan2(dy, dx) + np.pi / 2, np.pi)              # measured frequency direction per pixel
        return np.minimum((phi * (nb / np.pi)).astype(np.int64), nb - 1), inside

    # Theta is a CONTINUOUS arc: between two adjacent views the measured direction sweeps a small arc, so the fine bins
    # in between are filled (avoids holes from angular discretisation; sparse sampling is a different issue).
    src = geom.vectors()[:, :2]
    pairs = list(zip(range(len(src) - 1), range(1, len(src))))
    if geom.is_full_scan:
        pairs.append((len(src) - 1, 0))
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, j in pairs:
        bi, ini = cache.pop(i) if i in cache else bins_of(*src[i])
        bj, inj = bins_of(*src[j])
        cache[j] = (bj, inj)
        both = ini & inj
        d = np.mod(bj - bi + nb // 2, nb) - nb // 2                          # signed bin offset along the short way
        for t in range(int(np.abs(d[both]).max(initial=0)) + 1):
            m = both & (np.abs(d) >= t)
            b = np.mod(bi + np.sign(d) * t, nb)
            hit[b[m], rows[m], cols[m]] = True
        for bb, mm in ((bi, ini & ~inj), (bj, inj & ~ini)):                  # fan edge: only one of the two rays
            hit[bb[mm], rows[mm], cols[mm]] = True
    if len(src) == 1:
        b, ins = bins_of(*src[0])
        hit[b[ins], rows[ins], cols[ins]] = True
    return hit.reshape(n_dirs, fine, n, n).mean(axis=1).astype(np.float32)


@lru_cache(maxsize=16)
def _gates_cached(geom_key: tuple, img_size: int, n_dirs: int) -> np.ndarray:
    return visibility_gates(FanBeamGeometry.from_dict(dict(geom_key)), img_size, n_dirs)


def gates_for(geom: FanBeamGeometry, img_size: int, n_dirs: int) -> np.ndarray:
    return _gates_cached(tuple(sorted(geom.to_dict().items())), int(img_size), int(n_dirs))


# ---------------------------------------------------------------------------- Parseval directional frame
@lru_cache(maxsize=8)
def directional_frame(img_size: int, n_dirs: int) -> np.ndarray:
    """Filters (n_dirs + 1, n, n//2 + 1) on the rfft2 grid: H_0 low-pass, H_1..H_J directional; sum_j H_j^2 = 1."""
    n = int(img_size)
    kr = np.fft.fftfreq(n)[:, None]                 # along rows (the y axis is flipped)
    kc = np.fft.rfftfreq(n)[None, :]                # along columns (x axis)
    kx, ky = np.broadcast_to(kc, (n, kc.shape[1])), np.broadcast_to(-kr, (n, kc.shape[1]))
    rad = np.hypot(kx, ky) * n                      # units: frequency samples
    phi = np.mod(np.arctan2(ky, kx), np.pi)
    # smooth low-pass: rho = 1 within DC_RADIUS, cosine roll-off to 0 at 2*DC_RADIUS
    t = np.clip((rad - DC_RADIUS) / DC_RADIUS, 0.0, 1.0)
    rho = np.cos(0.5 * np.pi * t)
    width = np.pi / n_dirs
    w = np.zeros((n_dirs,) + phi.shape)
    for j in range(n_dirs):
        d = np.abs(np.mod(phi - (j + 0.5) * width + np.pi / 2, np.pi) - np.pi / 2)
        w[j] = np.where(d <= width, np.cos(0.5 * np.pi * d / width) ** 2, 0.0)
    w /= w.sum(axis=0, keepdims=True)               # partition of unity over directions
    # The rfft grid keeps half the spectrum: in the k_x = 0 and Nyquist columns, (k_y, k_x) and (-k_y, k_x) are a
    # conjugate pair, so the filter must agree at both points (otherwise irfft2 breaks Parseval). Average the two.
    mirror = (-np.arange(n)) % n
    cols = [0] + ([kc.shape[1] - 1] if n % 2 == 0 else [])
    for col in cols:
        w[:, :, col] = 0.5 * (w[:, :, col] + w[:, mirror, col])
    H = np.empty((n_dirs + 1,) + phi.shape)
    H[0] = rho
    H[1:] = np.sqrt(w * (1.0 - rho ** 2)[None])
    return H.astype(np.float32)


class VisibilityOperator(torch.nn.Module):
    """P_vis = T^H M_V T on images (B, 1, n, n); filters and gate depend on the geometry only."""

    H: torch.Tensor
    M: torch.Tensor

    def __init__(self, geom: FanBeamGeometry, img_size: int, n_dirs: int = 16, device=None):
        super().__init__()
        self.img_size, self.n_dirs = int(img_size), int(n_dirs)
        self.geom_name = geom.name
        H = torch.from_numpy(directional_frame(self.img_size, self.n_dirs))
        G = torch.from_numpy(gates_for(geom, self.img_size, self.n_dirs))
        M = torch.cat([torch.ones(1, self.img_size, self.img_size), G], dim=0)   # low band: visible
        self.register_buffer("H", H.to(device) if device is not None else H, persistent=False)
        self.register_buffer("M", M.to(device) if device is not None else M, persistent=False)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        """T x: (B, 1, n, n) -> (B, J+1, n, n)."""
        X = torch.fft.rfft2(x.float())
        return torch.fft.irfft2(X * self.H[None], s=x.shape[-2:])

    def synthesis(self, c: torch.Tensor) -> torch.Tensor:
        """T^H c: (B, J+1, n, n) -> (B, 1, n, n)."""
        C = torch.fft.rfft2(c)
        return torch.fft.irfft2((C * self.H[None]).sum(dim=1, keepdim=True), s=c.shape[-2:])

    def vis(self, x: torch.Tensor) -> torch.Tensor:
        return self.synthesis(self.M[None] * self.analysis(x)).to(x.dtype)

    def inv(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.vis(x)
