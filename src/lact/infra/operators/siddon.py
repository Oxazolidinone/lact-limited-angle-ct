"""Fan-beam projection matrix with the line model (Siddon 1985), pure NumPy, no ASTRA needed.

Entry A[r, p] = intersection length (mm) of ray r (source -> detector pixel centre) with pixel p, exactly ASTRA's
``line_fanflat`` model (verified by a direct comparison test when ASTRA is installed). Used on machines where ASTRA
cannot be installed (e.g. Linux aarch64 / Apple Silicon).

Conventions (as in ``astra_matrix``): row = view * n_det + channel; column = image row * n + image column;
column j -> x = (j + 0.5 - n/2) px, row i -> y = (n/2 - i - 0.5) px; geometry vectors from ``FanBeamGeometry.vectors``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ...core.imaging.geometry import FanBeamGeometry


def _view_entries(src: np.ndarray, det: np.ndarray, n: int, h: float, px: float):
    """Entries (ray, pixel index, length) of one view. src: (2,), det: (D, 2) detector pixel centres."""
    sx, sy = float(src[0]), float(src[1])
    rx, ry = det[:, 0] - sx, det[:, 1] - sy
    L = np.hypot(rx, ry)
    grid = -h + px * np.arange(n + 1)
    D = det.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = (grid[None, :] - sx) / rx[:, None]
        ty = (grid[None, :] - sy) / ry[:, None]
    # axis-parallel rays cross the box only when their fixed coordinate lies inside (-h, h)
    zx, zy = rx == 0, ry == 0
    inx = (sx > -h) & (sx < h)
    iny = (sy > -h) & (sy < h)
    tx_lo = np.where(zx, -np.inf if inx else np.inf, np.minimum(tx[:, 0], tx[:, -1]))
    tx_hi = np.where(zx, np.inf if inx else -np.inf, np.maximum(tx[:, 0], tx[:, -1]))
    ty_lo = np.where(zy, -np.inf if iny else np.inf, np.minimum(ty[:, 0], ty[:, -1]))
    ty_hi = np.where(zy, np.inf if iny else -np.inf, np.maximum(ty[:, 0], ty[:, -1]))
    tmin = np.maximum.reduce([tx_lo, ty_lo, np.zeros(D)])
    tmax = np.minimum.reduce([tx_hi, ty_hi, np.ones(D)])
    T = np.concatenate([tx, ty], axis=1)
    T = np.where((tmin[:, None] < T) & (tmax[:, None] > T), T, np.nan)
    T = np.concatenate([tmin[:, None], T, tmax[:, None]], axis=1)
    T.sort(axis=1)                                            # NaNs move to the end
    dT = np.diff(T, axis=1)
    mid = T[:, :-1] + 0.5 * dT
    xm = sx + mid * rx[:, None]
    ym = sy + mid * ry[:, None]
    col = np.floor((xm + h) / px)
    row = np.floor((h - ym) / px)
    ok = (np.isfinite(dT) & (dT > 0) & (tmax > tmin)[:, None]
          & (col >= 0) & (col < n) & (row >= 0) & (row < n))
    ray = np.broadcast_to(np.arange(D)[:, None], ok.shape)[ok]
    pix = (row[ok] * n + col[ok]).astype(np.int64)
    w = (dT * L[:, None])[ok]
    return ray, pix, w


def build_matrix_siddon(geom: FanBeamGeometry, img_size: int, dtype=np.float32) -> sp.csr_matrix:
    """A (V*n_det, n*n), CSR, entries = intersection lengths (mm)."""
    n = int(img_size)
    h = 0.5 * geom.fov_mm
    px = geom.fov_mm / n
    vec = geom.vectors()
    off = np.arange(geom.n_det) - 0.5 * (geom.n_det - 1)
    parts = []
    for v in range(geom.n_views):
        sx, sy, dx, dy, ux, uy = vec[v]
        det = np.stack([dx + off * ux, dy + off * uy], axis=1)
        ray, pix, w = _view_entries(np.array([sx, sy]), det, n, h, px)
        parts.append((ray + v * geom.n_det, pix, w))
    rows = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts]).astype(dtype)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(geom.n_views * geom.n_det, n * n), dtype=dtype)
    A.sum_duplicates()
    A.eliminate_zeros()
    return A
