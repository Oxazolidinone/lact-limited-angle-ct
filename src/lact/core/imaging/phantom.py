"""Numerical phantoms for operator checks and synthetic data."""
from __future__ import annotations

import numpy as np

# Modified Shepp-Logan (Toft): (intensity, a, b, x0, y0, phi in degrees), normalised coordinates [-1, 1]
_MSL = [
    (1.0, 0.69, 0.92, 0.0, 0.0, 0),
    (-0.8, 0.6624, 0.8740, 0.0, -0.0184, 0),
    (-0.2, 0.1100, 0.3100, 0.22, 0.0, -18),
    (-0.2, 0.1600, 0.4100, -0.22, 0.0, 18),
    (0.1, 0.2100, 0.2500, 0.0, 0.35, 0),
    (0.1, 0.0460, 0.0460, 0.0, 0.1, 0),
    (0.1, 0.0460, 0.0460, 0.0, -0.1, 0),
    (0.1, 0.0460, 0.0230, -0.08, -0.605, 0),
    (0.1, 0.0230, 0.0230, 0.0, -0.606, 0),
    (0.1, 0.0230, 0.0460, 0.06, -0.605, 0),
]


def ellipses_image(n: int, ellipses, supersample: int = 4, extent: float = 1.0) -> np.ndarray:
    """Rasterise a sum of ellipses (intensity, a, b, x0, y0, phi_deg) on an n x n grid.

    Row 0 is at the top (y = +extent) and column 0 on the left (x = -extent), as in ASTRA.
    """
    m = n * supersample
    c = (np.arange(m) + 0.5) / m * 2 * extent - extent
    X, Y = np.meshgrid(c, -c, indexing="xy")
    img = np.zeros((m, m), dtype=np.float64)
    for rho, a, b, x0, y0, phi in ellipses:
        p = np.deg2rad(phi)
        xr = (X - x0) * np.cos(p) + (Y - y0) * np.sin(p)
        yr = -(X - x0) * np.sin(p) + (Y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1.0] += rho
    img = img.reshape(n, supersample, n, supersample).mean(axis=(1, 3))
    return img.astype(np.float32)


def shepp_logan(n: int = 256, supersample: int = 4, scale: float = 0.95) -> np.ndarray:
    """Modified Shepp-Logan, shrunk by ``scale`` to fit inside the field of view."""
    ell = [(r, a * scale, b * scale, x * scale, y * scale, p) for r, a, b, x, y, p in _MSL]
    return np.clip(ellipses_image(n, ell, supersample), 0, None)


def point_phantom(n: int, points: list[tuple[int, int]], value: float = 1.0) -> np.ndarray:
    img = np.zeros((n, n), np.float32)
    for r, c in points:
        img[r, c] = value
    return img


def random_body_phantom(n: int, rng: np.random.Generator, supersample: int = 2) -> np.ndarray:
    """Synthetic abdominal slice: elliptical body (soft tissue ~0.25), fat, organs, bright bones, air pockets.

    Intensities are on the normalised [0, 1] scale of preprocessed Mayo images.
    """
    ell = []
    a, b = rng.uniform(0.70, 0.85), rng.uniform(0.50, 0.65)
    ell.append((0.24, a, b, 0.0, rng.uniform(-0.05, 0.05), rng.uniform(-5, 5)))
    ell.append((0.02, a * 0.92, b * 0.9, 0.0, 0.0, 0))  # soft-tissue layer
    for _ in range(rng.integers(3, 6)):  # organs
        ell.append((rng.uniform(-0.03, 0.05), rng.uniform(0.08, 0.25), rng.uniform(0.08, 0.2),
                    rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3), rng.uniform(0, 180)))
    ell.append((0.35, 0.07, 0.07, 0.0, -b + 0.15, 0))   # vertebra
    ell.append((-0.20, 0.035, 0.035, 0.0, -b + 0.15, 0))
    for sgn in (-1, 1):  # ribs
        for k in range(rng.integers(2, 4)):
            ang = rng.uniform(0.3, 1.2)
            ell.append((0.3, 0.03, 0.02, sgn * a * 0.93 * np.cos(ang), -b * 0.9 * np.sin(ang) * (k % 2 * 2 - 1), 0))
    for _ in range(rng.integers(0, 3)):  # bowel gas
        ell.append((-0.24, rng.uniform(0.02, 0.06), rng.uniform(0.02, 0.05),
                    rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.3), 0))
    img = ellipses_image(n, ell, supersample)
    return np.clip(img, 0.0, 1.0)
