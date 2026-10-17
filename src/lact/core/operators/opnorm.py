"""Spectral norm ||A||_2 estimated by power iteration on A^T A."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def power_iteration(matvec: Callable[[np.ndarray], np.ndarray],
                    rmatvec: Callable[[np.ndarray], np.ndarray],
                    n: int, iters: int = 200, tol: float = 1e-5, seed: int = 0,
                    dtype=np.float64) -> tuple[float, int]:
    """Return (||A||_2, iterations). Stops when the relative change of ||A||^2 drops below ``tol``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(dtype)
    x /= np.linalg.norm(x)
    lam_old = 0.0
    it = 0
    for it in range(1, iters + 1):
        z = rmatvec(matvec(x)).astype(dtype, copy=False)
        lam = float(np.linalg.norm(z.astype(np.float64)))
        if lam == 0.0:
            return 0.0, it
        x = z / lam
        if lam_old > 0 and abs(lam - lam_old) / lam < tol:
            break
        lam_old = lam
    return float(np.sqrt(lam)), it


def opnorm_sparse(A, iters: int = 200, tol: float = 1e-5, seed: int = 0) -> tuple[float, int]:
    """||A||_2 of a scipy.sparse matrix."""
    AT = A.T.tocsr()
    dt = np.float32 if A.dtype == np.float32 else np.float64
    return power_iteration(lambda v: A @ v, lambda v: AT @ v, A.shape[1], iters, tol, seed, dtype=dt)


def opnorm_torch(forward, adjoint, shape, iters: int = 200, tol: float = 1e-5, seed: int = 0,
                 device="cpu") -> tuple[float, int]:
    """||A||_2 of a PyTorch forward/adjoint operator pair acting on inputs of shape ``shape``."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64).to(device)
    x = x / x.norm()
    lam_old, it = 0.0, 0
    for it in range(1, iters + 1):
        z = adjoint(forward(x.float())).double()
        lam = float(z.norm())
        if lam == 0.0:
            return 0.0, it
        x = z / lam
        if lam_old > 0 and abs(lam - lam_old) / lam < tol:
            break
        lam_old = lam
    return float(np.sqrt(lam)), it
