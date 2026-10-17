"""Batched conjugate gradient (CG) for symmetric positive-definite systems, and the Sobolev metric operator D^T D.

Used by the second-order update of the unrolled models (``update = cg | admm``): each stage solves

    (lam * A^T A + B) d = lam * A^T (y - A z),      x = z + d

Every sample in the batch is its own system (same operator, different right-hand side). All steps are ordinary
tensor operations, so autograd propagates through the CG iterations actually run: the gradient is the EXACT gradient
of the algorithm stopped at that iteration.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

_TINY = 1e-30


def grad_gram(x: torch.Tensor) -> torch.Tensor:
    """D^T D x with D = forward differences and Neumann boundary (the D of ``grad2d``): the negated 5-point Laplacian.

    x: (B,C,H,W). Symmetric positive semi-definite with eigenvalues in [0, 8); constants lie in the kernel.
    Uses only subtraction and zero padding (D^T g = g[j-1] - g[j]) so that its GPU gradient is deterministic
    ("replicate" padding is not).
    """
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return F.pad(dx, (1, 0)) - F.pad(dx, (0, 1)) + F.pad(dy, (0, 0, 1, 0)) - F.pad(dy, (0, 0, 0, 1))


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(1).view(-1, *([1] * (a.dim() - 1)))


def cg_solve(apply_h: Callable[[torch.Tensor], torch.Tensor], rhs: torch.Tensor, iters: int,
             tol: float = 0.0) -> tuple[torch.Tensor, int]:
    """Solve H d = rhs by CG from d = 0. Returns (d, iterations run).

    ``iters``: maximum number of iterations. ``tol`` > 0: stop early once ||residual|| <= tol * ||rhs|| for EVERY
    sample of the batch (0 = always run ``iters`` iterations). Converged samples keep being updated with the batch;
    divisions are bounded below, so a zero residual does not produce NaN.
    """
    d = torch.zeros_like(rhs)
    r = rhs
    p = rhs
    rr = _dot(r, r)
    stop = (float(tol) ** 2) * rr.detach() if tol > 0 else None
    done = 0
    for _ in range(int(iters)):
        hp = apply_h(p)
        alpha = rr / _dot(p, hp).clamp_min(_TINY)
        d = d + alpha * p
        r = r - alpha * hp
        rn = _dot(r, r)
        done += 1
        if stop is not None and bool((rn.detach() <= stop).all()):
            break
        p = r + (rn / rr.clamp_min(_TINY)) * p
        rr = rn
    return d, done
