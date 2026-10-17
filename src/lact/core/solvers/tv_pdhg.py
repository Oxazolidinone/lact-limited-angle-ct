"""TV reconstruction with PDHG (Chambolle & Pock, 2011) on the same A_256 operator as the learned models.

    min_x  0.5 * ||A_n x - y_n||^2 + lam * TV_iso(x)   subject to  x >= 0

A_n = A / ||A||, y_n = y / ||A|| (as in the models). Uses K = [A_n; mu * grad] with mu = 1/sqrt(8)
(||K||^2 <= 2) to allow larger tau, sigma steps; the problem is equivalent with TV threshold lam / mu.
"""
from __future__ import annotations

import torch


def grad2d(x: torch.Tensor) -> torch.Tensor:
    """Forward differences with Neumann boundary. x: (B,1,H,W) -> (B,2,H,W)."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return torch.cat([gx, gy], dim=1)


def div2d(p: torch.Tensor) -> torch.Tensor:
    """Divergence = -grad^T. p: (B,2,H,W) -> (B,1,H,W)."""
    px, py = p[:, :1], p[:, 1:2]
    dx = torch.zeros_like(px)
    dy = torch.zeros_like(py)
    dx[..., :, 0] = px[..., :, 0]
    dx[..., :, 1:-1] = px[..., :, 1:-1] - px[..., :, :-2]
    dx[..., :, -1] = -px[..., :, -2]
    dy[..., 0, :] = py[..., 0, :]
    dy[..., 1:-1, :] = py[..., 1:-1, :] - py[..., :-2, :]
    dy[..., -1, :] = -py[..., -2, :]
    return dx + dy


def tv_value(x: torch.Tensor) -> torch.Tensor:
    g = grad2d(x)
    return g.pow(2).sum(1).sqrt().flatten(1).sum(1)


@torch.no_grad()
def tv_pdhg(projector, y_raw: torch.Tensor, lam: float, n_iter: int = 500, x_init: torch.Tensor | None = None,
            record_at: list[int] | None = None, nonneg: bool = True, theta: float = 1.0):
    """Return the final x (and a dict {iteration: x} when ``record_at`` is given)."""
    y = projector.normalize_data(y_raw)
    B = y.shape[0]
    H = W = projector.img_size
    x = x_init.clone() if x_init is not None else torch.zeros(B, 1, H, W, device=y.device)
    xbar = x.clone()
    p = torch.zeros_like(y)
    q = torch.zeros(B, 2, H, W, device=y.device)
    mu = 8.0 ** -0.5
    L = (1.0 + 8.0 * mu * mu) ** 0.5
    tau = sigma = 0.99 / L
    radius = lam / mu
    rec = {}
    record_at = sorted(set(record_at or []))
    for it in range(1, n_iter + 1):
        p = (p + sigma * (projector.forward(xbar) - y)) / (1.0 + sigma)
        q = q + sigma * mu * grad2d(xbar)
        nq = q.pow(2).sum(1, keepdim=True).sqrt()
        q = q / torch.clamp(nq / radius, min=1.0)
        x_new = x - tau * (projector.adjoint(p) - mu * div2d(q))
        if nonneg:
            x_new = x_new.clamp_min(0.0)
        xbar = x_new + theta * (x_new - x)
        x = x_new
        if it in record_at:
            rec[it] = x.clone()
    return (x, rec) if record_at else x
