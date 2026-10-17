"""Two-loop L-BFGS (Nocedal) for the data step of unrolled networks.

g = Aᵀ(Ax − y) is the gradient of the data term. The memory pairs come from earlier stages of the same forward pass:

    s = x − x_prev,    y = g − g_prev = AᵀA s        (exact: the data term is quadratic)

The two loops build H ≈ (AᵀA)⁻¹ on the directions visited so far and apply it to g; H₀ = I. With no pairs the result
is g itself, so the first stage is a plain gradient step. Pairs whose sᵀy is not clearly positive (unmeasured
directions) are dropped.

Why H₀ = I rather than γI with γ = sᵀy / yᵀy as in textbook L-BFGS: there is no line search here, the step α_k is
learned. The operator is normalised to ‖A‖ = 1, so outside the stored directions H₀ = I is exactly the gradient step
(stable for α < 2), whereas γ ≥ 1 amplifies every unstored direction. Plugged into a trained LEARN–ResNet (90°, 256
grid, 3 slices, α = 1, all stages stored): H₀ = I gives 24.45 dB right away, γI gives 22.69 dB (the original
first-order model: 26.51 dB; measured γ between 1.0 and 4.4). QN-Mixer also uses H₀ = I.

Stability while the network is still untrained (training from scratch). A pair with curvature κ = sᵀy / sᵀs close to
0 (the step x − x_prev lies mostly in unmeasured directions, where an untrained regularizer pushes the image) makes H
scale direction s by about 1/κ, and without a line search the image breaks at once: in from-scratch training on
2026-10-11, DenseNet diverged after ~2,300 steps and RegFormer after ~21,000. Two safeguards, both off when 0:

    damping = η  (Powell, relative to B₀ = I): a pair with sᵀy < η sᵀs has y replaced by ŷ = θ y + (1 − θ) s with
                 θ = (1 − η) sᵀs / (sᵀs − sᵀy), so that sᵀŷ = η sᵀs. H never scales a direction by more than 1/η;
                 sufficiently curved pairs are unchanged. With ‖A‖ = 1, κ always lies in [0, 1].
    curv_min = κ_min: drop pairs with sᵀy < κ_min sᵀs altogether (cautious update).
"""
from __future__ import annotations

import torch

_TINY = 1e-30
_CURV = 1e-8          # a pair is used only when sᵀy exceeds this fraction of ‖s‖‖y‖


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).flatten(1).sum(1).view(-1, *([1] * (a.dim() - 1)))


def _rho(s: torch.Tensor, y: torch.Tensor, curv_min: float = 0.0) -> torch.Tensor:
    sy, yy, ss = _dot(s, y), _dot(y, y), _dot(s, s)
    ok = sy > _CURV * (ss.sqrt() * yy.sqrt() + _TINY)
    if curv_min > 0:
        ok = ok & (sy >= curv_min * ss)
    return torch.where(ok, 1.0 / sy.clamp_min(_TINY), torch.zeros_like(sy))


def damp_pair(s: torch.Tensor, y: torch.Tensor, eta: float) -> torch.Tensor:
    sy, ss = _dot(s, y), _dot(s, s)
    theta = torch.where(sy >= eta * ss, torch.ones_like(sy), (1.0 - eta) * ss / (ss - sy).clamp_min(_TINY))
    return theta * y + (1.0 - theta) * s


def lbfgs_direction(grad: torch.Tensor, s: torch.Tensor, y: torch.Tensor, damping: float = 0.0,
                    curv_min: float = 0.0) -> torch.Tensor:
    if s.ndim != grad.ndim + 1 or s.shape[0] < 1 or s.shape[1:] != grad.shape or y.shape != s.shape:
        raise ValueError(f"s, y must have shape (M, *grad.shape), got s {tuple(s.shape)}, y {tuple(y.shape)}, "
                         f"grad {tuple(grad.shape)}")
    if not 0.0 <= damping < 1.0 or curv_min < 0.0:
        raise ValueError(f"need 0 <= damping < 1 and curv_min >= 0, got {damping}, {curv_min}")
    m = s.shape[0]
    if damping > 0:
        y = torch.stack([damp_pair(s[i], y[i], damping) for i in range(m)])
    q = grad
    alphas: list[torch.Tensor] = []
    rhos: list[torch.Tensor] = []
    for i in range(m - 1, -1, -1):
        rho = _rho(s[i], y[i], curv_min)
        alpha = rho * _dot(s[i], q)
        q = q - alpha * y[i]
        alphas.append(alpha)
        rhos.append(rho)
    r = q
    for i in range(m):
        j = m - 1 - i          # alphas/rhos are ordered from the newest pair
        beta = rhos[j] * _dot(y[i], r)
        r = r + s[i] * (alphas[j] - beta)
    return r
