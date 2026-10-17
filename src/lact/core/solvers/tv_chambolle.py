"""Batched TV denoising with Chambolle's projection algorithm (2004), on the tensor's device.

Same iteration as ``skimage.restoration.denoise_tv_chambolle`` (2-D, single channel), which the original DPMA code
calls on the CPU: step tau = 1/4, dual update p <- (p - tau * grad(out)) / (1 + tau/weight * |grad(out)|), output
out = image - adjoint-div(p), early stop when the energy changes by less than ``eps`` times its initial value. Each
sample is processed independently (stopping rule included), so an image's result does not depend on its batch.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def denoise_tv_chambolle(image: torch.Tensor, weight: float = 0.1, eps: float = 2e-4, max_iter: int = 200
                         ) -> torch.Tensor:
    """image: (B,1,H,W). A larger ``weight`` smooths more; at most ``max_iter`` iterations."""
    if image.dim() != 4 or image.shape[1] != 1:
        raise ValueError(f"expected a (B,1,H,W) tensor, got {tuple(image.shape)}")
    img = image[:, 0]
    px = torch.zeros_like(img)                  # components of p along rows (0) and columns (1)
    py = torch.zeros_like(img)
    out = img.clone()
    result = img.clone()
    active = torch.ones(img.shape[0], dtype=torch.bool, device=img.device)
    e_init = torch.zeros(img.shape[0], dtype=img.dtype, device=img.device)
    e_prev = torch.zeros_like(e_init)
    tau = 0.25
    for i in range(int(max_iter)):
        if i > 0:
            d = -(px + py)
            d[:, 1:, :] += px[:, :-1, :]
            d[:, :, 1:] += py[:, :, :-1]
            out = img + d
            energy = (d ** 2).flatten(1).sum(1)
        else:
            energy = torch.zeros_like(e_init)
        gx = torch.zeros_like(img)
        gy = torch.zeros_like(img)
        gx[:, :-1, :] = out[:, 1:, :] - out[:, :-1, :]
        gy[:, :, :-1] = out[:, :, 1:] - out[:, :, :-1]
        norm = torch.sqrt(gx ** 2 + gy ** 2)
        energy = (energy + weight * norm.flatten(1).sum(1)) / float(img[0].numel())
        result = torch.where(active.view(-1, 1, 1), out, result)
        if i == 0:
            e_init, e_prev = energy.clone(), energy.clone()
        else:
            active = active & ~((e_prev - energy).abs() < eps * e_init)
            e_prev = energy
            if not bool(active.any()):
                break
        scale = norm * (tau / weight) + 1.0
        px = (px - tau * gx) / scale
        py = (py - tau * gy) / scale
    return result.unsqueeze(1)
