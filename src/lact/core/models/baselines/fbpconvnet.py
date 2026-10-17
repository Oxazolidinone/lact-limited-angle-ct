"""FBPConvNet (Jin et al., IEEE TIP 2017): a U-Net post-processing the FBP image; no sinogram, no projector.

    x = x^0 + UNet(x^0)

The original U-Net: 5 levels of 64-128-256-512-1024 channels, two layers per level (3x3 conv + BatchNorm + ReLU),
three in the first level; 2x2 max-pool downsampling, 2x2 transposed convolution + BatchNorm + ReLU upsampling
concatenated with the same-level features; a 1x1 output layer; the input image is added at the end. 31.08M
parameters (the CG-GLORE paper reports 31.1M).

The model implements the ``Unrolled`` interface so training and evaluation need no special case: K = 1 and ``y`` is
ignored.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..unrolled import Unrolled


def _conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _up(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class FBPConvNet(Unrolled):
    def __init__(self, base: int = 64, depth: int = 4, grad_checkpoint: bool = False, projector=None):
        super().__init__()
        if base < 1 or depth < 1:
            raise ValueError("FBPConvNet needs base >= 1 and depth >= 1")
        self.K = 1
        self.depth = int(depth)
        widths = [base * 2 ** i for i in range(self.depth + 1)]
        self.enc = nn.ModuleList([nn.Sequential(_conv(1, base), _conv(base, base), _conv(base, base))])
        for lo, hi in zip(widths[:-1], widths[1:]):
            self.enc.append(nn.Sequential(_conv(lo, hi), _conv(hi, hi)))
        self.up = nn.ModuleList(_up(hi, lo) for lo, hi in zip(widths[:-1], widths[1:]))
        self.dec = nn.ModuleList(nn.Sequential(_conv(2 * lo, lo), _conv(lo, lo)) for lo in widths[:-1])
        self.out = nn.Conv2d(base, 1, 1)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.projector = projector

    def stage_params(self) -> dict[str, torch.Tensor]:
        return {}

    def regularizer_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def regularization_only(self, x: torch.Tensor) -> None:
        self._net(x)

    def _net(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2 ** self.depth or x.shape[-2] % 2 ** self.depth:
            raise ValueError(f"Image size {tuple(x.shape[-2:])} must be divisible by {2 ** self.depth}")
        skips = []
        z = x
        for i, enc in enumerate(self.enc):
            z = enc(z if i == 0 else F.max_pool2d(z, 2))
            skips.append(z)
        for i in reversed(range(self.depth)):
            z = self.dec[i](torch.cat([skips[i], self.up[i](z)], dim=1))
        return x + self.out(z)

    def forward(self, y: torch.Tensor, x0: torch.Tensor, return_stages: bool = False, return_terms: bool = False):
        """x0: FBP image (B,1,H,W); y is unused. Return options as in ``UnrolledLEARN.forward``."""
        x = self._net(x0)       # a single stage: gradient checkpointing would save nothing, so it is not used
        out = [x0, x] if return_stages else x
        if return_terms:
            with torch.no_grad():
                terms = [{"k": 0, "dc_norm": 0.0, "reg_norm": self._norm(x - x0)}]
            return out, terms
        return out
