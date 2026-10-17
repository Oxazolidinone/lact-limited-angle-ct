"""Four-level encoder-decoder of the staged regularizers (LEARN-DenseNet).

A family is defined by its block and downsampling; the stem, the decoder (ConvTranspose 2x2 + skip addition +
block) and the output layer are shared. Widths and depths per level come from ``layout.stage_layout`` (set by hand or
by the RegNet rule).
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from ..blocks import zero_init


class EncoderDecoder(nn.Module):
    def __init__(self, channels: list[int], depths: list[int], block: Callable[[int], nn.Module],
                 down: Callable[[int, int], nn.Module], stem: nn.Module, dec_depth: int = 1,
                 deepest: nn.Module | None = None):
        """``deepest``: module replacing the blocks of the deepest level (e.g. a Transformer core)."""
        super().__init__()
        if len(channels) != len(depths):
            raise ValueError("channels and depths must have the same length")
        self.levels = len(channels)
        self.stem = stem
        self.enc = nn.ModuleList(nn.Sequential(*[block(c) for _ in range(d)]) for c, d in zip(channels, depths))
        if deepest is not None:
            self.enc[-1] = deepest
        self.downs = nn.ModuleList(down(channels[i - 1], channels[i]) for i in range(1, self.levels))
        self.ups = nn.ModuleList(nn.ConvTranspose2d(channels[i], channels[i - 1], 2, stride=2)
                                 for i in range(self.levels - 1, 0, -1))
        self.dec = nn.ModuleList(nn.Sequential(*[block(channels[i - 1]) for _ in range(dec_depth)])
                                 for i in range(self.levels - 1, 0, -1))
        self.out = zero_init(nn.Conv2d(channels[0], 1, 3, padding=1))
        self.stride = 2 ** (self.levels - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H % self.stride or W % self.stride:
            raise ValueError(f"Image size must be divisible by {self.stride}")
        h = self.stem(x)
        skips = []
        for i, level in enumerate(self.enc):
            if i > 0:
                h = self.downs[i - 1](h)
            h = level(h)
            skips.append(h)
        for j, (up, dec) in enumerate(zip(self.ups, self.dec)):
            h = dec(up(h) + skips[-2 - j])
        return self.out(h)
