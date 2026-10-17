"""LEARN-CNN:(Chen et al., 2018)."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..blocks import zero_init


class CNN3(nn.Module):
    def __init__(self, channels: int = 48, kernel: int = 5, zero_last: bool = True, init_std: float | None = None):
        """``init_std``: draw every kernel from N(0, init_std^2) with zero bias (the original LEARN paper uses 0.01);
        None keeps the PyTorch default. ``zero_last`` (the project default) then zeroes the last layer."""
        super().__init__()
        p = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel, padding=p), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel, padding=p), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel, padding=p),
        )
        if init_std is not None:
            for m in self.net:
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, mean=0.0, std=float(init_std))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        if zero_last:
            zero_init(self.net[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
