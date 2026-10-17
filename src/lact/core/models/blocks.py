"""Building blocks shared by the regularizers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_init(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


def _se(c: int, se_ratio: float):
    """SE as in RegNetY: hidden channels = se_ratio x block width; se_ratio = 0 disables SE."""
    return SqueezeExcite(c, max(1, round(se_ratio * c))) if se_ratio > 0 else None


class DenseBlock(nn.Module):
    """DenseNet dense block (Huang et al., 2017): each GN - ReLU - Conv 3x3 layer takes the CHANNEL CONCATENATION of
    the input and all previous layers and adds ``growth`` channels; a final Conv 1x1 maps back to ``c`` channels."""

    def __init__(self, c: int, growth: int, n_layers: int = 3, groups: int = 8, se_ratio: float = 0.0):
        super().__init__()
        self.se = _se(c, se_ratio)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.GroupNorm(min(groups, c + i * growth), c + i * growth), nn.ReLU(inplace=True),
                          nn.Conv2d(c + i * growth, growth, 3, padding=1, bias=False)) for i in range(n_layers))
        total = c + n_layers * growth
        self.fuse = nn.Sequential(nn.GroupNorm(min(groups, total), total), nn.ReLU(inplace=True),
                                  nn.Conv2d(total, c, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for layer in self.layers:
            feats.append(layer(torch.cat(feats, dim=1)))
        h = self.fuse(torch.cat(feats, dim=1))
        return h if self.se is None else self.se(h)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018): channel weights from the global average."""

    def __init__(self, c: int, c_se: int):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c_se, 1)
        self.fc2 = nn.Conv2d(c_se, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
