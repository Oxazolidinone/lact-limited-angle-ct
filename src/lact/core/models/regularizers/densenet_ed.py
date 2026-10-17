"""LEARN-DenseNet: the shared encoder-decoder with dense blocks (Huang et al., 2017) and GroupNorm."""
from __future__ import annotations

import torch.nn as nn

from ..blocks import DenseBlock
from .encdec import EncoderDecoder
from .layout import stage_layout


def densenet_ed(channels=(32, 64, 128, 256), depths=(1, 1, 1, 1), dec_depth: int = 1, growth_ratio: float = 0.5,
                layers: int = 3, gn_groups: int = 8, regnet=None, se_ratio: float = 0.0) -> EncoderDecoder:
    """Each block has ``layers`` concatenating layers; each adds growth = growth_ratio * c channels (multiple of 8)."""
    channels, depths = stage_layout(channels, depths, regnet)

    def block(c):
        return DenseBlock(c, max(8, int(round(c * growth_ratio / 8) * 8)), layers, gn_groups, se_ratio)

    def down(cin, cout):    # DenseNet transition layer: GN - ReLU - Conv 1x1 - AvgPool 2x2
        return nn.Sequential(nn.GroupNorm(min(gn_groups, cin), cin), nn.ReLU(inplace=True),
                             nn.Conv2d(cin, cout, 1, bias=False), nn.AvgPool2d(2))

    stem = nn.Conv2d(1, channels[0], 3, padding=1)
    return EncoderDecoder(channels, depths, block, down, stem, dec_depth)
