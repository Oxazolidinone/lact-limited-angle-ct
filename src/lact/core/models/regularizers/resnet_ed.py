"""LEARN-ResNet: encoder-decoder with BasicBlocks (He et al., 2016) and GroupNorm."""
from __future__ import annotations

import torch.nn as nn

from ..blocks import BasicBlock
from .encdec import EncoderDecoder
from .layout import stage_layout


def resnet_ed(channels=(32, 64, 128, 256), depths=(2, 2, 2, 2), dec_depth: int = 1, gn_groups: int = 8,
              regnet=None, se_ratio: float = 0.0) -> EncoderDecoder:
    """``regnet`` = {w0, wa, wm, depth}: derive widths/depths with the RegNet rule instead of ``channels``/``depths``.
    ``se_ratio`` > 0 adds squeeze-and-excitation to every block (as in RegNetY's Y block)."""
    channels, depths = stage_layout(channels, depths, regnet)

    def block(c):
        return BasicBlock(c, gn_groups, se_ratio)

    def down(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                             nn.GroupNorm(min(gn_groups, cout), cout), nn.ReLU(inplace=True))

    stem = nn.Conv2d(1, channels[0], 3, padding=1)
    return EncoderDecoder(channels, depths, block, down, stem, dec_depth)
