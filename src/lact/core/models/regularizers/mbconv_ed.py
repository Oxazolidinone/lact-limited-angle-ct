"""LEARN-EfficientNet: the shared encoder-decoder with MBConv + SE blocks (Tan & Le, 2019) and GroupNorm."""
from __future__ import annotations

import torch.nn as nn

from ..blocks import MBConv
from .encdec import EncoderDecoder
from .layout import stage_layout


def mbconv_ed(channels=(32, 64, 128, 256), depths=(2, 2, 2, 2), dec_depth: int = 1, expand: int = 4, kernel: int = 3,
              se_ratio: float = 0.25, gn_groups: int = 8, regnet=None) -> EncoderDecoder:
    channels, depths = stage_layout(channels, depths, regnet)

    def block(c):
        return MBConv(c, c, 1, expand, kernel, se_ratio, gn_groups)

    def down(cin, cout):    # as in EfficientNet: the first MBConv block of a stage has stride 2
        return MBConv(cin, cout, 2, expand, kernel, se_ratio, gn_groups)

    stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
                         nn.GroupNorm(min(gn_groups, channels[0]), channels[0]), nn.SiLU())
    return EncoderDecoder(channels, depths, block, down, stem, dec_depth)
