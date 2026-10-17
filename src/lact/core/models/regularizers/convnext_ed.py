"""LEARN-ConvNeXt: ConvNeXt blocks (Liu et al., 2022) with LayerNorm."""
from __future__ import annotations

import torch.nn as nn

from ..blocks import ConvNeXtBlock, LayerNorm2d
from .encdec import EncoderDecoder
from .layout import stage_layout


def convnext_ed(channels=(32, 64, 128, 256), depths=(2, 2, 2, 2), dec_depth: int = 1, kernel: int = 7,
                mlp_ratio: int = 4, layer_scale_init: float = 1e-6, drop_path: float = 0.0,
                regnet=None, se_ratio: float = 0.0) -> EncoderDecoder:
    channels, depths = stage_layout(channels, depths, regnet)

    def block(c):
        return ConvNeXtBlock(c, kernel, mlp_ratio, layer_scale_init, drop_path, se_ratio)

    def down(cin, cout):
        return nn.Sequential(LayerNorm2d(cin), nn.Conv2d(cin, cout, 2, stride=2))

    stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1), LayerNorm2d(channels[0]))
    return EncoderDecoder(channels, depths, block, down, stem, dec_depth)
