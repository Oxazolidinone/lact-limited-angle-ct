"""LEARN-UNet: plain VGG/U-Net style convolution blocks (Ronneberger et al., 2015), no skip inside a block."""
from __future__ import annotations

import torch.nn as nn

from ..blocks import PlainBlock
from .encdec import EncoderDecoder
from .layout import stage_layout


def unet_ed(channels=(32, 64, 128, 256), depths=(2, 2, 2, 2), dec_depth: int = 1, gn_groups: int = 8,
            regnet=None, se_ratio: float = 0.0) -> EncoderDecoder:
    channels, depths = stage_layout(channels, depths, regnet)

    def block(c):
        return PlainBlock(c, c, gn_groups, se_ratio)

    def down(cin, cout):
        return nn.Sequential(nn.MaxPool2d(2), PlainBlock(cin, cout, gn_groups))

    return EncoderDecoder(channels, depths, block, down, PlainBlock(1, channels[0], gn_groups), dec_depth)
