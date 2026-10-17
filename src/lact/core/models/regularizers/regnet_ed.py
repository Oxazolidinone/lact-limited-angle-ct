"""LEARN-RegNet and LEARN-ResNeXt: the shared encoder-decoder with X blocks (grouped-conv bottleneck) and GroupNorm.

- ``regnet_ed``: RegNet layout rule (Radosavovic et al., 2020); RegNetY-200MF by default (X block + SE).
- ``resnext_ed``: the same X block in a uniform layout (doubling widths, equal depth per level) as in ResNeXt
  (Xie et al., 2017).

The first block of stages 2-4 has stride 2 and does the downsampling (as in the original); stage 1 keeps the input
resolution. ``se_ratio`` > 0 adds squeeze-and-excitation (RegNetY's Y block); ``se_ratio`` = 0 is RegNetX.
"""
from __future__ import annotations

import torch.nn as nn

from ..blocks import RegNetBlock
from .encdec import EncoderDecoder
from .layout import REGNET_200MF, SE_RATIO_Y, regnet_stages, stage_layout

__all__ = ["regnet_ed", "regnet_stages", "resnext_ed", "xblock_parts"]


def xblock_parts(channels: list[int], group_width: int, se_ratio: float, gn_groups: int):
    """(block, down, stem) of the X-block family, shared by RegNet, ResNeXt and the ViT-RegNet hybrid."""
    def block(c):
        return RegNetBlock(c, c, 1, min(group_width, c), se_ratio, gn_groups)

    def down(cin, cout):
        return RegNetBlock(cin, cout, 2, min(group_width, cout), se_ratio, gn_groups)

    stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
                         nn.GroupNorm(min(gn_groups, channels[0]), channels[0]), nn.ReLU(inplace=True))
    return block, down, stem


def _xblock_ed(channels, depths, regnet, group_width, se_ratio, dec_depth, gn_groups) -> EncoderDecoder:
    channels, depths = stage_layout(channels, depths, regnet, group_width)
    block, down, stem = xblock_parts(channels, group_width, se_ratio, gn_groups)
    # the stride-2 block is already the first block of stages 2-4, so each of those levels adds only d - 1 blocks
    enc_depths = [depths[0]] + [d - 1 for d in depths[1:]]
    return EncoderDecoder(channels, enc_depths, block, down, stem, dec_depth)


def regnet_ed(regnet=None, channels=None, depths=None, group_width: int = 8, se_ratio: float = SE_RATIO_Y,
              dec_depth: int = 1, gn_groups: int = 8) -> EncoderDecoder:
    """RegNetY-200MF by default (w = [24, 56, 152, 368], d = [1, 1, 4, 7], g = 8, SE 0.25). ``regnet`` overrides
    individual rule parameters; pass ``channels`` and ``depths`` to set the layout by hand."""
    if (channels is None) != (depths is None):
        raise ValueError("channels and depths must be given together or both omitted")
    if channels is None:
        regnet = {**REGNET_200MF, **(regnet or {})}
    return _xblock_ed(channels, depths, regnet if channels is None else None, group_width, se_ratio, dec_depth,
                      gn_groups)


def resnext_ed(channels=(64, 128, 256, 512), depths=(2, 2, 2, 2), group_width: int = 16, se_ratio: float = 0.0,
               dec_depth: int = 1, gn_groups: int = 8) -> EncoderDecoder:
    return _xblock_ed(channels, depths, None, group_width, se_ratio, dec_depth, gn_groups)
