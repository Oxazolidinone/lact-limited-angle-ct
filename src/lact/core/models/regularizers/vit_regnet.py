"""LEARN-ViT-RegNet: the LEARN-ViT Transformer core placed in a RegNet layout.

A ViT is isotropic (no stages), so the RegNet rule cannot be applied directly. Here the first three stages of
RegNetY-200MF (Y blocks, 8x downsampling) replace LEARN-ViT's 8x8 patch embedding, the last stage is the Transformer
core (the same 32x32 tokens for a 256x256 image), and the shared decoder with skips replaces the linear patch fold.
"""
from __future__ import annotations

from ..blocks import TransformerStage
from .encdec import EncoderDecoder
from .layout import REGNET_200MF, SE_RATIO_Y, stage_layout
from .regnet_ed import xblock_parts


def vit_regnet(img_size: int = 256, regnet=None, group_width: int = 8, se_ratio: float = SE_RATIO_Y, dim: int = 192,
               depth: int = 6, heads: int = 6, mlp_ratio: int = 4, drop_path: float = 0.0, dec_depth: int = 1,
               gn_groups: int = 8) -> EncoderDecoder:
    widths, depths = stage_layout(None, None, {**REGNET_200MF, **(regnet or {})}, group_width)
    if img_size % 8:
        raise ValueError("img_size must be divisible by 8")
    if dim % min(group_width, dim):
        raise ValueError(f"dim ({dim}) must be divisible by the group width ({group_width})")
    channels = [*widths[:3], dim]
    block, down, stem = xblock_parts(channels, group_width, se_ratio, gn_groups)
    enc_depths = [depths[0], depths[1] - 1, depths[2] - 1, 0]
    core = TransformerStage(dim, depth, heads, img_size // 8, mlp_ratio, drop_path)
    return EncoderDecoder(channels, enc_depths, block, down, stem, dec_depth, deepest=core)
