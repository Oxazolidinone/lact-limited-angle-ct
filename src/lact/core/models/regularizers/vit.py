"""LEARN-ViT: a Vision Transformer (Dosovitskiy et al., 2021) adapted to image-valued outputs.

Conv 3x3 (1->C0) -> p x p patches -> N = (H/p)(W/p) tokens of dimension D -> L Transformer blocks
-> Linear D -> p*p*C0 and fold back to an image -> concatenate with the first conv features (skip keeps detail)
-> Conv 3x3 -> GELU -> Conv 3x3 (-> 1 channel, zero-initialised).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import TransformerBlock, zero_init


def sincos_2d(grid: int, dim: int) -> torch.Tensor:
    """Fixed 2-D sin-cos positional embedding, (1, grid*grid, dim)."""
    if dim % 4:
        raise ValueError("dim must be divisible by 4 for the sin-cos embedding")
    y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32),
                          indexing="ij")
    omega = 1.0 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))
    out = [torch.sin(x.flatten()[:, None] * omega), torch.cos(x.flatten()[:, None] * omega),
           torch.sin(y.flatten()[:, None] * omega), torch.cos(y.flatten()[:, None] * omega)]
    return torch.cat(out, dim=1)[None]


class ViTRegularizer(nn.Module):
    def __init__(self, img_size: int = 256, patch: int = 8, dim: int = 192, depth: int = 6, heads: int = 6,
                 mlp_ratio: int = 4, stem_channels: int = 32, pos: str = "learned", drop_path: float = 0.0):
        super().__init__()
        if img_size % patch:
            raise ValueError("img_size must be divisible by patch")
        self.img_size, self.patch = img_size, patch
        self.grid = img_size // patch
        C0 = stem_channels
        self.stem = nn.Sequential(nn.Conv2d(1, C0, 3, padding=1), nn.GELU())
        self.embed = nn.Conv2d(C0, dim, patch, stride=patch)
        if pos == "learned":
            self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        elif pos == "sincos":
            self.register_buffer("pos", sincos_2d(self.grid, dim), persistent=False)
        else:
            raise ValueError(pos)
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.unpatch = nn.Linear(dim, patch * patch * C0)
        self.tail = nn.Sequential(nn.Conv2d(2 * C0, C0, 3, padding=1), nn.GELU(),
                                  zero_init(nn.Conv2d(C0, 1, 3, padding=1)))
        self.apply(self._init)
        zero_init(self.tail[-1])

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @property
    def n_tokens(self) -> int:
        return self.grid ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if self.img_size != H or self.img_size != W:
            raise ValueError(f"ViT was built for {self.img_size}x{self.img_size} images, got {H}x{W}")
        f = self.stem(x)
        t = self.embed(f).flatten(2).transpose(1, 2) + self.pos
        t = self.norm(self.blocks(t))
        u = self.unpatch(t).transpose(1, 2)                      # (B, p*p*C0, N)
        u = F.fold(u, (H, W), kernel_size=self.patch, stride=self.patch)
        return self.tail(torch.cat([f, u], dim=1))

    def attention_flops(self) -> int:
        """Attention FLOPs (QK^T and AV) that FLOP counters may miss: 4 * N^2 * D * L."""
        N = self.n_tokens
        D = self.embed.out_channels
        L = len(self.blocks)
        return int(4 * N * N * D * L)


def vit(img_size=256, patch=8, dim=192, depth=6, heads=6, mlp_ratio=4, stem_channels=32, pos="learned",
        drop_path=0.0) -> ViTRegularizer:
    return ViTRegularizer(img_size, patch, dim, depth, heads, mlp_ratio, stem_channels, pos, drop_path)
