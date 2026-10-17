"""Building blocks shared by the regularizers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_init(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        return x * mask / keep


def _se(c: int, se_ratio: float):
    """SE as in RegNetY: hidden channels = se_ratio x block width; se_ratio = 0 disables SE."""
    return SqueezeExcite(c, max(1, round(se_ratio * c))) if se_ratio > 0 else None


class BasicBlock(nn.Module):
    """ResNet BasicBlock (He et al., 2016); ``se_ratio`` > 0 adds SE on the residual branch (SE-ResNet)."""

    def __init__(self, c: int, groups: int = 8, se_ratio: float = 0.0):
        super().__init__()
        g = min(groups, c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(g, c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(g, c)
        self.se = _se(c, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.conv1(x)))
        h = self.n2(self.conv2(h))
        if self.se is not None:
            h = self.se(h)
        return F.relu(x + h)


class PlainBlock(nn.Module):
    """Plain VGG/U-Net conv block: Conv 3x3 - GroupNorm - ReLU, no shortcut (the "V" block of the RegNet paper)."""

    def __init__(self, c_in: int, c_out: int, groups: int = 8, se_ratio: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(groups, c_out), c_out)
        self.se = _se(c_out, se_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm(self.conv(x)))
        return h if self.se is None else self.se(h)


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


class MBConv(nn.Module):
    """MobileNetV2 / EfficientNet MBConv (Sandler et al., 2018; Tan & Le, 2019): Conv 1x1 expanding ``expand`` times
    -> depthwise Conv 3x3 -> SE -> projecting Conv 1x1; the shortcut is added when the shape is unchanged."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1, expand: int = 4, kernel: int = 3,
                 se_ratio: float = 0.25, groups: int = 8):
        super().__init__()
        mid = c_in * expand
        self.expand = nn.Sequential(nn.Conv2d(c_in, mid, 1, bias=False), nn.GroupNorm(min(groups, mid), mid), nn.SiLU())
        self.dw = nn.Sequential(nn.Conv2d(mid, mid, kernel, stride=stride, padding=kernel // 2, groups=mid, bias=False),
                                nn.GroupNorm(min(groups, mid), mid), nn.SiLU())
        self.se = SqueezeExcite(mid, max(1, round(se_ratio * c_in))) if se_ratio > 0 else None
        self.project = nn.Sequential(nn.Conv2d(mid, c_out, 1, bias=False), nn.GroupNorm(min(groups, c_out), c_out))
        self.residual = stride == 1 and c_in == c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(self.expand(x))
        if self.se is not None:
            h = self.se(h)
        h = self.project(h)
        return x + h if self.residual else h


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018): channel weights from the global average."""

    def __init__(self, c: int, c_se: int):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c_se, 1)
        self.fc2 = nn.Conv2d(c_se, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))


class RegNetBlock(nn.Module):
    """RegNet X block (Radosavovic et al., 2020): Conv 1x1 -> grouped Conv 3x3 (group width g) -> Conv 1x1 plus a
    shortcut; bottleneck ratio b = 1. ``se_ratio`` > 0 adds SE after the Conv 3x3 (RegNetY's Y block).

    ``stride`` = 2 or ``c_in`` != ``c_out``: the shortcut is a Conv 1x1 (same stride) + norm, as in the first block of
    each stage. GroupNorm replaces the original BatchNorm because the batch size is 1.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, group_width: int = 16, se_ratio: float = 0.0,
                 gn_groups: int = 8):
        super().__init__()
        if c_out % group_width:
            raise ValueError(f"channels ({c_out}) must be divisible by the group width ({group_width})")
        g = min(gn_groups, c_out)
        self.conv1 = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.n1 = nn.GroupNorm(g, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=stride, padding=1, groups=c_out // group_width, bias=False)
        self.n2 = nn.GroupNorm(g, c_out)
        self.se = SqueezeExcite(c_out, max(1, round(se_ratio * c_in))) if se_ratio > 0 else None
        self.conv3 = nn.Conv2d(c_out, c_out, 1, bias=False)
        self.n3 = nn.GroupNorm(g, c_out)
        self.proj = None
        if stride != 1 or c_in != c_out:
            self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False), nn.GroupNorm(g, c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.n1(self.conv1(x)))
        h = F.relu(self.n2(self.conv2(h)))
        if self.se is not None:
            h = self.se(h)
        h = self.n3(self.conv3(h))
        return F.relu((x if self.proj is None else self.proj(x)) + h)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block (Liu et al., 2022): DWConv kxk -> LN -> Linear 4x -> GELU -> Linear -> layer scale."""

    def __init__(self, c: int, kernel: int = 7, mlp_ratio: int = 4, layer_scale_init: float = 1e-6,
                 drop_path: float = 0.0, se_ratio: float = 0.0):
        super().__init__()
        self.se = _se(c, se_ratio)
        self.dw = nn.Conv2d(c, c, kernel, padding=kernel // 2, groups=c)
        self.ln = nn.LayerNorm(c, eps=1e-6)
        self.pw1 = nn.Linear(c, mlp_ratio * c)
        self.pw2 = nn.Linear(mlp_ratio * c, c)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(c)) if layer_scale_init > 0 else None
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x).permute(0, 2, 3, 1)
        h = self.pw2(F.gelu(self.pw1(self.ln(h))))
        if self.gamma is not None:
            h = h * self.gamma
        h = h.permute(0, 3, 1, 2)
        if self.se is not None:
            h = self.se(h)
        return x + self.drop(h)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by the number of heads ({heads})")
        self.heads = heads
        self.n1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.n2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))
        self.drop = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.proj(a))
        return x + self.drop(self.mlp(self.n2(x)))


class TransformerStage(nn.Module):
    """Transformer core on a feature map (B, D, h, w): every position is a token, a learned positional embedding is
    added, ``depth`` Transformer blocks are applied and the map shape is restored. Used as the deepest encoder level."""

    def __init__(self, dim: int, depth: int, heads: int, grid: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        self.grid, self.dim = int(grid), int(dim)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, mlp_ratio, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, h, w = x.shape
        if h != self.grid or w != self.grid:
            raise ValueError(f"the Transformer core was built for a {self.grid}x{self.grid} grid, got {h}x{w}")
        t = self.norm(self.blocks(x.flatten(2).transpose(1, 2) + self.pos))
        return t.transpose(1, 2).reshape(B, D, h, w)

    def attention_flops(self) -> int:
        """Attention FLOPs (QK^T and AV) that FLOP counters may miss: 4 * N^2 * D * L."""
        return int(4 * self.grid ** 4 * self.dim * len(self.blocks))

