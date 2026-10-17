"""RegFormer (Xia, Yang, Lu, Wang, Zhang: "RegFormer: A Local-Nonlocal Regularization-Based Model for Sparse-View CT
Reconstruction", IEEE TRPMS 8(2), 2024; conference version MICCAI 2022), rebuilt after the architecture of the public
code (github.com/Deep-Imaging-Group/RegFormer) on this project's projector and FBP.

For each stage k = 0..K-1, with x^0 = FBP(y) and no incoming features in the first stage:

    x^{k+1} = ReLU( x^k + w_k * FBP(y - A x^k) + N_k(x^k, z^k) ),        z^{k+1} = features of N_k

Differences from LEARN (``unrolled.py``):
- The data step is the FILTERED backprojection (FBP) of the sinogram residual instead of A^T; w_k is a scalar
  initialised to 0.
- N_k alternates: even stages are convolutional blocks (three 5x5 layers), odd stages are Swin Transformer blocks (one
  token per pixel, attention within ``window`` x ``window`` windows, odd layers shift the windows by half a window).
- Features pass between stages: each block also emits ``dim/2`` feature channels; the next block concatenates them with
  ``dim/2`` channels it extracts from the image itself (the first block extracts all ``dim`` channels, the last block
  emits no features).
- Every stage has its own weights; each stage output goes through a ReLU.

The defaults (dim 96, window 8, 2 Swin layers, 3 heads, K = 10) give 2.68M parameters, matching the 10-stage / 2.7M
configuration the CG-GLORE paper compares against. Three differences from the original:
- The FBP and the projector are this project's (flat-detector fan beam, limited angle).
- The shifted-window mask is computed from the runtime image size, so the model is not tied to one grid size.
- ``zero_out=True`` (default): each block's image-output convolution starts at 0, so every stage initially returns its
  input (like the zero-initialised last layer of the LEARN framework). The original initialises every convolution
  from N(0, 0.01^2); the Swin block (output after LayerNorm, magnitude ~1) then adds an almost constant offset of
  random sign, about 0.3 per block. On one 256 x 256 Mayo slice with 6 seeds, the initial mean output ranged from
  0.01 to 0.97 (true image 0.31), and for one seed 71% of the pixels were 0 after the ReLU; an all-zero output stops
  the gradient entirely. ``zero_out=False`` restores the original initialisation.

``data_step = "lbfgs"`` replaces the FBP data step by a quasi-Newton step on the data gradient (the same formula as
``update = "lbfgs"`` in the LEARN framework, see ``core/solvers/lbfgs.py``):

    g = A^T (A x^k - y),    x^{k+1} = ReLU( x^k - alpha_k * H_k g + N_k(x^k, z^k) )

alpha_k > 0 (softplus, initialised at 1) is a separate parameter (``a_qn``); the FBP weight ``w`` is unused. The N_k
blocks keep their names, so FBP-variant checkpoints still load.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ...operators.fbp import FanBeamFBP
from ...solvers.lbfgs import lbfgs_direction
from ..unrolled import Unrolled


class DropPath(nn.Module):
    """Stochastic depth: in training, drop a sample's residual branch with probability ``p`` (rescaled by 1 - p)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = torch.rand(x.shape[0], *([1] * (x.dim() - 1)), device=x.device, dtype=x.dtype) >= self.p
        return x * keep / (1.0 - self.p)


class WindowAttention(nn.Module):
    """Multi-head attention within a window, with a learned relative position bias (Swin)."""

    def __init__(self, dim: int, window: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")
        self.window, self.heads = int(window), int(heads)
        self.scale = (dim // heads) ** -0.5
        self.bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        ij = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = (ij[:, :, None] - ij[:, None, :]).permute(1, 2, 0) + (window - 1)       # (w^2, w^2, 2), each axis 0..2w-2
        self.rel_index: torch.Tensor
        self.register_buffer("rel_index", rel[..., 0] * (2 * window - 1) + rel[..., 1], persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (windows x B, w^2, C); mask: (windows, w^2, w^2), added to the attention scores."""
        n, t, c = x.shape
        q, k, v = self.qkv(x).reshape(n, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)                                 # (n, heads, t, t)
        attn = attn + self.bias_table[self.rel_index.reshape(-1)].view(t, t, -1).permute(2, 0, 1)
        if mask is not None:
            attn = (attn.view(-1, mask.shape[0], self.heads, t, t) + mask[None, :, None]).view(n, self.heads, t, t)
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(n, t, c))


class SwinBlock(nn.Module):
    """One Swin layer on a token grid (B, H, W, C): window attention (shifted by ``shift`` when > 0), then an MLP."""

    def __init__(self, dim: int, heads: int, window: int, shift: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.window, self.shift = int(window), int(shift)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = DropPath(drop_path)
        self._masks: dict[tuple, torch.Tensor] = {}

    def _mask(self, h: int, w: int, like: torch.Tensor) -> torch.Tensor | None:
        """Shifted windows join regions from opposite edges: block attention between tokens of different regions."""
        if not self.shift:
            return None
        key = (h, w, like.device, like.dtype)
        if key not in self._masks:
            ws, s = self.window, self.shift
            region = torch.zeros(h, w, device=like.device)
            cuts = (slice(0, -ws), slice(-ws, -s), slice(-s, None))
            for n, (a, b) in enumerate((a, b) for a in cuts for b in cuts):
                region[a, b] = n
            ids = region.view(h // ws, ws, w // ws, ws).permute(0, 2, 1, 3).reshape(-1, ws * ws)
            diff = ids[:, None, :] - ids[:, :, None]
            self._masks = {key: torch.zeros_like(diff, dtype=like.dtype).masked_fill(diff != 0, -100.0)}
        return self._masks[key]

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        ws = self.window
        if self.shift:
            x = torch.roll(x, (-self.shift, -self.shift), dims=(1, 2))
        win = x.view(b, h // ws, ws, w // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)
        win = self.attn(win, self._mask(h, w, x))
        x = win.view(b, h // ws, w // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
        return torch.roll(x, (self.shift, self.shift), dims=(1, 2)) if self.shift else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self._attend(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


def _feature_head(dim: int, last: bool) -> nn.Conv2d | None:
    """Layer emitting dim/2 feature channels for the next stage (absent in the last stage)."""
    return None if last else nn.Conv2d(dim, dim // 2, 3, padding=1)


class ConvStage(nn.Module):
    """Local regularization: Conv 5x5 -> ReLU -> [concat z] -> Conv 5x5 -> ReLU -> Conv 5x5 (to 1 channel)."""

    def __init__(self, dim: int, first: bool, last: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim if first else dim // 2, 5, padding=2)
        self.conv2 = nn.Conv2d(dim, dim, 5, padding=2)
        self.conv3 = nn.Conv2d(dim, 1, 5, padding=2)
        self.to_feat = _feature_head(dim, last)

    @property
    def out_layer(self) -> nn.Conv2d:
        return self.conv3

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        f = F.relu(self.conv1(x))
        if z is not None:
            f = torch.cat((f, z), dim=1)
        f = F.relu(self.conv2(f))
        return self.conv3(f), (self.to_feat(f) if self.to_feat is not None else None)


class SwinStage(nn.Module):
    """Non-local regularization: Conv 3x3 pixel embedding -> [concat z] -> Swin layers -> LayerNorm -> Conv 3x3."""

    def __init__(self, dim: int, first: bool, last: bool, depth: int, heads: int, window: int, mlp_ratio: float,
                 drop_path: float):
        super().__init__()
        self.window = int(window)
        self.embed = nn.Conv2d(1, dim if first else dim // 2, 3, padding=1)
        rates = [drop_path * i / max(1, depth - 1) for i in range(depth)]     # linear increase, 0 for the first layer
        self.blocks = nn.ModuleList(SwinBlock(dim, heads, window, 0 if i % 2 == 0 else window // 2, mlp_ratio, r)
                                    for i, r in enumerate(rates))
        self.norm = nn.LayerNorm(dim)
        self.to_image = nn.Conv2d(dim, 1, 3, padding=1)
        self.to_feat = _feature_head(dim, last)

    @property
    def out_layer(self) -> nn.Conv2d:
        return self.to_image

    def forward(self, x: torch.Tensor, z: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.shape[-2] % self.window or x.shape[-1] % self.window:
            raise ValueError(f"Image size {tuple(x.shape[-2:])} must be divisible by the window {self.window}")
        t = self.embed(x)
        if z is not None:
            t = torch.cat((t, z), dim=1)
        t = t.permute(0, 2, 3, 1)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t).permute(0, 3, 1, 2)
        return self.to_image(t), (self.to_feat(t) if self.to_feat is not None else None)


class RegFormer(Unrolled):
    def __init__(self, K: int = 10, dim: int = 96, window: int = 8, depth: int = 2, heads: int = 3,
                 mlp_ratio: float = 4.0, drop_path: float = 0.1, zero_out: bool = True,
                 grad_checkpoint: bool = False, projector=None, fbp_filter: str = "ram-lak", fbp_cutoff: float = 1.0,
                 data_step: str = "fbp", lbfgs_memory: int = 13, lbfgs_damping: float = 0.0,
                 lbfgs_curv_min: float = 0.0):
        super().__init__()
        if data_step not in ("fbp", "lbfgs"):
            raise ValueError(f"data_step must be 'fbp' or 'lbfgs', got {data_step!r}")
        if data_step == "lbfgs" and int(lbfgs_memory) < 1:
            raise ValueError("lbfgs_memory must be >= 1")
        if K < 1 or dim % 2:
            raise ValueError("RegFormer needs K >= 1 and an even dim")
        self.data_step = data_step
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_damping, self.lbfgs_curv_min = float(lbfgs_damping), float(lbfgs_curv_min)
        self.K = int(K)
        stages: list[ConvStage | SwinStage] = [
            ConvStage(dim, k == 0, k == K - 1) if k % 2 == 0
            else SwinStage(dim, k == 0, k == K - 1, depth, heads, window, mlp_ratio, drop_path) for k in range(K)]
        self.stages = nn.ModuleList(stages)
        if data_step == "fbp":
            self.w = nn.Parameter(torch.zeros(self.K))
        else:
            self.a_qn = nn.Parameter(torch.full((self.K,), math.log(math.expm1(1.0))))     # softplus -> 1
        self.grad_checkpoint = bool(grad_checkpoint)
        self.fbp_filter, self.fbp_cutoff = fbp_filter, float(fbp_cutoff)
        self.fbp: FanBeamFBP | None = None
        self.projector = None
        self.apply(self._init)
        if zero_out:
            for stage in stages:
                nn.init.zeros_(stage.out_layer.weight)
        if projector is not None:
            self.set_projector(projector)

    @staticmethod
    def _init(m: nn.Module) -> None:
        """As in the original: convolutions N(0, 0.01^2) with zero bias; Swin linear layers use the Swin init."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def set_projector(self, projector) -> RegFormer:
        """Attach the projector and build an FBP for the same geometry and device (call again to change geometry)."""
        self.projector = projector
        self.fbp = FanBeamFBP(projector.geom, projector.img_size, self.fbp_filter, self.fbp_cutoff,
                              device=projector.device)
        return self

    @property
    def alphas(self) -> torch.Tensor:
        """Per-stage step size of the L-BFGS direction (only with ``data_step = "lbfgs"``)."""
        return F.softplus(self.a_qn)

    def stage_params(self) -> dict[str, torch.Tensor]:
        return {"weights": self.w} if self.data_step == "fbp" else {"alphas": self.alphas}

    def regularizer_params(self) -> int:
        return sum(p.numel() for p in self.stages.parameters())

    def regularization_only(self, x: torch.Tensor) -> None:
        """Run only the learned part of all K stages on image x, without projections (for FLOP counting)."""
        z = None
        for stage in self.stages:
            _, z = stage(x, z)

    # ------------------------------------------------------------------ one stage
    def _data_step(self, x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
        """w_k * FBP(y - A x): raw sinogram in, image on the scale of x out."""
        assert self.fbp is not None
        return self.w[k] * self.fbp.to(x.device).apply(y - self.projector.raw_forward(x))

    def _qn_direction(self, x: torch.Tensor, y: torch.Tensor, hist: list) -> torch.Tensor:
        """L-BFGS direction of the normalised data gradient. ``hist`` = [previous x, previous g, the s, the y]."""
        y_n = self.projector.normalize_data(y)
        g = self.projector.adjoint(self.projector.forward(x) - y_n)
        prev_x, prev_g, ss, ys = hist
        if prev_x is not None:              # memory pairs are constants in backpropagation (as in UnrolledLEARN)
            ss.append((x - prev_x).detach())
            ys.append((g - prev_g).detach())
            if len(ss) > self.lbfgs_memory:
                ss.pop(0)
                ys.pop(0)
        d = g if not ss else lbfgs_direction(g, torch.stack(ss), torch.stack(ys), self.lbfgs_damping,
                                             self.lbfgs_curv_min)
        hist[0], hist[1] = x, g
        return d

    def _stage_with_data(self, x: torch.Tensor, z: torch.Tensor | None, data: torch.Tensor, k: int
                         ) -> tuple[torch.Tensor, torch.Tensor | None]:
        reg, z_out = self.stages[k](x, z)
        return F.relu(x + data + reg), z_out

    def _stage_with_data_last(self, x: torch.Tensor, z: torch.Tensor | None, data: torch.Tensor, k: int
                              ) -> torch.Tensor:
        return self._stage_with_data(x, z, data, k)[0]

    def _stage(self, x: torch.Tensor, z: torch.Tensor | None, y: torch.Tensor, k: int
               ) -> tuple[torch.Tensor, torch.Tensor | None]:
        reg, z_out = self.stages[k](x, z)
        return F.relu(x + self._data_step(x, y, k) + reg), z_out

    def _stage_last(self, x: torch.Tensor, z: torch.Tensor | None, y: torch.Tensor, k: int) -> torch.Tensor:
        return self._stage(x, z, y, k)[0]

    # ------------------------------------------------------------------ full model
    def forward(self, y: torch.Tensor, x0: torch.Tensor, return_stages: bool = False, return_terms: bool = False):
        """y: raw sinogram (B,1,V,D); x0: FBP image (B,1,H,W). Return options as in ``UnrolledLEARN.forward``."""
        if self.projector is None:
            raise RuntimeError("No projector set (call set_projector)")
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        x: torch.Tensor = x0
        z: torch.Tensor | None = None
        hist: list = [None, None, [], []]
        xs: list[torch.Tensor] = [x0]
        terms: list[dict] = []
        qn = self.data_step == "lbfgs"
        for k in range(self.K):
            # L-BFGS: the data direction is computed outside the stage function (it needs earlier stages); FBP: inside
            data = -self.alphas[k] * self._qn_direction(x, y, hist) if qn else None
            if return_terms:
                with torch.no_grad():
                    dc = data if data is not None else self._data_step(x, y, k)
                    step = {"alpha": self.alphas[k].item()} if qn else {"weight": self.w[k].item()}
                    terms.append({"k": k, "dc_norm": self._norm(dc),
                                  "reg_norm": self._norm(self.stages[k](x, z)[0]), **step})
            third = data if data is not None else y
            if k == self.K - 1:      # the last stage emits no features: return only the image so checkpointing works
                last = self._stage_with_data_last if qn else self._stage_last
                x = checkpoint(last, x, z, third, k, use_reentrant=False) if use_ckpt else last(x, z, third, k)
            else:
                stage = self._stage_with_data if qn else self._stage
                if use_ckpt:
                    x, z = checkpoint(stage, x, z, third, k, use_reentrant=False)
                else:
                    x, z = stage(x, z, third, k)
            if return_stages:
                xs.append(x)
        out = xs if return_stages else x
        return (out, terms) if return_terms else out
