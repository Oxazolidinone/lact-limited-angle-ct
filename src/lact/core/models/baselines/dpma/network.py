"""DPMA networks M1 and M2: U-Net of MFF blocks (kernel-basis attention convolutions + LSH sparse non-local
attention), as in the original code."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class LayerNorm2d(nn.Module):
    """Per-pixel normalisation over channels."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class _KBA(torch.autograd.Function):
    """Convolution with a per-pixel kernel: a combination ``att`` of ``nset`` basis kernels ``w``.

    The derivative is written by hand (as in KBNet and the original code) to avoid storing the (B, HW, c * c/g * k^2)
    kernel tensor for the backward pass."""

    @staticmethod
    def forward(ctx, x, att, k, g, b, w):
        B, nset, H, W = att.shape
        kk, c = k * k, x.shape[1]
        ctx.k, ctx.g, ctx.c, ctx.kk, ctx.nset = k, g, c, kk, nset
        ctx.save_for_backward(x, att, b, w)
        att = att.reshape(B, nset, H * W).transpose(-2, -1)
        bias = att @ b
        attk = (att @ w).reshape(B, H * W, g, c // g, c // g * kk)
        uf = F.unfold(x, kernel_size=k, padding=k // 2).reshape(B, g, c // g * kk, H * W).permute(0, 3, 1, 2)
        out = (attk @ uf.unsqueeze(-1)).squeeze(-1).reshape(B, H * W, c) + bias
        return out.transpose(-1, -2).reshape(B, c, H, W)

    @staticmethod
    def backward(ctx, grad_output):
        x, att, b, w = ctx.saved_tensors
        k, g, c, kk, nset = ctx.k, ctx.g, ctx.c, ctx.kk, ctx.nset
        B, _, H, W = grad_output.shape
        att = att.reshape(B, nset, H * W).transpose(-2, -1)
        dbias = grad_output.reshape(B, c, H * W).transpose(-1, -2)
        db = att.transpose(-2, -1) @ dbias
        datt = dbias @ b.transpose(-2, -1)
        attk = (att @ w).reshape(B, H * W, g, c // g, c // g * kk)
        uf = F.unfold(x, kernel_size=k, padding=k // 2).reshape(B, g, c // g * kk, H * W).permute(0, 3, 1, 2)
        dx = dbias.reshape(B, H * W, g, c // g, 1)
        dattk = (dx @ uf.reshape(B, H * W, g, 1, c // g * kk)).reshape(B, H * W, -1)
        duf = attk.transpose(-2, -1) @ dx
        datt = datt + dattk @ w.transpose(-2, -1)
        dw = att.transpose(-2, -1) @ dattk
        duf = duf.permute(0, 2, 3, 4, 1).reshape(B, -1, H * W)
        dx = F.fold(duf, output_size=(H, W), kernel_size=k, padding=k // 2)
        datt = datt.transpose(-1, -2).reshape(B, nset, H, W)
        return dx, datt, None, None, db.sum(0, keepdim=True), dw.sum(0, keepdim=True)


class LSHAttention(nn.Module):
    """Sparse non-local attention (NLSN, Mei et al., 2021): each token attends only to tokens in its LSH bucket.

    ``chunk_size`` tokens per bucket; ``n_hashes`` hashing rounds are mixed by their bucket scores. The sequence must
    hold at least ``chunk_size`` tokens (an image too small for ``chunk_size`` raises instead of running wrongly)."""

    EVAL_SEED = 0

    def __init__(self, channels: int, n_hashes: int = 4, reduction: int = 4, chunk_size: int = 144,
                 res_scale: float = 1.0):
        super().__init__()
        self.n_hashes, self.reduction, self.chunk_size, self.res_scale = n_hashes, reduction, chunk_size, res_scale
        self.conv_match = nn.Sequential(nn.Conv2d(channels, channels // reduction, 3, padding=1))
        self.conv_assembly = nn.Sequential(nn.Conv2d(channels, channels, 1))

    def _rotations(self, c: int, buckets: int, like: torch.Tensor) -> torch.Tensor:
        shape = (1, c, self.n_hashes, buckets // 2)
        if self.training:
            return torch.randn(shape, dtype=like.dtype, device=like.device)
        gen = torch.Generator().manual_seed(self.EVAL_SEED)         # inference: same image, same result
        return torch.randn(shape, generator=gen).to(like)

    def _hash(self, buckets: int, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        rot = self._rotations(x.shape[-1], buckets, x).expand(n, -1, -1, -1)
        rotated = torch.einsum("btf,bfhi->bhti", x, rot)
        codes = torch.argmax(torch.cat([rotated, -rotated], dim=-1), dim=-1)
        offsets = torch.arange(self.n_hashes, device=x.device).reshape(1, -1, 1) * buckets
        return (codes + offsets).reshape(n, -1)

    @staticmethod
    def _with_neighbours(x: torch.Tensor) -> torch.Tensor:
        back = torch.cat([x[:, :, -1:], x[:, :, :-1]], dim=2)
        forward = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=2)
        return torch.cat([x, back, forward], dim=3)

    @staticmethod
    def _select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values.gather(1, indices[:, :, None].expand(-1, -1, values.shape[-1]))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        n, _, h, w = inp.shape
        x_embed = self.conv_match(inp).reshape(n, -1, h * w).permute(0, 2, 1)
        y_embed = self.conv_assembly(inp).reshape(n, -1, h * w).permute(0, 2, 1)
        length, c = x_embed.shape[-2:]
        cs, nh = self.chunk_size, self.n_hashes
        if length < cs:
            raise ValueError(f"LSHAttention: {length} tokens < chunk_size = {cs} (image too small for this setting)")
        buckets = min(length // cs + (length // cs) % 2, 128)
        codes = self._hash(buckets, x_embed).detach()
        _, indices = codes.sort(dim=-1)
        _, undo = indices.sort(dim=-1)
        mod = indices % length
        xb = self._select(x_embed, mod).reshape(n, nh, -1, c)
        yb = self._select(y_embed, mod).reshape(n, nh, -1, c * self.reduction)
        padding = cs - length % cs if length % cs else 0
        if padding:
            xb = torch.cat([xb, xb[:, :, -padding:]], dim=2)
            yb = torch.cat([yb, yb[:, :, -padding:]], dim=2)
        xb = xb.reshape(n, nh, -1, cs, c)
        yb = yb.reshape(n, nh, -1, cs, c * self.reduction)
        match = self._with_neighbours(F.normalize(xb, p=2, dim=-1, eps=5e-5))
        yb = self._with_neighbours(yb)
        raw = torch.einsum("bhkie,bhkje->bhkij", xb, match)
        bucket_score = torch.logsumexp(raw, dim=-1, keepdim=True)
        ret = torch.einsum("bukij,bukje->bukie", torch.exp(raw - bucket_score), yb)
        ret = ret.reshape(n, nh, -1, c * self.reduction)
        bucket_score = bucket_score.reshape(n, nh, -1)
        if padding:
            ret, bucket_score = ret[:, :, :-padding], bucket_score[:, :, :-padding]
        ret = self._select(ret.reshape(n, -1, c * self.reduction), undo).reshape(n, nh, length, c * self.reduction)
        bucket_score = bucket_score.reshape(n, -1).gather(1, undo).reshape(n, nh, length, 1)
        ret = torch.sum(ret * F.softmax(bucket_score, dim=1), dim=1)
        return ret.permute(0, 2, 1).reshape(n, -1, h, w) * self.res_scale + inp


class _ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        return self.conv2(F.silu(self.norm2(h))) + x


class MFFBlock(nn.Module):
    """Multi-scale fusion block (``MFF_s`` in the original): the KBA branch multiplied by the non-local and channel
    attention branches, followed by an FFN.

    Every layer of the original is created (including ``non_local_block`` in blocks that do not use it and ``conv11``
    in blocks that do) so that the parameter count matches the original."""

    def __init__(self, c: int, ffn_expand: int = 1, nset: int = 32, k: int = 3, gc: int = 4,
                 use_non_local: bool = True, chunk_size: int = 144):
        super().__init__()
        self.k, self.g, self.use_non_local = k, c // gc, use_non_local
        self.residual_block = _ResidualBlock(c)
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
        init.uniform_(self.b, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.non_local_block = LSHAttention(c, n_hashes=4, reduction=4, chunk_size=chunk_size)
        self.conv11 = nn.Sequential(nn.Conv2d(c, c, 1), nn.Conv2d(c, c, 3, padding=1, groups=c))
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv21 = nn.Conv2d(c, c, 3, padding=1, groups=c)
        interc = min(c, 32)
        self.conv2 = nn.Sequential(nn.Conv2d(c, interc, 3, padding=1, groups=interc), SimpleGate(),
                                   nn.Conv2d(interc // 2, nset, 1))
        self.conv211 = nn.Conv2d(c, nset, 1)
        self.conv3 = nn.Conv2d(c, c, 1)
        ffn = int(ffn_expand * c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.ga1 = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))
        self.attgamma = nn.Parameter(torch.full((1, nset, 1, 1), 1e-2))
        self.beta = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))
        self.gamma = nn.Parameter(torch.full((1, c, 1, 1), 1e-2))
        self.sg = SimpleGate()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        sca = self.residual_block(x) + self.sca(x)
        x1 = self.non_local_block(x) if self.use_non_local else self.conv11(x)
        att = self.conv2(x) * self.attgamma + self.conv211(x)
        uf = self.conv21(self.conv1(x))
        x = _KBA.apply(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        y = inp + self.conv3(x * x1 * sca) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class DPMANet(nn.Module):
    """Four-level U-Net of MFF blocks (``Network`` in the original).

    The highest resolution level uses no non-local attention.
    """

    def __init__(self, width: int = 16, middle: int = 2, enc: tuple[int, ...] = (2, 2, 2, 2),
                 dec: tuple[int, ...] = (2, 2, 2, 2), chunk_size: int = 144):
        super().__init__()
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()
        chan = width
        for i, num in enumerate(enc):
            self.encoders.append(nn.Sequential(*[MFFBlock(chan, use_non_local=i != 0, chunk_size=chunk_size)
                                                 for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2
        self.middle_blks = nn.Sequential(*[MFFBlock(chan, chunk_size=chunk_size) for _ in range(middle)])
        for i, num in enumerate(dec):
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[MFFBlock(chan, use_non_local=i != len(dec) - 1,
                                                          chunk_size=chunk_size) for _ in range(num)]))
        self.pad_to = 2 ** len(enc)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        h, w = inp.shape[-2:]
        inp = F.pad(inp, (0, (self.pad_to - w % self.pad_to) % self.pad_to,
                          0, (self.pad_to - h % self.pad_to) % self.pad_to))
        x = self.intro(inp)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        return (self.ending(x) + inp)[:, :, :h, :w]
