"""Flat-detector fan-beam FBP in PyTorch (CPU or GPU, no ASTRA needed).

Algorithm (Kak & Slaney, section 3.4.2, equispaced flat detector), using a virtual detector through the rotation axis:
    1. Cosine weighting: R'(beta, s) = R(beta, s) * D / sqrt(D^2 + s^2)
    2. Ramp filtering along s: Q = R' * h   (discrete h: h(0)=1/(4 tau^2), h(odd)=-1/(n pi tau)^2)
    3. Weighted backprojection: f(x, y) = c * sum_beta (D / L)^2 Q(beta, s'(x, y)) * dbeta
       with L = source-to-point distance along the central ray and s' = D * (lateral coordinate) / L.
Factor c: 0.5 for a 360° scan (every line is measured twice), 1.0 for a limited scan; short scans use Parker weights.

Image grid convention as in ASTRA: column j -> x = (j + 0.5 - n/2) * px, row i -> y = (n/2 - i - 0.5) * px.
The input is the UNNORMALISED sinogram (units: normalised intensity x mm); the output is on the image scale.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ...imaging.geometry import FanBeamGeometry
from .filters import parker_weights, ramp_filter_freq


class _Backproject(torch.autograd.Function):
    """Differentiable FBP backprojection: the forward pass keeps no autograd graph and the backward pass applies the
    explicit adjoint (``backproject_t``). Letting autograd trace ``gather`` would keep ~1 GB of indices per call
    on the 256 grid."""

    @staticmethod
    def forward(ctx, Q: torch.Tensor, fbp: FanBeamFBP) -> torch.Tensor:
        ctx.fbp = fbp
        return fbp.backproject(Q)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return ctx.fbp.backproject_t(g), None


class FanBeamFBP:
    """FBP for a ``FanBeamGeometry`` on an ``img_size`` x ``img_size`` grid."""

    # Sign of gamma in the Parker weights, matching the rotation direction of the ASTRA convention (fixed by a test).
    PARKER_SIGN = -1.0

    def __init__(self, geom: FanBeamGeometry, img_size: int, filt: str = "ram-lak", cutoff: float = 1.0,
                 parker: bool | None = None, device="cpu", view_chunk: int = 32, sparse: bool | str = "auto"):
        """``sparse``: backproject with a sparse matrix built once (same interpolation points and weights as the
        per-view-chunk path; only the float32 summation order differs). "auto" = use it on GPUs."""
        if sparse not in (True, False, "auto"):
            raise ValueError(f"sparse must be True, False or 'auto', got {sparse!r}")
        self.sparse = sparse
        self._mats: tuple[torch.Tensor, torch.Tensor] | None = None
        self.geom = geom
        self.img_size = int(img_size)
        self.filt, self.cutoff = filt, cutoff
        self.view_chunk = int(view_chunk)
        D = geom.dso_mm
        self.D = D
        self.tau = geom.det_spacing_mm * D / geom.dsd_mm
        s = geom.det_coords_mm() * D / geom.dsd_mm
        w_cos = (D / np.sqrt(D ** 2 + s ** 2)).astype(np.float32)
        self.P = int(2 ** math.ceil(math.log2(2 * geom.n_det)))
        H = ramp_filter_freq(geom.n_det, self.tau, filt, cutoff).astype(np.float32)
        if parker is None:
            parker = (not geom.is_full_scan) and geom.span_deg >= geom.short_scan_deg - 1e-6
        self.parker = bool(parker)
        wv = np.ones((geom.n_views, geom.n_det), np.float32)
        if self.parker:
            wv = parker_weights(geom, self.PARKER_SIGN)
        self.factor = 0.5 if (geom.is_full_scan and not self.parker) else 1.0
        self.dbeta = math.radians(geom.step_deg)
        px = geom.pixel_size_mm(img_size)
        c = (np.arange(img_size) + 0.5 - img_size / 2) * px
        Y, X = np.meshgrid(-c, c, indexing="ij")  # row i -> y = -(i+0.5-n/2)px; column j -> x
        th = geom.angles_rad()
        self._np = dict(w=(w_cos[None, :] * wv), H=H, X=X.ravel(), Y=Y.ravel(),
                        cos=np.cos(th), sin=np.sin(th))
        self.device: torch.device | None = None
        self.to(device)

    def to(self, device) -> FanBeamFBP:
        device = torch.device(device)
        if device == self.device:
            return self
        f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)
        self.w = f(self._np["w"])
        self.H = f(self._np["H"])
        self.X, self.Y = f(self._np["X"]), f(self._np["Y"])
        self.cos, self.sin = f(self._np["cos"]), f(self._np["sin"])
        self.device = device
        self._mats = None
        return self

    # ------------------------------------------------------------------ steps
    def filter(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B,1,V,D) -> weighted and filtered sinogram, (B,V,D)."""
        q = y[:, 0].float() * self.w
        Q = torch.fft.irfft(torch.fft.rfft(q, n=self.P, dim=-1) * self.H, n=self.P, dim=-1)
        return Q[..., : self.geom.n_det]

    def _rays(self, v0: int, v1: int, nd: int):
        """For views v0..v1: left detector index i0 (clamped), fractional part fr and weight (D/L)^2 (zero when the
        ray through the pixel misses the detector) for every pixel; each array has shape (views, pixels)."""
        c = self.cos[v0:v1, None]
        s = self.sin[v0:v1, None]
        L = self.D - self.X[None] * s + self.Y[None] * c
        lat = self.X[None] * c + self.Y[None] * s
        idx = self.D * lat / L / self.tau + 0.5 * (nd - 1)
        i0 = torch.floor(idx)
        fr = idx - i0
        i0 = i0.long()
        wgt = (self.D / L) ** 2 * ((i0 >= 0) & (i0 < nd - 1))
        return i0.clamp(0, nd - 2), fr, wgt

    def _use_sparse(self) -> bool:
        return self.sparse is True or (self.sparse == "auto" and self.device is not None and self.device.type == "cuda")

    def _matrices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Backprojection matrix M (pixels x (views * channels)) and M^T as CSR on the device, built once.

        The coefficient of each (pixel, view) pair comes straight from ``_rays`` (float32, on this very device):
        channel i0 gets (1 - fr) * w and channel i0 + 1 gets fr * w. Built on the CPU with SciPy to avoid temporary
        GPU memory."""
        if self._mats is None:
            import scipy.sparse as sp

            from ..projector import _to_torch_csr

            V, nd, N = self.geom.n_views, self.geom.n_det, self.X.numel()
            rows, cols, vals = [], [], []
            pix = np.arange(N, dtype=np.int64)
            for v0 in range(0, V, self.view_chunk):
                v1 = min(V, v0 + self.view_chunk)
                i0, fr, wgt = self._rays(v0, v1, nd)
                w0 = ((1 - fr) * wgt).cpu().numpy()
                w1 = (fr * wgt).cpu().numpy()
                base = (torch.arange(v0, v1, device=i0.device).view(-1, 1) * nd + i0).cpu().numpy()
                keep = wgt.cpu().numpy() != 0
                r = np.broadcast_to(pix, keep.shape)[keep]
                rows += [r, r]
                cols += [base[keep], base[keep] + 1]
                vals += [w0[keep], w1[keep]]
            M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(N, V * nd), dtype=np.float32)
            self._mats = (_to_torch_csr(M, self.device), _to_torch_csr(M.T.tocsr(), self.device))
        return self._mats

    def backproject(self, Q: torch.Tensor) -> torch.Tensor:
        """Backprojection weighted by (D/L)^2 with linear interpolation. Q: (B,V,D) -> (B,1,H,W)."""
        B, V, nd = Q.shape
        if self._use_sparse():
            M, _ = self._matrices()
            out = (M @ Q.reshape(B, V * nd).t().contiguous()).t()
            return (out * (self.dbeta * self.factor)).reshape(B, 1, self.img_size, self.img_size)
        out = torch.zeros(B, self.X.numel(), device=Q.device, dtype=Q.dtype)
        for v0 in range(0, V, self.view_chunk):
            v1 = min(V, v0 + self.view_chunk)
            i0, fr, wgt = self._rays(v0, v1, nd)
            q = Q[:, v0:v1]                                               # (B, c, nd)
            g0 = torch.gather(q, 2, i0.unsqueeze(0).expand(B, -1, -1))
            g1 = torch.gather(q, 2, (i0 + 1).unsqueeze(0).expand(B, -1, -1))
            out += ((g0 * (1 - fr) + g1 * fr) * wgt).sum(dim=1)
        img = out * (self.dbeta * self.factor)
        return img.view(B, 1, self.img_size, self.img_size)

    def backproject_t(self, img: torch.Tensor) -> torch.Tensor:
        """Adjoint of ``backproject``: (B,1,H,W) -> (B,V,D). Accumulates with ``index_add_`` on a flat array
        (deterministic on GPUs)."""
        B = img.shape[0]
        V, nd = self.geom.n_views, self.geom.n_det
        if self._use_sparse():
            _, MT = self._matrices()
            g = img.reshape(B, -1) * (self.dbeta * self.factor)
            return (MT @ g.t().contiguous()).t().reshape(B, V, nd)
        g = img.reshape(B, 1, -1) * (self.dbeta * self.factor)               # (B, 1, N)
        out = torch.zeros(B * V * nd, device=img.device, dtype=img.dtype)
        row = torch.arange(B, device=img.device).view(B, 1, 1) * (V * nd)
        for v0 in range(0, V, self.view_chunk):
            v1 = min(V, v0 + self.view_chunk)
            i0, fr, wgt = self._rays(v0, v1, nd)
            base = row + (torch.arange(v0, v1, device=img.device).view(1, -1, 1) * nd + i0)   # (B, c, N)
            out.index_add_(0, base.reshape(-1), (g * ((1 - fr) * wgt)).reshape(-1))
            out.index_add_(0, (base + 1).reshape(-1), (g * (fr * wgt)).reshape(-1))
        return out.view(B, V, nd)

    def apply(self, y: torch.Tensor) -> torch.Tensor:
        """FBP differentiable with respect to y (used inside models, e.g. the RegFormer data step). y: (B,1,V,D)."""
        return _Backproject.apply(self.filter(y), self)

    @torch.no_grad()
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = y.to(self.device)
        return self.backproject(self.filter(y))

    def __repr__(self) -> str:
        return (f"FanBeamFBP({self.geom.name}, img={self.img_size}, filter={self.filt}, "
                f"parker={self.parker}, factor={self.factor})")
