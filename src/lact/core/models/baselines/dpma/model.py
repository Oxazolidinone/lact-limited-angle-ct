"""The DPMA model: two-stage training (M1, then M2 with M1 frozen) and the non-learned refinement at inference."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ....operators.fbp import FanBeamFBP
from ....operators.projector import SparseFanBeam
from ....solvers.tv_chambolle import denoise_tv_chambolle
from ...unrolled import Unrolled
from .network import DPMANet


class DPMA(Unrolled):
    def __init__(self, stage: int = 2, width: int = 16, chunk_size: int = 144, sino_scale: float = 0.192,
                 refine_iters: int = 300, refine_step: float = 0.15, refine_lambda: float = 0.1,
                 refine_delta: float = 0.005, refine_tv_iters: int = 10, grad_checkpoint: bool = False,
                 projector=None, target_projector=None, fbp_filter: str = "ram-lak", fbp_cutoff: float = 1.0):
        super().__init__()
        if stage not in (1, 2):
            raise ValueError(f"DPMA: stage must be 1 (train M1) or 2 (M1 frozen, train M2), got {stage}")
        self.K = 1
        self.stage = int(stage)
        self.m1 = DPMANet(width, chunk_size=chunk_size)
        self.m2: DPMANet | None = None
        if self.stage == 2:
            self.m2 = DPMANet(width, chunk_size=chunk_size)
            for p in self.m1.parameters():
                p.requires_grad_(False)
        self.sino_scale = float(sino_scale)
        self.refine_iters, self.refine_step = int(refine_iters), float(refine_step)
        self.refine_lambda, self.refine_delta = float(refine_lambda), float(refine_delta)
        self.refine_tv_iters = int(refine_tv_iters)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.fbp_filter, self.fbp_cutoff = fbp_filter, float(fbp_cutoff)
        self.projector: SparseFanBeam | None = None
        self.target: SparseFanBeam | None = None        # projector over the extended scan range
        self.fbp: FanBeamFBP | None = None
        self.fbp_target: FanBeamFBP | None = None
        self.view_offset = 0
        if target_projector is not None:
            self.set_target(target_projector)
        if projector is not None:
            self.set_projector(projector)

    # ------------------------------------------------------------------ operators
    def set_target(self, target_projector) -> DPMA:
        self.target = target_projector
        # sparse=False keeps the per-view-chunk backprojection (same computation). The sparse matrix of the 240° target
        # costs ~2.5 GB more while stage 1 already uses ~24 GB, and the FBP is not the slow part of DPMA.
        self.fbp_target = FanBeamFBP(target_projector.geom, target_projector.img_size, self.fbp_filter,
                                     self.fbp_cutoff, device=target_projector.device, sparse=False)
        if self.projector is not None:
            self._align()
        return self

    def set_projector(self, projector) -> DPMA:
        self.projector = projector
        self.fbp = FanBeamFBP(projector.geom, projector.img_size, self.fbp_filter, self.fbp_cutoff,
                              device=projector.device, sparse=False)
        if self.target is not None:
            self._align()
        return self

    def _align(self) -> None:
        """Locate the measured views in the extended range; both angular grids must coincide view by view."""
        if self.projector is None or self.target is None:
            raise RuntimeError("DPMA: set both projectors before aligning view grids")
        g, t = self.projector.geom, self.target.geom
        same = (g.step_deg, g.n_det, g.det_spacing_mm, g.dso_mm, g.dod_mm, g.fov_mm) == (
            t.step_deg, t.n_det, t.det_spacing_mm, t.dso_mm, t.dod_mm, t.fov_mm)
        off = (g.angles_deg()[0] - t.angles_deg()[0]) / t.step_deg
        if not same or abs(off - round(off)) > 1e-6 or round(off) < 0 or round(off) + g.n_views > t.n_views:
            raise ValueError(f"DPMA: scan range {t.name} does not contain exactly the views of {g.name}")
        self.view_offset = round(off)

    def train(self, mode: bool = True) -> DPMA:
        super().train(mode)
        if self.stage == 2:
            self.m1.eval()                              # M1 is frozen: always in inference mode
        return self

    def stage_params(self) -> dict[str, torch.Tensor]:
        return {}

    def regularizer_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def regularization_only(self, x: torch.Tensor) -> None:
        self.m1(x)
        if self.m2 is not None:
            self.m2(x)

    def _ops(self, device):
        if self.projector is None or self.target is None or self.fbp is None or self.fbp_target is None:
            raise RuntimeError("Projector / extended-range operator not set (set_projector, set_target)")
        if self.target.device != device:
            self.target = self.target.to(device)
        return self.projector, self.target, self.fbp.to(device), self.fbp_target.to(device)

    # ------------------------------------------------------------------ the three parts of the method
    def _stage1(self, y: torch.Tensor, x0: torch.Tensor):
        """Return (p_hat multiplied by sino_scale, x_a, x_b)."""
        _, target, _, fbp_t = self._ops(y.device)
        with torch.no_grad():
            p_in = target.raw_forward(x0)
            p_in[:, :, self.view_offset:self.view_offset + y.shape[2]] = y
            p_in = p_in * self.sino_scale
        p_hat = self.m1(p_in)
        x_a = fbp_t.apply(p_hat / self.sino_scale)
        return p_hat, x_a, self.m1(x_a)

    def _data_consistency(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        proj, _, fbp, _ = self._ops(y.device)
        return x - fbp.apply(proj.raw_forward(x) - y)

    @torch.no_grad()
    def refine(self, net: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """The 300 iterations of the original ``iter.py``, started from the network image ``net``."""
        proj, _, fbp, _ = self._ops(y.device)
        lam = self.refine_lambda
        ata = fbp.apply(proj.raw_forward(torch.ones_like(net[:1]))).clamp_min(0.0)
        x = net.clone()
        v = torch.zeros_like(net)
        f = torch.zeros_like(net)
        for _ in range(self.refine_iters - 1):
            residual = fbp.apply(proj.raw_forward(x) - y)
            x = (x - self.refine_step * (residual + lam * (x - net - v - f)) / (ata + lam)).clamp_min(0.0)
            diff = x - net - f
            lo = diff.amin(dim=(1, 2, 3), keepdim=True)
            span = (diff.amax(dim=(1, 2, 3), keepdim=True) - lo).clamp_min(1e-12)
            v = denoise_tv_chambolle((diff - lo) / span, weight=self.refine_delta,
                                     max_iter=self.refine_tv_iters) * span + lo
            f = f + (v - x + net)
        return x

    # ------------------------------------------------------------------ trainer interface
    def training_loss(self, y: torch.Tensor, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The original loss of the stage being trained (the trainer calls it instead of the output MSE)."""
        if self.stage == 1:
            _, target, _, _ = self._ops(y.device)
            p_hat, x_a, x_b = self._stage1(y, x0)
            with torch.no_grad():
                p_true = target.raw_forward(x) * self.sino_scale
            return 0.1 * F.l1_loss(p_hat, p_true) + 0.1 * F.l1_loss(x_a, x) + F.l1_loss(x_b, x)
        with torch.no_grad():
            x_b = self._stage1(y, x0)[2]
        return F.mse_loss(self._stage2_net()(x_b), x)

    def _stage2_net(self) -> DPMANet:
        if self.m2 is None:
            raise RuntimeError("DPMA: the stage-2 network exists only when stage == 2")
        return self.m2

    def validation_forward(self, y: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """Image of the learned part (without the 300 refinement iterations), used for checkpoint selection."""
        x_b = self._stage1(y, x0)[2]
        if self.stage == 1:
            return x_b
        return self._stage2_net()(self._data_consistency(x_b, y))

    def forward(self, y: torch.Tensor, x0: torch.Tensor, return_stages: bool = False, return_terms: bool = False):
        """y: raw sinogram (B,1,V,D); x0: FBP image (B,1,H,W). Stage-2 inference includes the 300 refinement steps."""
        net = self.validation_forward(y, x0)
        out = net
        if self.stage == 2 and not self.training and self.refine_iters > 1:
            out = self.refine(net, y)
        res = [x0, out] if return_stages else out
        if return_terms:
            with torch.no_grad():
                terms = [{"k": 0, "dc_norm": self._norm(out - net), "reg_norm": self._norm(net - x0)}]
            return res, terms
        return res
