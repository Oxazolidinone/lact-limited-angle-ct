"""LEARN with split visible / hidden branches (design of 2026-10-11; this file only runs TRIAL INFERENCE, no training).

At each LEARN stage the regularizer step is split by the project's visibility operator (x = P_vis x + P_inv x):

    x^{k+1} = x^k - alpha_k A^T(A x^k - y) - beta_k P_vis r1(x^k) - gamma_k P_inv r2(x^k)

r1, r2 are TWO output channels of one backbone; beta_k is the visible-branch step, gamma_k the hidden-branch step.
Plain LEARN is the special case r1 = r2, gamma_k = beta_k (because P_vis + P_inv = I).

Trial on a personal machine: load a trained LEARN run into BOTH branches (doubled output layer, gamma = beta) and run
inference on the illustrative slices in figdata. Sinograms are re-projected here from the 256 x 256 true images with a
Siddon matrix (personal machines have no data store and no ASTRA), so PSNR differs from the official run numbers;
compare only rows within the same table.

    python experiments/analysis/learn_split.py --runs ../ket-qua/vast_results_2026-10-07/outputs/runs/e3_main \\
        --images ../ket-qua/vast_results_2026-10-07/outputs/figdata/images.npz \\
        --ops <directory with A_<geometry>_256.npz> \\
        --run fan_a120=e3_main_learn_resnet_fan_a120_lr0.0003_s1 fan_a90=e3_main_learn_resnet_fan_a90_lr0.0003_s2

Besides the "loaded into both branches" row (which must match the original run), the script probes the branches by
scaling each branch's step at inference: hidden branch off, visible branch off, hidden step up / down.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from lact.config import load_config_file, load_geometry
from lact.core.analysis.visibility import VisibilityOperator
from lact.core.metrics import psnr, ssim
from lact.core.models import build_regularizer
from lact.core.models.blocks import zero_init
from lact.core.models.unrolled import UnrolledLEARN
from lact.core.operators import FanBeamFBP, SparseFanBeam


class SplitLEARN(UnrolledLEARN):
    """First-order LEARN with the regularizer step split into a visible branch (beta_k, output 0) and a hidden branch
    (gamma_k, output 1)."""

    def __init__(self, make_regularizer, visibility: VisibilityOperator, **kw):
        super().__init__(make_regularizer, **kw)
        if self.update != "gd" or not self.share_theta:
            raise ValueError("SplitLEARN supports only update = 'gd' with a shared regularizer")
        head = getattr(self.reg, "out", None)
        if not isinstance(head, nn.Conv2d) or head.out_channels != 1:
            raise ValueError("the regularizer must have a single-channel Conv2d output layer `out` (encoder-decoder)")
        self.reg.out = zero_init(nn.Conv2d(head.in_channels, 2, head.kernel_size, padding=head.padding))
        self.g_inv = nn.Parameter(self.b.detach().clone())            # gamma_k, parameterised like beta_k
        self.V = visibility
        self.probe = (1.0, 1.0)      # step multipliers (visible, hidden branch); ONLY for probing at inference

    @property
    def gammas_inv(self) -> torch.Tensor:
        return F.softplus(self.g_inv)

    def stage_params(self) -> dict[str, torch.Tensor]:
        return {**super().stage_params(), "gammas_inv": self.gammas_inv}

    def branches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(P_vis r1, P_inv r2) of image x."""
        r = self.reg(x)
        return self.V.vis(r[:, :1]), self.V.inv(r[:, 1:])

    def _stage(self, x: torch.Tensor, y_n: torch.Tensor, k: int) -> torch.Tensor:
        P = self.projector
        dc = P.adjoint(P.forward(x) - y_n)
        rv, ri = self.branches(x)
        return x - self.alphas[k] * dc - self.probe[0] * self.betas[k] * rv - self.probe[1] * self.gammas_inv[k] * ri

    def load_base(self, state: dict) -> None:
        """Load a plain LEARN run's weights into both branches: doubled output layer, gamma_k = beta_k."""
        new = {k: v.clone() for k, v in state.items()}
        new["reg.out.weight"] = state["reg.out.weight"].repeat(2, 1, 1, 1)
        new["reg.out.bias"] = state["reg.out.bias"].repeat(2)
        new["g_inv"] = state["b"].clone()
        self.load_state_dict(new)


@torch.no_grad()
def split_mse(V: VisibilityOperator, out: torch.Tensor, x: torch.Tensor) -> tuple[float, float]:
    """MSE (images clipped to [0, 1]) split into visible and invisible parts, averaged over slices."""
    e = out.clamp(0, 1) - x
    return float((e * V.vis(e)).flatten(1).mean(1).mean()), float((e * V.inv(e)).flatten(1).mean(1).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="directory of trained LEARN runs (config.yaml, best.pt)")
    ap.add_argument("--images", required=True, help="figdata/images.npz (keys <geometry>/gt/<patient>/<slice>)")
    ap.add_argument("--ops", required=True, help="directory with A_<geometry>_256.npz matrices")
    ap.add_argument("--run", nargs="+", required=True, help="<geometry>=<run name>")
    ap.add_argument("--n-dirs", type=int, default=16)
    a = ap.parse_args()
    torch.manual_seed(0)
    z = np.load(a.images)
    for item in a.run:
        geom_name, run_name = item.split("=", 1)
        run = Path(a.runs) / run_name
        cfg = load_config_file(run / "config.yaml")
        m, n = cfg.require_model(), int(cfg.data.img_size_rec)
        geom = load_geometry(geom_name)
        P = SparseFanBeam(sp.load_npz(Path(a.ops) / f"A_{geom_name}_{n}.npz"), geom, n)
        keys = sorted(k for k in z.files if k.startswith(f"{geom_name}/gt/"))
        x = torch.from_numpy(np.stack([z[k] for k in keys]).astype(np.float32))[:, None]
        y = P.raw_forward(x)
        x0 = FanBeamFBP(geom, n)(y)
        V = VisibilityOperator(geom, n, a.n_dirs)
        state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)["model"]

        def make(reg=m.regularizer):
            return build_regularizer(reg.name, **reg.params)

        base = UnrolledLEARN(make, K=m.K, projector=P).eval()
        base.load_state_dict(state)
        split = SplitLEARN(make, V, K=m.K, projector=P).eval()
        split.load_base(state)
        n_base, n_split = (sum(p.numel() for p in mod.parameters()) for mod in (base, split))
        print(f"\n== {run_name} | {geom_name} | {len(keys)} slices: {[k.split('/', 2)[2] for k in keys]}")
        print(f"   parameters: original LEARN {n_base:,} | split branches {n_split:,} (+{n_split - n_base})")
        with torch.no_grad():
            ref = base(y, x0)
        rows = [("FBP", x0), ("original LEARN", ref)]
        for name, probe in [("Split, loaded into both branches (visible x1, hidden x1)", (1.0, 1.0)),
                            ("  hidden branch off (visible x1, hidden x0)", (1.0, 0.0)),
                            ("  visible branch off (visible x0, hidden x1)", (0.0, 1.0)),
                            ("  hidden branch x0.5", (1.0, 0.5)), ("  hidden branch x1.25", (1.0, 1.25)),
                            ("  hidden branch x1.5", (1.0, 1.5)), ("  visible branch x1.5", (1.5, 1.0))]:
            split.probe = probe
            with torch.no_grad():
                rows.append((name, split(y, x0)))
        same = float((rows[2][1] - ref).abs().max())
        print(f"   largest difference between 'loaded into both branches' and original LEARN: {same:.2e} (image scale "
              "[0, 1])")
        print(f"   {'':52s} {'PSNR':>6s} {'SSIM':>7s} {'MSE vis':>10s} {'MSE hid':>10s}   (MSE x 1e-3)")
        for name, out in rows:
            v, i = split_mse(V, out, x)
            print(f"   {name:52s} {float(psnr(out, x).mean()):6.2f} {float(ssim(out, x).mean()):7.4f} "
                  f"{v * 1e3:10.4f} {i * 1e3:10.4f}")
        print("   per-slice PSNR (original LEARN):", " ".join(f"{float(p):.2f}" for p in psnr(ref, x)))


if __name__ == "__main__":
    main()
