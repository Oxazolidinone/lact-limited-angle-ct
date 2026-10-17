"""Projection-operator checks: the E0 operator check (pipeline step 03) and the E1 angle-density check (step 05)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ...config import Config
from ...core.imaging.geometry import FanBeamGeometry
from ...core.imaging.intensity import block_mean
from ...core.imaging.phantom import shepp_logan
from ...core.metrics import psnr
from ...core.operators import SparseFanBeam
from ...core.solvers import tv_pdhg
from ...infra.io.files import save_json
from ...infra.operators.operator_store import load_matrix
from ..data.staging import operator_dir
from ..factories import fbp_for, image_store, projector_for

log = logging.getLogger("lact")

# E0 pass thresholds (see docs/decisions.md)
E0_THRESHOLDS = {"adjoint_rel": 1e-5, "grad_rel_fp32": 1e-4, "fd_rel_fp64": 1e-6, "fbp_full_psnr": 30.0,
                 "fbp_short_psnr": 30.0, "mean_ratio_tol": 0.05, "sino_edge_ratio": 0.02}
# E1 pass threshold: |PSNR(0.125°) - PSNR(0.25°)| < 0.1 dB
E1_MAX_DELTA_DB = 0.1


# ---------------------------------------------------------------------------- step 03 (E0)
@dataclass
class OperatorCheck:
    geometry: FanBeamGeometry
    result: dict
    sinogram: np.ndarray
    fbp: np.ndarray

    @property
    def passed(self) -> bool:
        return bool(self.result["pass"])


def _psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(10 * np.log10(1.0 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


def _check_adjoint(P: SparseFanBeam) -> float:
    # positive inputs so the sums do not cancel (avoids spurious float32 relative errors)
    n = P.img_size
    x = torch.rand(1, 1, n, n, generator=torch.Generator().manual_seed(0))
    v = torch.rand(1, 1, P.n_views, P.n_det, generator=torch.Generator().manual_seed(1))
    lhs = float((P.raw_forward(x) * v).double().sum())
    rhs = float((x * P.raw_adjoint(v)).double().sum())
    return abs(lhs - rhs) / abs(lhs)


def _check_gradients(P: SparseFanBeam, A, rng: np.random.Generator) -> tuple[float, float]:
    """(float32 autograd versus float64 A^T(Ax-y), float64 central differences)."""
    n = P.img_size
    A64 = A.astype(np.float64)
    x64 = rng.random(n * n)
    y64 = A64 @ rng.random(n * n)
    g_ref = A64.T @ (A64 @ x64 - y64)
    xt = torch.tensor(x64, dtype=torch.float32).view(1, 1, n, n).requires_grad_(True)
    yt = torch.tensor(y64, dtype=torch.float32).view(1, 1, P.n_views, P.n_det)
    (0.5 * ((P.raw_forward(xt) - yt) ** 2).sum()).backward()
    assert xt.grad is not None
    g_t = xt.grad.double().numpy().ravel()
    grad_rel = float(np.linalg.norm(g_t - g_ref) / np.linalg.norm(g_ref))
    d = rng.standard_normal(n * n)
    eps = 1e-4

    def f(z):
        return 0.5 * np.sum((A64 @ z - y64) ** 2)

    fd = (f(x64 + eps * d) - f(x64 - eps * d)) / (2 * eps)
    return grad_rel, float(abs(fd - g_ref @ d) / abs(fd))


def check_operator(cfg: Config, geom: FanBeamGeometry, rng: np.random.Generator | None = None,
                   thresholds: dict | None = None) -> OperatorCheck:
    """Adjoint, gradient, FBP on a fine-grid simulated phantom, sinogram edges, data-consistency floor."""
    th = thresholds or E0_THRESHOLDS
    rng = rng or np.random.default_rng(0)
    n_sim, n_rec = cfg.data.img_size_sim, cfg.data.img_size_rec
    ph_sim = shepp_logan(n_sim)
    ph_rec = block_mean(ph_sim, n_sim // n_rec)
    A_rec, meta = load_matrix(geom, n_rec, operator_dir(cfg, geom, n_rec))
    A_sim, _ = load_matrix(geom, n_sim, operator_dir(cfg, geom, n_sim))
    P = SparseFanBeam(A_rec, geom, n_rec, norm=meta.get("opnorm"))
    r: dict = {"geometry": geom.to_dict(), "nnz": int(A_rec.nnz), "opnorm": P.norm}
    r["adjoint_rel"] = _check_adjoint(P)
    r["grad_rel_fp32"], r["fd_rel_fp64"] = _check_gradients(P, A_rec, rng)

    y = torch.from_numpy((A_sim @ ph_sim.ravel()).astype(np.float32)).view(1, 1, geom.n_views, geom.n_det)
    fbp = fbp_for(cfg, geom)
    rec = fbp(y)[0, 0].numpy()
    s = y[0, 0].numpy()
    r["fbp_psnr"] = _psnr_np(rec, ph_rec)
    r["fbp_mean_ratio"] = float(rec.mean() / ph_rec.mean())
    r["fbp_parker"] = fbp.parker
    r["sino_edge_ratio"] = float(max(np.abs(s[:, :2]).max(), np.abs(s[:, -2:]).max()) / np.abs(s).max())
    r["dc_floor_phantom"] = float(np.linalg.norm(A_rec @ ph_rec.ravel() - s.ravel()) / np.linalg.norm(s))

    checks = {"adjoint": r["adjoint_rel"] < th["adjoint_rel"],
              "grad_fp32": r["grad_rel_fp32"] < th["grad_rel_fp32"],
              "fd_fp64": r["fd_rel_fp64"] < th["fd_rel_fp64"],
              "sino_edge": r["sino_edge_ratio"] < th["sino_edge_ratio"]}
    if geom.is_full_scan:
        checks["fbp_full"] = r["fbp_psnr"] > th["fbp_full_psnr"]
        checks["fbp_scale"] = abs(r["fbp_mean_ratio"] - 1) < th["mean_ratio_tol"]
    elif fbp.parker:
        checks["fbp_short"] = r["fbp_psnr"] > th["fbp_short_psnr"]
        checks["fbp_scale"] = abs(r["fbp_mean_ratio"] - 1) < th["mean_ratio_tol"]
    r["checks"] = checks
    r["pass"] = all(checks.values())
    log.info("%-10s adj %.1e | grad %.1e | fd %.1e | FBP %.2f dB (mean ratio %.3f) | edge %.3f | DC floor %.4f | %s",
             geom.name, r["adjoint_rel"], r["grad_rel_fp32"], r["fd_rel_fp64"], r["fbp_psnr"], r["fbp_mean_ratio"],
             r["sino_edge_ratio"], r["dc_floor_phantom"], "PASS" if r["pass"] else "FAIL")
    return OperatorCheck(geom, r, s, rec)


def check_operators(cfg: Config, geoms: list[FanBeamGeometry], figure: bool = True) -> dict[str, bool]:
    """E0 for several geometries; writes <outputs>/checks/e0_report.json (+ figure). Returns {geometry: passed?}."""
    out = cfg.paths.checks
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    results = [check_operator(cfg, g, rng) for g in geoms]
    save_json({"thresholds": E0_THRESHOLDS, "results": {c.geometry.name: c.result for c in results}},
              out / "e0_report.json")
    if figure:
        from ...viz.plots import operator_check_figure

        n_sim, n_rec = cfg.data.img_size_sim, cfg.data.img_size_rec
        operator_check_figure(block_mean(shepp_logan(n_sim), n_sim // n_rec),
                              [(c.geometry.name, c.geometry.n_views, c.geometry.n_det, c.sinogram, c.fbp,
                                c.result["fbp_psnr"]) for c in results], out / "e0_fbp_phantom.png")
    ok = all(c.passed for c in results)
    log.info("E0 %s -> %s", "PASS" if ok else "FAILED", out / "e0_report.json")
    return {c.geometry.name: c.passed for c in results}


# ---------------------------------------------------------------------------- step 05 (E1)
def angle_density(cfg: Config, pairs: list[tuple[FanBeamGeometry, FanBeamGeometry]], n_val: int = 20,
                  tv_lambda: float = 1e-4, tv_iters: int = 300, device="cpu") -> pd.DataFrame:
    """Compare FBP and TV between the standard and the half angular step (each pair: (standard geometry, *_half)).

    Samples: the Shepp-Logan phantom + ``n_val`` evenly spaced slices of the first validation patient.
    Writes <outputs>/checks/e1_angle_density{,_raw}.csv and e1_summary.json.
    """
    n_sim, n_rec = cfg.data.img_size_sim, cfg.data.img_size_rec
    samples = {"phantom": shepp_logan(n_sim)[None]}
    store = image_store(cfg)
    val_pid = cfg.split.val[0]
    n = store.n_slices(val_pid)
    idx = sorted(set(np.linspace(0, n - 1, min(n_val, n)).round().astype(int).tolist()))
    samples[val_pid] = store.read(val_pid, "img512", idx)

    rows = []
    for base, half in pairs:
        span = round(base.span_deg)
        for geom in (base, half):
            step = geom.step_deg
            Ps = projector_for(cfg, geom, n_sim, device, with_adjoint=False)
            Pr = projector_for(cfg, geom, n_rec, device)
            fbp = fbp_for(cfg, geom, device)
            for name, arr in samples.items():
                x_sim = torch.from_numpy(arr).float().unsqueeze(1).to(device)
                x_ref = torch.from_numpy(block_mean(arr, n_sim // n_rec)).float().unsqueeze(1).to(device)
                y = Ps.raw_forward(x_sim)
                x0 = fbp(y)
                xt = tv_pdhg(Pr, y, tv_lambda, tv_iters, x_init=x0)
                for method, xr in (("fbp", x0), ("tv", xt)):
                    rows.append({"span": span, "step": step, "sample": name, "method": method,
                                 "psnr": float(psnr(xr, x_ref).mean())})
            log.info("span %d° step %.3f° done (%d views)", span, step, geom.n_views)
    df = pd.DataFrame(rows)
    steps = sorted(df["step"].unique())
    piv = df.pivot_table(index=["span", "sample", "method"], columns="step", values="psnr").reset_index()
    fine, coarse = steps[0], steps[-1]
    piv["delta_db"] = piv[fine] - piv[coarse]
    piv["pass"] = piv["delta_db"].abs() < E1_MAX_DELTA_DB
    out = cfg.paths.checks
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "e1_angle_density_raw.csv", index=False)
    piv.to_csv(out / "e1_angle_density.csv", index=False)
    save_json({"all_pass": bool(piv["pass"].all()), "tv_lambda": tv_lambda, "tv_iters": tv_iters},
              out / "e1_summary.json")
    log.info("E1 %s", "PASS" if piv["pass"].all() else "NOT PASSED: see e1_angle_density.csv")
    return piv
