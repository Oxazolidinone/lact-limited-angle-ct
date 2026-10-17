"""Check the 2DeteCT geometry on slices downloaded with all 360° (``lact data download-2detect --full ...``) before
training.

For each flip/rotation of the reference image (8 of them) and each detector direction: build A on the dt_full geometry
(256 grid), find the least-squares scale c and the relative residual ||c A x - y|| / ||y||. The right convention gives
a small residual (a few %) and c close to 1 (correct mu scale). Then shift the detector by half / one element to detect
a rotation-centre offset, compare the full-angle FBP of the measured data with the reference image, and plot the
reference, the full-angle FBP and the 90° and 120° FBPs for visual inspection.

    python experiments/analysis/detect_geometry_check.py --raw-dir data_raw/2detect --slices 1 3000 4600 \
        --out outputs_detect/geometry_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from lact.core.imaging.geometry import FanBeamGeometry
from lact.core.imaging.intensity import block_mean
from lact.core.operators.fbp import FanBeamFBP
from lact.datasets import detect
from lact.infra.operators.operator_store import build_matrix_any


def geometry(name: str) -> FanBeamGeometry:
    return FanBeamGeometry.from_dict(yaml.safe_load(open(f"configs/geometry/{name}.yaml")))


def rel_residual(Ax: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    c = float(Ax @ y / (Ax @ Ax))
    return float(np.linalg.norm(c * Ax - y) / np.linalg.norm(y)), c


def shift_det(y: np.ndarray, s: float) -> np.ndarray:
    grid = np.arange(y.shape[1], dtype=np.float64)
    return np.stack([np.interp(grid + s, grid, row) for row in y])


def psnr(a: np.ndarray, b: np.ndarray, peak: float) -> float:
    return float(10 * np.log10(peak ** 2 / np.mean((a - b) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data_raw/2detect")
    ap.add_argument("--slices", nargs="+", type=int, default=[1, 3000, 4600])
    ap.add_argument("--det-bin", type=int, default=8)
    ap.add_argument("--angle-bin", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="outputs_detect/geometry_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gfull = geometry("dt_full")
    A = build_matrix_any(gfull, a.size)[0].tocsr()
    nv = gfull.n_views
    report = {"slices": {}}
    for i in a.slices:
        z = np.load(Path(a.raw_dir) / f"slice{i:05d}.npz")
        if z["sino"].shape[0] < nv * a.angle_bin:
            raise SystemExit(f"slice {i} lacks the full 360° (download again with --full {i})")
        rows = []
        for flip in (True, False):
            y = detect.measured_sinogram(z, nv, a.det_bin, a.angle_bin, flip).astype(np.float64).ravel()
            for t in detect.TRANSFORMS:
                x = block_mean(detect.reference_mu(z, t), 1024 // a.size)
                r, c = rel_residual(A @ x.ravel(), y)
                rows.append({"flip_det": flip, "transform": t, "rel_residual": r, "scale_c": c})
        rows.sort(key=lambda d: d["rel_residual"])
        best = rows[0]
        y = detect.measured_sinogram(z, nv, a.det_bin, a.angle_bin, best["flip_det"]).astype(np.float64)
        x = block_mean(detect.reference_mu(z, best["transform"]), 1024 // a.size)
        Ax = A @ x.ravel()
        shifts = {s: rel_residual(Ax, shift_det(y, s).ravel())[0] for s in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)}
        fbp = FanBeamFBP(gfull, a.size, "ram-lak")
        rec_full = fbp(torch.from_numpy(y[None, None].astype(np.float32)))[0, 0].numpy()
        peak = float(np.percentile(x, 99.9))
        recs = {"ref": x, "fbp_360": rec_full}
        for name in ("dt_a90", "dt_a120"):
            g = geometry(name)
            yl = detect.measured_sinogram(z, g.n_views, a.det_bin, a.angle_bin, best["flip_det"])
            recs["fbp_" + name] = FanBeamFBP(g, a.size, "ram-lak")(torch.from_numpy(yl[None, None]))[0, 0].numpy()
        report["slices"][i] = {"ranking": rows[:6], "det_shift_residual": shifts,
                               "psnr_fbp360_vs_ref": psnr(rec_full, x, peak),
                               "corr_fbp360_ref": float(np.corrcoef(rec_full.ravel(), x.ravel())[0, 1])}
        print(f"slice {i}: best flip_det={best['flip_det']} transform={best['transform']} "
              f"residual {best['rel_residual']:.4f} c={best['scale_c']:.4f}; second {rows[1]['transform']}/"
              f"{rows[1]['flip_det']} {rows[1]['rel_residual']:.4f}; detector shifts {shifts}; "
              f"FBP 360° PSNR {report['slices'][i]['psnr_fbp360_vs_ref']:.2f} dB, "
              f"correlation {report['slices'][i]['corr_fbp360_ref']:.4f}", flush=True)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
            for k, (name, im) in enumerate(recs.items()):
                ax[k].imshow(im, cmap="gray", vmin=0, vmax=peak)
                ax[k].set_title(name)
                ax[k].axis("off")
            fig.tight_layout()
            fig.savefig(out / f"slice{i:05d}.png", dpi=110)
            plt.close(fig)
        except ImportError:
            pass
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out}/report.json")


if __name__ == "__main__":
    main()
