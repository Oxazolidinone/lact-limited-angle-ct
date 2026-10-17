"""Figure of the README: the limited-angle problem on a synthetic phantom (no patient data, no thesis results).

    python experiments/figures/make_readme_figures.py --out docs/images

(a) the scan geometry, (b) the full sinogram with the measured 90° range, (c) the Fourier coverage of the 90° operator,
(d)-(g) the phantom and its FBP from 360°, 120° and 90°. Projection matrices are built with the Siddon fallback when
ASTRA is not installed (a few minutes the first time) and cached in ``--operators``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Arc, Ellipse, Rectangle

from lact.config import load_geometry
from lact.core.analysis.spectral import coverage_map
from lact.core.imaging.phantom import random_body_phantom
from lact.core.metrics import psnr
from lact.core.operators import FanBeamFBP, SparseFanBeam
from lact.infra.operators.operator_store import build_and_save, load_matrix
from lact.viz import paper as pp

N = 256
BLUE, GREY, INK = "#2f6fd0", "#9a9a9a", "#1a1a1a"
WINDOW = (0.0, 0.55)          # display window on the normalised intensity scale


def projector(geom, op_dir: Path) -> SparseFanBeam:
    build_and_save(geom, N, op_dir)                       # no-op when the matrix is already cached
    A, meta = load_matrix(geom, N, op_dir)
    return SparseFanBeam(A, geom, N, norm=meta.get("opnorm"))


def draw_geometry(ax, span: float) -> None:
    """Fan-beam schematic: source trajectory, measured arc, one fan and the flat detector."""
    R, D = 1.0, 0.48
    ax.add_patch(Arc((0, 0), 2 * R, 2 * R, theta1=90 + span / 2, theta2=90 - span / 2 + 360, color=GREY, lw=1.2,
                     ls=(0, (3, 3))))
    ax.add_patch(Arc((0, 0), 2 * R, 2 * R, theta1=90 - span / 2, theta2=90 + span / 2, color=BLUE, lw=3.0))
    ax.add_patch(Ellipse((0, 0), 0.62, 0.44, fc="#e8e8e8", ec=INK, lw=0.8))
    ax.add_patch(Ellipse((0.1, 0.03), 0.16, 0.12, fc="#c9c9c9", ec="none"))
    ax.add_patch(Ellipse((0, -0.14), 0.07, 0.07, fc="#8c8c8c", ec="none"))
    src = np.array([0.0, R])
    half = np.deg2rad(26)
    for a in (-half, 0.0, half):
        end = np.array([np.tan(a) * (R + D), -D])
        ax.plot([src[0], end[0]], [src[1], end[1]], color=BLUE, lw=0.6, alpha=0.7)
    w = np.tan(half) * (R + D)
    ax.add_patch(Rectangle((-w, -D - 0.04), 2 * w, 0.04, fc=INK, ec="none"))
    ax.plot(*src, marker="o", ms=5, color=BLUE)
    ax.text(0, 1.13, f"measured {span:g}°", ha="center", va="bottom", color=BLUE, fontsize=8)
    ax.text(0, -1.17, f"missing {360 - span:g}°", ha="center", va="top", color=GREY, fontsize=8)
    ax.text(0, -D - 0.08, "detector", ha="center", va="top", fontsize=7, color=INK)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/images", help="output directory")
    ap.add_argument("--operators", default=".regression_cache/operators", help="cache of projection matrices")
    ap.add_argument("--seed", type=int, default=3, help="seed of the synthetic phantom")
    a = ap.parse_args()
    op_dir, out = Path(a.operators), Path(a.out)
    op_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    base = load_geometry("fan_a90")
    geoms = {"360°": base.replace(name="fan_full_1deg", span_deg=360.0, step_deg=1.0),
             "120°": load_geometry("fan_a120"), "90°": base}
    x = torch.from_numpy(random_body_phantom(N, np.random.default_rng(a.seed)).astype(np.float32))[None, None]
    recon, sino_full = {}, None
    for label, g in geoms.items():
        P = projector(g, op_dir)
        y = P.raw_forward(x)
        recon[label] = FanBeamFBP(g, N)(y)
        if label == "360°":
            sino_full = y[0, 0].numpy()
    cov = coverage_map(projector(base, op_dir))

    pp.use("ieee", lang="en")
    fig = plt.figure(figsize=(10.0, 5.3), dpi=200)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.28, wspace=0.12,
                          left=0.02, right=0.99, top=0.93, bottom=0.03)
    ax = fig.add_subplot(gs[0, 0])
    draw_geometry(ax, base.span_deg)
    ax.set_title("(a) Limited-angle fan-beam scan", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 1:3])
    lo, hi = -base.span_deg / 2, base.span_deg / 2
    ax.imshow(sino_full.T, cmap="gray", aspect="auto", extent=(-180, 180, base.n_det, 0))
    for x0, x1 in ((-180, lo), (hi, 180)):
        ax.axvspan(x0, x1, fc="white", alpha=0.72, lw=0)
        ax.axvspan(x0, x1, fc="none", ec="#b5b5b5", hatch="///", lw=0)
        ax.text((x0 + x1) / 2, base.n_det * 0.08, "not measured", ha="center", va="top", color="#6b6b6b", fontsize=8)
    for v in (lo, hi):
        ax.axvline(v, color=BLUE, lw=1.4)
    ax.text(0, base.n_det * 0.08, "measured", ha="center", va="top", color="white", fontsize=8, fontweight="bold")
    ax.set_xlabel("source angle (°)", fontsize=8)
    ax.set_ylabel("detector element", fontsize=8)
    ax.set_xticks([-180, -90, -45, 0, 45, 90, 180])
    ax.tick_params(labelsize=7)
    ax.set_title("(b) Sinogram: only 90° of the 360° rotation is measured", loc="left", fontsize=9)

    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(cov, cmap="magma", extent=(-0.5, 0.5, -0.5, 0.5), vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("$k_x$", fontsize=8, labelpad=1)
    ax.set_ylabel("$k_y$", fontsize=8, labelpad=1)
    for yy in (0.33, -0.33):
        ax.text(0, yy, "missing\nwedge", ha="center", va="center", color="white", fontsize=7.5, linespacing=1.0)
    ax.set_title("(c) Fourier coverage (90°)", loc="left", fontsize=9)

    panels = [("(d) Ground truth", x)] + [(f"({c}) FBP, {k}", v) for c, (k, v) in zip("efg", recon.items())]
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(img[0, 0].numpy(), cmap="gray", vmin=WINDOW[0], vmax=WINDOW[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cfcfcf")
        ax.set_title(title, loc="left", fontsize=9)
        if i:
            ax.text(0.97, 0.03, f"PSNR {float(psnr(img, x)):.1f} dB", transform=ax.transAxes, ha="right",
                    va="bottom", color="white", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
    fig.savefig(out / "limited_angle.png", dpi=220, facecolor="white")
    print(f"Wrote {out / 'limited_angle.png'}")


if __name__ == "__main__":
    main()
