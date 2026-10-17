"""Qualitative figures: reconstruction grid + zoomed ROI + error maps; intensity line profiles."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy import ndimage

from .style import AXIS, CMAP_SEQ, INK, INK_2, SURFACE, apply_style
from .windows import WINDOWS, apply_window, to_hu


def roi_by_fbp_error(fbp: np.ndarray, ref: np.ndarray, size: int = 48, margin: int = 16) -> tuple[int, int, int, int]:
    """Pick the ROI with a MODEL-INDEPENDENT criterion: where the local FBP error is largest."""
    err = ndimage.uniform_filter(np.abs(fbp - ref), size)
    H, W = err.shape
    err[: margin + size // 2] = err[H - margin - size // 2:] = 0
    err[:, : margin + size // 2] = err[:, W - margin - size // 2:] = 0
    r, c = np.unravel_index(np.argmax(err), err.shape)
    return int(r - size // 2), int(c - size // 2), size, size


def recon_grid(images: dict[str, np.ndarray], ref: np.ndarray, mu_ref: float, mu_water: float,
               window: str = "soft", roi: tuple[int, int, int, int] | None = None, zoom: int = 3,
               err_max_hu: float = 200.0, metrics: dict[str, tuple[float, float]] | None = None,
               title: str | None = None, out_path: str | Path | None = None, show_error: bool = True,
               max_cols: int = 9):
    """images: {label: normalised image (H,W)} in display order (reference excluded).

    More than ``max_cols`` methods are split into several bands, each starting with the reference image."""
    apply_style()
    W, L = WINDOWS[window]
    names = list(images)
    n_groups = max(1, math.ceil(len(names) / max_cols))
    per = math.ceil(len(names) / n_groups)
    rows_per = 2 if show_error else 1
    ncol = per + 1
    fig, axes = plt.subplots(n_groups * rows_per, ncol, squeeze=False,
                             figsize=(1.9 * ncol, (2.05 * rows_per + 0.3) * n_groups))
    ref_hu = to_hu(ref, mu_ref, mu_water)
    im = None
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    for gi in range(n_groups):
        group = names[gi * per: (gi + 1) * per]
        labels = ["Tham chiếu", *group]
        arrs = [ref, *[images[k] for k in group]]
        top = gi * rows_per
        for j in range(len(arrs), ncol):                  # the last band may have empty cells
            for r in range(rows_per):
                axes[top + r, j].set_facecolor(SURFACE)
        for j, (lab, arr) in enumerate(zip(labels, arrs)):
            ax = axes[top, j]
            hu = to_hu(arr, mu_ref, mu_water)
            ax.imshow(apply_window(hu, W, L), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            head = lab
            if metrics and lab in metrics:
                p, s_ = metrics[lab]
                head = f"{lab}\n{p:.2f} dB | {s_:.4f}"
            ax.set_title(head, fontsize=8 if n_groups == 1 else 7, color=INK)
            if roi is not None:
                r0, c0, h, w = roi
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), w, h, fill=False, ec="#eda100", lw=1.0))
                crop = apply_window(hu[r0: r0 + h, c0: c0 + w], W, L)
                ins = ax.inset_axes([0.0, 0.0, 0.42, 0.42])
                ins.imshow(np.kron(crop, np.ones((zoom, zoom))), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ins.set_xticks([]), ins.set_yticks([])
                for sp in ins.spines.values():
                    sp.set_edgecolor("#eda100")
                    sp.set_linewidth(0.8)
            if show_error:
                axe = axes[top + 1, j]
                if j == 0:
                    axe.text(0.5, 0.5, f"|sai số| (HU)\nthang 0–{err_max_hu:g}", ha="center", va="center",
                             color=INK_2, fontsize=8, transform=axe.transAxes)
                    axe.set_facecolor(SURFACE)
                    continue
                im = axe.imshow(np.abs(hu - ref_hu), cmap=CMAP_SEQ, vmin=0, vmax=err_max_hu, interpolation="nearest")
    if show_error and im is not None:
        cax = fig.add_axes((0.92, 0.08, 0.01, 0.35 / n_groups))
        cb = fig.colorbar(im, cax=cax)
        cb.ax.spines["outline"].set_edgecolor(AXIS)
        cb.ax.tick_params(labelsize=7, colors=INK_2)
    if title:
        fig.suptitle(title, fontsize=10, color=INK, y=1.0)
    fig.subplots_adjust(wspace=0.04, hspace=0.12 if n_groups == 1 else 0.3, right=0.91)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def line_profile(images: dict[str, np.ndarray], ref: np.ndarray, mu_ref: float, mu_water: float, row: int,
                 cols: tuple[int, int], colors: dict[str, str], out_path: str | Path | None = None,
                 title: str | None = None):
    """HU intensity along one image row (across a bone / soft-tissue boundary)."""
    apply_style()
    c0, c1 = cols
    x = np.arange(c0, c1)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.plot(x, to_hu(ref, mu_ref, mu_water)[row, c0:c1], color=INK, lw=1.6, label="Tham chiếu")
    for lab, a in images.items():
        ax.plot(x, to_hu(a, mu_ref, mu_water)[row, c0:c1], color=colors.get(lab, INK_2), lw=1.2, label=lab)
    ax.set_xlabel("Cột ảnh (pixel)")
    ax.set_ylabel("HU")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    if title:
        ax.set_title(title, loc="left")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
