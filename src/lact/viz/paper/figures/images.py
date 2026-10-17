"""Image figures: reconstructions with zoomed regions and error maps, visibility maps."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge
from scipy import ndimage

from ...windows import apply_window, to_hu
from ..style import (
    CMAP_SEQ,
    CTX,
    INK,
    INK_2,
    ROI_COLOR,
    VIOLET,
    WHITE,
    despine_image,
    geom_label,
    key_of,
    label,
    panel_label,
    stroke,
    t,
    width,
)
from .layout import SHORT, _fit_font, _text_w


# ============================================================================ 1. qualitative comparison
def roi_by_structure(ref: np.ndarray, mu_ref: float, mu_water: float, size: int = 44,
                     window: tuple[float, float] = (400.0, 40.0), margin: int = 8) -> tuple[int, int, int, int]:
    """MODEL-INDEPENDENT ROI: the square fully inside the body with the most structure (gradient energy of the
    reference in the soft-tissue window). Avoids homogeneous regions and bone-only regions."""
    hu = to_hu(ref, mu_ref, mu_water)
    img = apply_window(hu, *window)
    gy, gx = np.gradient(img)
    energy = ndimage.uniform_filter(gx * gx + gy * gy, size)
    body = ndimage.binary_fill_holes(hu > -500)
    inside = ndimage.uniform_filter(body.astype(float), size) > 0.999
    H, W = ref.shape
    half = size // 2
    valid = np.zeros_like(inside)
    valid[margin + half:H - margin - half, margin + half:W - margin - half] = True
    score = np.where(inside & valid, energy, -np.inf)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return int(r - half), int(c - half), size, size


def qualitative(blocks: Sequence[Mapping], methods: Sequence[str], mu_ref: float, mu_water: float,
                window: tuple[float, float] = (400.0, 40.0), err_max: float = 150.0, zoom: bool = True,
                errors: bool = True, w: str | float = "full", show_metrics: bool = True, max_cols: int = 9):
    """Reconstruction grid: columns = reference + methods; each block (e.g. one scan range) has rows for the full
    images (with the ROI box), the zoomed ROI and the |error| map (HU, common scale). The HU window goes in the caption.
    More than ``max_cols`` methods split each block into several bands, each starting with the reference image and
    with its own title row (short labels without the "LEARN-" prefix).

    blocks: [{label, ref, images: {method: arr}, roi: (r0, c0, h, w), metrics: {method: (psnr, ssim)}}]
    """
    W = width(w)
    methods = list(methods)
    n_bands = max(1, math.ceil(len(methods) / max_cols))
    per = math.ceil(len(methods) / n_bands)
    bands = [methods[i * per:(i + 1) * per] for i in range(n_bands)]
    multi = n_bands > 1
    n = 1 + per
    rows_per = 1 + int(zoom) + int(errors)
    lm, rm, g, th, bg = 0.19, (0.46 if errors else 0.03), 0.022, 0.17, 0.08
    cell = (W - lm - rm - (n - 1) * g) / n
    small = CTX.preset.small
    titles = {m: t("ref_short") if m == "gt" else label(m) for m in ["gt", *methods]}
    tfs = small
    if max(len(x) for x in titles.values()) * 0.0078 * small > cell * 0.98:     # column titles wider than a cell
        if multi:
            titles = {m: x if m == "gt" else SHORT.get(key_of(m), x) for m, x in titles.items()}
        titles = {m: x.replace("-", "-\n", 1) if "-" in x else x for m, x in titles.items()}
        if multi:
            tfs = _fit_font(list(titles.values()), cell + g / 2, small)
        th = 0.08 + 2 * tfs / 72 * 1.25
    block_h = rows_per * cell + (rows_per - 1) * g
    # band = (block, methods, has a title row, gap above)
    strips: list[tuple[int, list[str], bool, float]] = []
    for bi in range(len(blocks)):
        for ki, band in enumerate(bands):
            gap = 0.0 if not strips else bg if not multi else 0.2 if ki == 0 else 0.05
            strips.append((bi, band, multi or bi == 0, gap))
    H = sum(gap + (th if titled else 0.0) + block_h for _, _, titled, gap in strips) + 0.03
    fig = plt.figure(figsize=(W, H))
    Wl, Ll = window

    def ax_at(r_top_in: float, c: int, hh: float = cell, extra_x: float = 0.0, ww: float = cell):
        x0 = lm + c * (cell + g) + extra_x
        return fig.add_axes((x0 / W, 1 - (r_top_in + hh) / H, ww / W, hh / H))

    y = 0.0
    spans: dict[int, list[float]] = {}
    for bi, band, titled, gap in strips:
        blk = blocks[bi]
        y += gap + (th if titled else 0.0)
        spans.setdefault(bi, [y, y])[1] = y + block_h
        ref = np.asarray(blk["ref"], np.float64)
        ref_hu = to_hu(ref, mu_ref, mu_water)
        r0, c0, rh, rw = blk.get("roi") or (0, 0, 0, 0)
        im_err = None
        for ci, m in enumerate(["gt", *band]):
            arr = ref if m == "gt" else np.asarray(blk["images"][m], np.float64)
            hu = to_hu(arr, mu_ref, mu_water)
            # --- full image
            ax = ax_at(y, ci)
            ax.imshow(apply_window(hu, Wl, Ll), cmap="gray", vmin=0, vmax=1, interpolation="none")
            despine_image(ax)
            if rh:
                ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), rw, rh, fill=False, ec=ROI_COLOR, lw=0.8))
            if titled:
                ax.set_title(titles[m], fontsize=tfs, pad=2.5, linespacing=1.0)
            if show_metrics and m != "gt" and m in blk.get("metrics", {}):
                p, s = blk["metrics"][m]
                mfs = min(small - 1.0, 6.5)
                txt = f"{p:.2f} / {s:.3f}"
                if multi and _text_w(txt, mfs) > cell * 0.9:        # narrow cell: PSNR and SSIM on two lines
                    txt = f"{p:.2f}\n{s:.3f}"
                ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=mfs, color=WHITE, ha="left",
                        va="bottom", linespacing=0.95 if multi else None, path_effects=stroke(1.4))
            yy = y + cell + g
            # --- zoomed ROI
            if zoom and rh:
                az = ax_at(yy, ci)
                az.imshow(apply_window(hu[r0:r0 + rh, c0:c0 + rw], Wl, Ll), cmap="gray", vmin=0, vmax=1,
                          interpolation="none")
                az.set_xticks([])
                az.set_yticks([])
                for s_ in az.spines.values():
                    s_.set_visible(True)
                    s_.set_edgecolor(ROI_COLOR)
                    s_.set_linewidth(0.8)
                yy += cell + g
            # --- error map
            if errors:
                ae = ax_at(yy, ci)
                despine_image(ae)
                if m == "gt":
                    lab, efs = t("abs_err"), small - 0.5
                    if _text_w(lab, efs) > cell:
                        lab = lab.replace(" (", "\n(")
                        efs = _fit_font([lab], cell, efs)
                    ae.text(0.5, 0.5, lab, ha="center", va="center", fontsize=efs, color=INK_2,
                            transform=ae.transAxes)
                else:
                    im_err = ae.imshow(np.abs(hu - ref_hu), cmap=CMAP_SEQ, vmin=0, vmax=err_max,
                                       interpolation="none")
        if errors and im_err is not None:
            cy = y + (rows_per - 1) * (cell + g)
            cax = fig.add_axes(((W - rm + 0.07) / W, 1 - (cy + cell) / H, 0.07 / W, cell / H))
            cb = fig.colorbar(im_err, cax=cax, ticks=[0, err_max / 2, err_max])
            cb.outline.set_linewidth(0.4)
            cb.ax.tick_params(labelsize=small - 1, width=0.4, length=2, pad=1.5)
            cb.ax.set_title("HU", fontsize=small - 1, pad=2)
        y += block_h
    for bi, (a, b) in spans.items():
        fig.text((lm - 0.07) / W, 1 - (a + b) / 2 / H, blocks[bi].get("label", ""), rotation=90,
                 ha="center", va="center", fontsize=CTX.preset.font, fontweight="bold")
    return fig


# ============================================================================ 6. visibility maps
def visibility(maps: Mapping[str, np.ndarray], gates: Mapping[str, np.ndarray], sources: Mapping[str, np.ndarray],
               w: str | float = "full", n_glyph: int = 4):
    """Per scan range: background = fraction of recoverable edge directions per pixel (inside the circular field of
    view); bow-tie glyphs = the recoverable EDGE DIRECTIONS (edges tangent to at least one measured ray); purple arc =
    source positions (true angles, drawn at a reduced radius).

    maps: {geom: (n, n)}; gates: {geom: (J, n, n)} (frequency directions, physical coordinates, y up);
    sources: {geom: (V, 2) source positions in pixels (column, row)}.
    """
    geoms = list(maps)
    W = width(w)
    k = len(geoms)
    fig = plt.figure(figsize=(W, W / k * 0.98 + 0.5))
    gs = fig.add_gridspec(2, k, height_ratios=[1, 0.04], hspace=0.16, wspace=0.02, left=0.005, right=0.995,
                          bottom=0.13, top=0.95)
    im = None
    for gi, g in enumerate(geoms):
        ax = fig.add_subplot(gs[0, gi])
        F = np.asarray(maps[g], float)
        n = F.shape[0]
        cc = (n - 1) / 2
        yy, xx = np.mgrid[0:n, 0:n]
        fov = (yy - cc) ** 2 + (xx - cc) ** 2 <= (n / 2) ** 2
        im = ax.imshow(np.where(fov, F, np.nan), cmap=CMAP_SEQ, vmin=0, vmax=1, interpolation="bilinear",
                       extent=(-0.5, n - 0.5, n - 0.5, -0.5))
        ax.add_patch(Circle((cc, cc), n / 2, fill=False, ec=INK_2, lw=0.5))
        G = np.asarray(gates[g])
        J = G.shape[0]
        half_bin = 90.0 / J
        step = n / (n_glyph + 1)
        R = step * 0.4
        for a in range(1, n_glyph + 1):
            for b in range(1, n_glyph + 1):
                r, c = a * step, b * step
                if (r - cc) ** 2 + (c - cc) ** 2 > (n / 2 - R) ** 2:
                    continue
                ax.add_patch(Circle((c, r), R, fc=WHITE, ec=INK_2, lw=0.4, alpha=0.9, zorder=3))
                for j in range(J):
                    if G[j, int(r), int(c)] <= 0.5:
                        continue
                    edge = (j + 0.5) * 180.0 / J - 90.0          # edge direction = frequency direction - 90° (y up)
                    for base in (edge, edge + 180.0):            # the data y axis points down => flip the angle sign
                        ax.add_patch(Wedge((c, r), R, -base - half_bin, -base + half_bin, fc=INK, ec=INK,
                                           lw=0.2, zorder=4))
        src = np.asarray(sources[g], float)
        ang = np.unwrap(np.arctan2(-(src[:, 1] - cc), src[:, 0] - cc))
        Rs = n * 0.575
        ax.plot(cc + Rs * np.cos(ang), cc - Rs * np.sin(ang), color=VIOLET, lw=2.2, solid_capstyle="round",
                zorder=5, clip_on=False)
        mid = float(np.median(ang))
        lr = Rs + n * 0.065
        ax.text(cc + lr * np.cos(mid), cc - lr * np.sin(mid), t("source"), color=VIOLET,
                fontsize=CTX.preset.small - 0.5, ha="center", va="center",
                rotation=np.degrees(mid) + 90 if np.sin(mid) < 0 else np.degrees(mid) - 90)
        lim = 0.68 * n
        ax.set_xlim(cc - lim, cc + lim)
        ax.set_ylim(cc + lim, cc - lim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0.5, 1.0, geom_label(g), transform=ax.transAxes, ha="center", va="top",
                fontsize=CTX.preset.font, fontweight="bold")
        panel_label(ax, gi, x=0.08, y=0.93)
    cax = fig.add_subplot(gs[1, :])
    pos = cax.get_position()
    cax.set_position((pos.x0 + pos.width * 0.5, pos.y0, pos.width * 0.36, pos.height))
    assert im is not None, "at least one geometry is required"
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=CTX.preset.small - 0.5, width=0.4, length=2, pad=1.5)
    cb.set_label(t("vis_frac"), fontsize=CTX.preset.small, labelpad=2)
    # legend for the bow-tie glyphs
    lx = pos.x0 + pos.width * 0.08
    gax = fig.add_axes((lx, pos.y0 - 0.035, 0.05, 0.11))
    gax.set_xlim(-1.1, 1.1)
    gax.set_ylim(-1.1, 1.1)
    gax.set_aspect("equal")
    gax.axis("off")
    gax.add_patch(Circle((0, 0), 1, fc=WHITE, ec=INK_2, lw=0.4))
    for base in (70.0, 250.0):
        gax.add_patch(Wedge((0, 0), 1, base - 30, base + 30, fc=INK, ec=INK, lw=0.2))
    fig.text(lx + 0.062, pos.y0 + pos.height / 2, t("glyph"), fontsize=CTX.preset.small, va="center", ha="left")
    return fig
