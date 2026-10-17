"""Quantitative thesis plots (matplotlib, PDF/PNG output).

Rules: one y axis, colours fixed per architecture, points + error bars (standard deviation over seeds), a legend
whenever there are >= 2 series plus selective direct labels, thin grid lines, neutral colours for baselines.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import ARCH_MARKERS, ARCH_ORDER, CMAP_SEQ, INK, INK_2, MUTED, apply_style, color_of, geom_label, label_of


def _save(fig, out_path):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def _spread(values: list[float], gap: float) -> list[float]:
    """Shift labels along y so consecutive labels are at least ``gap`` apart (order preserved)."""
    order = np.argsort(values)
    out = np.array(values, float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out.tolist()


def _order(keys):
    return sorted(keys, key=lambda k: ARCH_ORDER.index(k) if k in ARCH_ORDER else 99)


MANY = 6            # above this many learned models: widen the figure, rotate tick labels, drop overlapping labels


def _marker(reg: str, size: float) -> tuple[str, float]:
    """(marker, size). Stars (RegNet variants) look smaller at equal size, so they are enlarged for visual balance."""
    m = ARCH_MARKERS.get(reg, "o")
    return m, size * (1.45 if m == "*" else 1.0)


def metric_by_arch(method_df: pd.DataFrame, metric: str = "psnr", geometries: list[str] | None = None,
                   ylabel: str = "PSNR (dB)", out_path=None, title: str | None = None):
    """Figure 1: one panel per geometry (shared y axis); point = mean over seeds, bar = ±1 standard deviation.
    FBP/TV are drawn as neutral horizontal lines with direct labels."""
    apply_style()
    geometries = geometries or sorted(method_df.geometry_test.unique())
    n_learned = method_df[~method_df.regularizer.isin(["fbp", "tv"])].regularizer.nunique()
    many = n_learned > MANY
    fig, axes = plt.subplots(1, len(geometries), sharey=True, squeeze=False,
                             figsize=(max(3.2, 0.42 * n_learned + 1.0) * len(geometries), 3.6 if many else 2.9))
    for ax, g in zip(axes[0], geometries):
        d = method_df[method_df.geometry_test == g]
        learned = _order([r for r in d.regularizer.unique() if r not in ("fbp", "tv")])
        for i, r in enumerate(learned):
            row = d[d.regularizer == r].iloc[0]
            m, s = row[f"{metric}_mean"], row.get(f"{metric}_std", np.nan)
            mk, ms = _marker(r, 7)
            ax.errorbar(i, m, yerr=None if not np.isfinite(s) else s, fmt=mk, ms=ms,
                        color=color_of(r), ecolor=color_of(r), elinewidth=1.2, capsize=0, zorder=3)
            if many:        # numeric labels stand vertically above the points so they never cover a neighbour
                ax.annotate(f"{m:.2f}", (i, m), xytext=(0, 8), textcoords="offset points", ha="center",
                            va="bottom", rotation=90, fontsize=6.5, color=INK_2)
            else:
                ax.annotate(f"{m:.2f}", (i, m), xytext=(8, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK_2)
        for b in ("fbp", "tv"):
            rb = d[d.regularizer == b]
            if len(rb):
                v = rb.iloc[0][f"{metric}_mean"]
                ax.axhline(v, color=color_of(b), lw=1.0, zorder=1)
                ax.annotate(f"{label_of(b)} {v:.2f}", (len(learned) - 0.5, v), xytext=(0, 3),
                            textcoords="offset points", ha="right", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(learned)))
        tick_kw = {"rotation": 40, "ha": "right", "rotation_mode": "anchor", "fontsize": 7.5} if many else {}
        ax.set_xticklabels([label_of(r).replace("LEARN–", "") for r in learned], **tick_kw)
        ax.set_xlim(-0.6, len(learned) - 0.4)
        if many:
            ax.margins(y=0.18)
        ax.set_title(geom_label(g), loc="left")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel(ylabel)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    return _save(fig, out_path)


def interaction_plot(method_df: pd.DataFrame, metric: str = "psnr", order_geoms: list[str] | None = None,
                     ylabel: str = "PSNR (dB)", out_path=None):
    """Figure 2 (RQ2): quality versus scan range, one line per architecture, direct labels at the right end."""
    apply_style()
    order_geoms = order_geoms or sorted(method_df.geometry_test.unique(),
                                        key=lambda g: -float(geom_label(g).rstrip("°") or 0))
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    xs = np.arange(len(order_geoms))
    regs = _order(method_df.regularizer.unique())
    ends = []
    for r in regs:
        d = method_df[method_df.regularizer == r].set_index("geometry_test").reindex(order_geoms)
        m = d[f"{metric}_mean"].to_numpy(float)
        s = d.get(f"{metric}_std", pd.Series(np.nan, index=d.index)).to_numpy(float)
        base = r in ("fbp", "tv")
        mk, ms = _marker(r, 6)
        ax.plot(xs, m, color=color_of(r), lw=1.0 if base else 1.6, marker=mk, ms=ms,
                label=label_of(r), zorder=2 if base else 3)
        if not base and np.isfinite(s).any():
            ax.errorbar(xs, m, yerr=s, fmt="none", ecolor=color_of(r), elinewidth=1.0, capsize=0)
        if np.isfinite(m[-1]):
            ends.append((m[-1], label_of(r).replace("LEARN–", "")))
    if ends:
        lo, hi = ax.get_ylim()
        ys = _spread([e[0] for e in ends], 0.045 * (hi - lo))
        for (_, text), yv in zip(ends, ys):
            ax.text(xs[-1] + 0.08, yv, text, va="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(xs)
    ax.set_xticklabels([geom_label(g) for g in order_geoms])
    ax.set_xlim(-0.2, len(xs) - 0.4)
    ax.set_xlabel("Khoảng quét")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    return _save(fig, out_path)


def pareto_plot(df: pd.DataFrame, x: str, y: str = "psnr_mean", xlabel: str = "", ylabel: str = "PSNR (dB)",
                logx: bool = False, out_path=None):
    """Figure 3: quality versus cost. df has columns regularizer, x, y (and *_std when available).
    Marker shape is a secondary encoding; few points get direct labels, many points only a legend."""
    apply_style()
    many = df.regularizer.nunique() > MANY
    fig, ax = plt.subplots(figsize=(5.4, 3.6) if many else (4.2, 3.0))
    if many:                                    # legend ordered by architecture family
        rank = {reg: i for i, reg in enumerate(_order(df.regularizer.unique()))}
        df = df.sort_values("regularizer", key=lambda c: c.map(rank))
    for _, r in df.iterrows():
        reg = r["regularizer"]
        mk, ms = _marker(reg, 8)
        ax.scatter(r[x], r[y], s=ms ** 2, marker=mk, color=color_of(reg),
                   edgecolor="#fcfcfb", linewidth=1.0, zorder=3, label=label_of(reg))
        ys = r.get(y.replace("_mean", "_std"), np.nan)
        if np.isfinite(ys):
            ax.errorbar(r[x], r[y], yerr=ys, fmt="none", ecolor=color_of(reg), elinewidth=1.0, zorder=2)
        if not many:
            ax.annotate(label_of(reg).replace("LEARN–", ""), (r[x], r[y]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    h, lab = ax.get_legend_handles_labels()
    uniq = dict(zip(lab, h))
    ax.legend(uniq.values(), uniq.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    return _save(fig, out_path)


def per_slice_plot(df: pd.DataFrame, metric: str = "psnr", ylabel: str = "PSNR (dB)", out_path=None):
    """Figure 4: metric by slice position (z) per patient (small multiples), averaged over seeds."""
    apply_style()
    pids = sorted(df.pid.unique())
    fig, axes = plt.subplots(len(pids), 1, figsize=(6.4, 2.2 * len(pids)), sharey=True, squeeze=False)
    for ax, pid in zip(axes[:, 0], pids):
        d = df[df.pid == pid]
        for r in _order(d.regularizer.unique()):
            g = d[d.regularizer == r].groupby("idx")[metric].mean()
            ax.plot(g.index, g.values, color=color_of(r), lw=1.0 if r in ("fbp", "tv") else 1.3,
                    label=label_of(r))
        ax.set_title(pid, loc="left")
        ax.set_ylabel(ylabel)
    axes[-1, 0].set_xlabel("Chỉ số lát (theo trục z)")
    axes[-1, 0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3)
    fig.subplots_adjust(hspace=0.45)
    return _save(fig, out_path)


def training_curves(curves: dict[str, list[pd.DataFrame]], metric: str = "val_psnr", ylabel: str = "PSNR xác thực (dB)",
                    out_path=None, title: str | None = None):
    """Figure 10: {regularizer: [metrics.csv per seed]} -> mean and min–max band over seeds."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for r in _order(curves.keys()):
        frames = [f[["epoch", metric]].dropna() for f in curves[r] if metric in f]
        if not frames:
            continue
        m = pd.concat(frames).groupby("epoch")[metric]
        mean, lo, hi = m.mean(), m.min(), m.max()
        ax.plot(mean.index, mean.values, color=color_of(r), lw=1.5, label=label_of(r),
                marker=_marker(r, 5)[0] if len(mean) < 8 else None, ms=_marker(r, 5)[1])
        if len(frames) > 1:
            ax.fill_between(mean.index, lo.values, hi.values, color=color_of(r), alpha=0.15, lw=0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return _save(fig, out_path)


def stagewise_plot(summaries: dict[str, pd.DataFrame], out_path=None):
    """Figure 8: PSNR of x_k per stage (left panel) and alpha_k, beta_k (two right panels), one axis each."""
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.9))
    for r in _order(summaries.keys()):
        s = summaries[r]
        c = color_of(r)
        mk, ms = _marker(r, 4)
        axes[0].plot(s.k, s.psnr_mean, color=c, marker=mk, ms=ms, lw=1.4, label=label_of(r))
        t = s[s.k > 0]
        if "alpha_mean" in t:
            axes[1].plot(t.k, t.alpha_mean, color=c, lw=1.4, marker=mk, ms=ms)
            axes[2].plot(t.k, t.beta_mean, color=c, lw=1.4, marker=mk, ms=ms)
    axes[0].set_ylabel("PSNR của x_k (dB)")
    axes[1].set_ylabel("alpha_k")
    axes[2].set_ylabel("beta_k")
    for ax in axes:
        ax.set_xlabel("Giai đoạn k")
        ax.ticklabel_format(axis="y", useOffset=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.22), ncol=4)
    fig.subplots_adjust(wspace=0.35)
    return _save(fig, out_path)


def wedge_fraction_plot(df: pd.DataFrame, regions=("measured", "partial", "missing"), out_path=None):
    """Figure 7: share of error energy per frequency region; one panel per region, one bar per architecture."""
    apply_style()
    names = {"measured": "Đo đủ", "partial": "Đo một phần", "missing": "Thiếu"}
    fig, axes = plt.subplots(1, len(regions), figsize=(3.0 * len(regions), 2.8), sharey=True, squeeze=False)
    regs = _order(df.regularizer.unique())
    for ax, reg_name in zip(axes[0], regions):
        col = f"frac_{reg_name}"
        vals = [df[df.regularizer == r][col].mean() for r in regs]
        ax.barh(range(len(regs)), vals, height=0.6, color=[color_of(r) for r in regs], edgecolor="#fcfcfb",
                linewidth=2)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.2f}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=7.5,
                        color=INK_2)
        ax.set_title(names.get(reg_name, reg_name), loc="left")
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Tỉ lệ năng lượng sai số")
    axes[0, 0].set_yticks(range(len(regs)))
    axes[0, 0].set_yticklabels([label_of(r) for r in regs])
    axes[0, 0].invert_yaxis()
    return _save(fig, out_path)


def heatmaps(maps: dict[str, np.ndarray], titles: dict[str, str] | None = None, cmap=CMAP_SEQ, vmax=None,
             radius: dict[str, float] | None = None, crop: int | None = None, cbar_label: str = "",
             out_path=None):
    """Figure 9 (ERF) and the frequency-coverage map: a grid of single-hue heat maps on a common scale."""
    apply_style()
    keys = list(maps.keys())
    nrow = math.ceil(len(keys) / 8)                       # at most 8 panels per row
    ncol = math.ceil(len(keys) / nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.5 * nrow), squeeze=False)
    vmax = vmax if vmax is not None else max(float(np.max(m)) for m in maps.values())
    im = None
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax, k in zip(axes.ravel(), keys):
        m = maps[k]
        if crop:
            c = m.shape[0] // 2
            m = m[c - crop: c + crop, c - crop: c + crop]
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title((titles or {}).get(k, label_of(k)), fontsize=8.5)
        if radius and k in radius:
            c = m.shape[0] / 2 - 0.5
            ax.add_patch(plt.Circle((c, c), radius[k], fill=False, ec=INK, lw=0.8))
            ax.annotate(f"r90 = {radius[k]:.0f} px", (0.02, 0.02), xycoords="axes fraction", fontsize=7, color=INK)
    assert im is not None
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label(cbar_label, color=INK_2)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    return _save(fig, out_path)


def alphas_from_metrics(metrics: pd.DataFrame) -> tuple[list[float], list[float]]:
    """alpha_k, beta_k at the last epoch from a metrics.csv table (already read into a DataFrame)."""
    last = metrics.dropna(subset=["alphas"]).iloc[-1]
    return json.loads(last["alphas"]), json.loads(last["betas"])


def operator_check_figure(phantom: np.ndarray, panels: list[tuple], out_path=None):
    """Figure E0: phantom + (FBP, sinogram) for each geometry.

    ``panels``: list of (name, n_views, n_det, sinogram, FBP image, FBP PSNR).
    """
    apply_style()
    fig, axes = plt.subplots(2, len(panels) + 1, figsize=(2.3 * (len(panels) + 1), 4.8), squeeze=False)
    axes[0, 0].imshow(phantom, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Phantom")
    axes[1, 0].axis("off")
    for j, (name, n_views, n_det, sino, rec, p) in enumerate(panels, start=1):
        axes[0, j].imshow(np.clip(rec, 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(f"{name}\n{p:.1f} dB", fontsize=8)
        axes[1, j].imshow(sino, cmap="gray", aspect="auto")
        axes[1, j].set_title(f"sinogram {n_views}x{n_det}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    return _save(fig, out_path)
