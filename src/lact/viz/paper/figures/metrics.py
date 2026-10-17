"""Metric figures: per-slice strips, paired differences, trends over the scan range, error split, Pareto fronts."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator

from .. import stats as S
from ..style import (
    CTX,
    GRID,
    HAIR,
    INK,
    INK_2,
    WHITE,
    color,
    figure,
    geom_angle,
    geom_label,
    key_of,
    label,
    marker,
    panel_title,
    t,
    tint,
    ygrid,
)
from .layout import MANY, SHORT, _end_labels, _fit_ticklabels, _legend_cols, _method_axis, _ms, _text_w


# ============================================================================ 2. metric per method
def metric_strip(df: pd.DataFrame, metric: str, geoms: Sequence[str], methods: Sequence[str],
                 w: str | float = "full", h: float = 1.9, show_slices: bool = True, ylabel: str | None = None):
    """One panel per scan range: faint dots = slices (seed means), filled marker = mean, bar = 95% CI.
    With >1 seed, small hollow markers show each seed's mean. With many methods, each scan range gets a full-width row
    (tick labels only on the last row when all rows share the same methods)."""
    present = [[m for m in methods if m in set(df[df.geometry == g].method)] for g in geoms]
    stacked = len(geoms) > 1 and max(len(p_) for p_ in present) > MANY
    same = all(p_ == present[0] for p_ in present)
    if stacked:
        fig = figure(w, h * len(geoms))
        gs = fig.add_gridspec(len(geoms), 1, hspace=0.28 if same else 1.0, left=0.085, right=0.995, bottom=0.12,
                              top=0.95)
    else:
        fig = figure(w, h)
        gs = fig.add_gridspec(1, len(geoms), wspace=0.22, left=0.07, right=0.995, bottom=0.25, top=0.9,
                              width_ratios=[max(2.6, len(p_) + 0.8) for p_ in present])
    rng = np.random.default_rng(0)
    for gi, g in enumerate(geoms):
        ax = fig.add_subplot(gs[gi, 0] if stacked else gs[0, gi])
        hidden = stacked and same and gi < len(geoms) - 1
        dg = df[df.geometry == g]
        ms = present[gi]
        ps = S.per_slice(dg, metric)
        for i, m in enumerate(ms):
            v = ps[ps.method == m][metric].to_numpy()
            c = color(m)
            if show_slices and v.size:
                ax.scatter(i + rng.uniform(-0.2, 0.2, v.size), v, s=2.2, color=tint(c, 0.55), lw=0, zorder=1,
                           rasterized=True)
            lo, hi = S.bootstrap_ci(v)
            ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot(i, v.mean(), marker=marker(m), ms=_ms(m, 4.6), mfc=c, mec=WHITE, mew=0.6, zorder=4, ls="none")
            if "seed" in dg and dg[dg.method == m].seed.nunique() > 1:
                sm = dg[dg.method == m].groupby("seed")[metric].mean()
                ax.plot(np.full(len(sm), i + 0.32), sm.values, marker=marker(m), ms=2.6, mfc=WHITE, mec=c, mew=0.6,
                        ls="none", zorder=4)
        _method_axis(ax, ms, bracket=not hidden)
        if hidden:
            ax.tick_params(labelbottom=False)
        ygrid(ax)
        panel_title(ax, gi if len(geoms) > 1 else None, geom_label(g))
        if gi == 0 or stacked:
            ax.set_ylabel(ylabel or t(metric))
    _fit_ticklabels(fig)
    return fig


# ============================================================================ 3. paired Δ (forest plot)
def paired_forest(pairs: pd.DataFrame, metric: str, geoms: Sequence[str], methods: Sequence[str], ref: str,
                  w: str | float = "full", h: float | None = None):
    """Per-slice Δ metric versus ``ref``: faint dots = slices, marker = mean, bar = 95% CI; right column: value,
    share of slices that improve, significance (paired Wilcoxon, Holm)."""
    ms = [m for m in methods if m != ref]
    h = h or 0.32 + 0.24 * len(ms)
    fig = figure(w, h)
    W = fig.get_figwidth()
    left_in = 0.12 + 0.0078 * CTX.preset.small * max(len(label(m)) for m in ms)   # ~ width of the longest label
    gs = fig.add_gridspec(1, len(geoms), wspace=0.6, left=left_in / W, right=1 - 0.62 / W, bottom=0.2, top=0.88)
    rng = np.random.default_rng(1)
    for gi, g in enumerate(geoms):
        ax = fig.add_subplot(gs[0, gi])
        pg = pairs[pairs.geometry == g].set_index("method")
        for yi, m in enumerate(ms):
            if m not in pg.index:
                continue
            r = pg.loc[m]
            c = color(m)
            d = np.asarray(r["deltas"], float)
            ax.scatter(d, yi + rng.uniform(-0.18, 0.18, d.size), s=2.0, color=tint(c, 0.6), lw=0, zorder=1,
                       rasterized=True)
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=INK, lw=1.0, zorder=3, solid_capstyle="butt")
            ax.plot(r.delta_mean, yi, marker=marker(m), ms=_ms(m, 4.6), mfc=c, mec=WHITE, mew=0.6, zorder=4)
            ax.text(1.02, yi, f"{r.delta_mean:+.2f}  {r.stars}", transform=transforms.blended_transform_factory(
                ax.transAxes, ax.transData), va="center", ha="left", fontsize=CTX.preset.small - 0.5, color=INK)
        ax.axvline(0, color=INK_2, lw=0.6, zorder=0)
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([label(m) for m in ms] if gi == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(ms) - 0.5, -0.5)
        ax.grid(True, axis="x", color=GRID, lw=0.5)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(t("delta", m=t(metric).split(" (")[0], ref=label(ref)))
        panel_title(ax, gi if len(geoms) > 1 else None, geom_label(g))
    return fig


# ============================================================================ 4. metric versus scan range
def angle_lines(summary: pd.DataFrame, methods: Sequence[str], metric_label: str, w: str | float = "col",
                h: float = 2.2, train_angles: Sequence[float] | None = None):
    """summary: geometry, method, mean, ci_lo, ci_hi. Lines + CI; direct labels at the right end (no legend)."""
    n = sum(m in set(summary.method) for m in methods)
    if n > MANY:                                   # room for one label per line
        w, h = ("full" if w == "col" else w), max(h, 0.5 + 0.16 * n)
    fig = figure(w, h)
    ax = fig.add_axes((0.15, 0.17, 0.6, 0.78) if n <= MANY else (0.1, 0.13, 0.56, 0.84))
    ends = []
    for m in methods:
        s = summary[summary.method == m].copy()
        if s.empty:
            continue
        s["a"] = s.geometry.map(geom_angle)
        s = s.sort_values("a")
        c = color(m)
        ax.plot(s.a, s["mean"], color=c, lw=1.2, zorder=2)
        ax.vlines(s.a, s.ci_lo, s.ci_hi, color=c, lw=1.0, zorder=2)
        ax.plot(s.a, s["mean"], ls="none", marker=marker(m), ms=_ms(m, 4.2), mfc=c, mec=WHITE, mew=0.6, zorder=3)
        ends.append((m, float(s.a.iloc[-1]), float(s["mean"].iloc[-1])))
    angles = sorted({geom_angle(g) for g in summary.geometry})
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{a:g}°" for a in angles])
    if train_angles is not None:
        for a in angles:
            if a not in train_angles:
                ax.axvspan(a - 4, a + 4, color=GRID, lw=0, zorder=0)
    ax.set_xlim(angles[0] - 6, angles[-1] + 6)
    ax.set_xlabel(t("angle"))
    ax.set_ylabel(metric_label)
    ygrid(ax)
    if ends:
        lo, hi = ax.get_ylim()
        ys = _end_labels([e[2] for e in ends], lo, hi)
        for (m, xa, ya), yl in zip(ends, ys):
            ax.annotate(label(m), (xa, ya), xytext=(angles[-1] + 9, yl), textcoords="data", va="center",
                        ha="left", fontsize=CTX.preset.small, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.5, shrinkA=0, shrinkB=3))
    return fig


# ============================================================================ 7. visible / invisible error split
def error_split(df: pd.DataFrame, methods: Sequence[str], geometry: str, n_pix: int, w: str | float = "col",
                h: float | None = None):
    """100% bars: share of the MSE in the INVISIBLE part (dark) and the visible part (light); right column: total MSE.
    The two parts add up exactly to the MSE because P_vis + P_inv = I and we use <e, P e> (Parseval frame)."""
    d = df[df.geometry == geometry]
    ms = [m for m in methods if m in set(d.method)]
    h = h or 0.75 + 0.19 * len(ms)
    if len(ms) > MANY:                              # margins fixed in inches: they do not grow with the row count
        fig = figure("full" if w == "col" else w, h)
        W, H = fig.get_size_inches()
        left = 0.12 + max(_text_w(label(m), CTX.preset.small) for m in ms)
        ax = fig.add_axes((left / W, 0.78 / H, min(3.2, W - left - 0.75) / W, 1 - 1.0 / H))
    else:
        fig = figure(w, h)
        ax = fig.add_axes((0.33, 0.36 if len(ms) > 3 else 0.42, 0.45, 0.58))
    dark, light = INK_2, "#c4c4c4"
    tr = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, m in enumerate(ms):
        r = d[d.method == m]
        inv, vis = r.err_inv.mean() / n_pix, r.err_vis.mean() / n_pix
        tot = inv + vis
        pi = 100 * inv / tot
        ax.barh(yi, pi, height=0.64, color=dark, edgecolor=WHITE, lw=1.0, zorder=2)
        ax.barh(yi, 100 - pi, left=pi, height=0.64, color=light, edgecolor=WHITE, lw=1.0, zorder=2)
        ax.text(pi - 2 if pi > 18 else pi + 2, yi, f"{pi:.0f}%", va="center", ha="right" if pi > 18 else "left",
                fontsize=CTX.preset.small - 0.5, color=WHITE if pi > 18 else INK)
        ax.text(1.03, yi, f"{tot * 1e3:.2f}", transform=tr, va="center", ha="left",
                fontsize=CTX.preset.small - 0.5, color=INK)
    ax.text(1.03, -0.75, "MSE" + r" (×10$^{-3}$)", transform=tr, va="bottom", ha="left",
            fontsize=CTX.preset.small - 0.5, color=INK_2)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([label(m) for m in ms])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(ms) - 0.5, -0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(t("err_share"))
    box = ax.get_position()
    fig.legend(handles=[Patch(color=dark, label=t("invisible")), Patch(color=light, label=t("visible"))],
               loc="lower center", bbox_to_anchor=(0.55 if len(ms) <= MANY else (box.x0 + box.x1) / 2, 0.0), ncol=2,
               frameon=False, handlelength=1.0, borderaxespad=0.1)
    return fig


# ============================================================================ 9. quality–cost Pareto
def pareto(df: pd.DataFrame, x: str, xlabel: str, ylabel: str, logx: bool = False, w: str | float = "col",
           h: float = 2.2):
    """df: method, <x>, mean, ci_lo, ci_hi. Grey step line = Pareto front (cheaper and better).
    Few points get labels next to them; many points use a legend below the figure (direct labels would overlap)."""
    many = len(df) > MANY
    fig = figure("full" if many and w == "col" else w, h * (1.25 if many else 1.0))
    ax = fig.add_axes((0.15, 0.18, 0.8, 0.77) if not many else (0.1, 0.16, 0.88, 0.8))
    d = df.sort_values(x)
    front, best = [], -np.inf
    for _, r in d.iterrows():
        if r["mean"] > best:
            best = r["mean"]
            front.append((r[x], r["mean"]))
    if len(front) > 1:
        fx, fy = zip(*front)
        ax.step(fx, fy, where="post", color=HAIR, lw=1.0, zorder=1)
    for _, r in d.iterrows():
        m = r.method
        ax.vlines(r[x], r.ci_lo, r.ci_hi, color=color(m), lw=1.0, zorder=2)
        ax.plot(r[x], r["mean"], marker=marker(m), ms=_ms(m, 5.0), mfc=color(m), mec=WHITE, mew=0.6, zorder=3)
        if not many:
            ax.annotate(label(m), (r[x], r["mean"]), xytext=(5, 3), textcoords="offset points",
                        fontsize=CTX.preset.small - 0.5, color=INK)
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    lo, hi = ax.get_xlim()
    if not many:                                    # keep room on the right for direct labels
        ax.set_xlim(lo, hi * (2.2 if logx else 1.0) if logx else hi + (hi - lo) * 0.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ygrid(ax)
    if many:
        order = list(SHORT)                         # legend order = architecture family order
        ms = sorted(df.method, key=lambda m: order.index(key_of(m)) if key_of(m) in order else -1)
        hs = [Line2D([], [], ls="none", marker=marker(m), ms=_ms(m, 5.0), mfc=color(m), mec=WHITE, mew=0.6,
                     label=label(m)) for m in ms]
        fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.54, 0.02), frameon=False, columnspacing=1.0,
                   handletextpad=0.3, ncol=_legend_cols([label(m) for m in ms], fig.get_figwidth(),
                                                        CTX.preset.small))
    return fig
