"""Training figures: validation curves, learning-rate sweeps, per-stage parameters."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..style import (
    CTX,
    HAIR,
    INK,
    INK_2,
    WHITE,
    color,
    figure,
    key_of,
    label,
    marker,
    panel_title,
    t,
    tint,
    width,
    ygrid,
)
from .layout import MANY, SHORT, _end_labels, _fit_font, _legend_cols, _ms


# ============================================================================ 5. learning curves
def _xy(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    d = df.dropna(subset=[metric])
    e = d["epoch"].to_numpy(float)
    return (e + 1 if e.size and e.min() == 0 else e), d[metric].to_numpy(float)


def curves(runs: Mapping[str, Sequence[pd.DataFrame]], metric: str = "val_psnr", w: str | float = "col",
           h: float = 2.1, ylim: tuple[float, float] | None = None):
    """Per method: seed-mean line + min–max band across seeds; direct labels at the end of each line."""
    n = len(runs)
    if n > MANY:                                   # room for one label per line
        w, h = ("full" if w == "col" else w), max(h, 0.5 + 0.16 * n)
    fig = figure(w, h)
    ax = fig.add_axes((0.14, 0.18, 0.62, 0.78) if n <= MANY else (0.1, 0.13, 0.56, 0.84))
    ends = []
    for m, dfs in runs.items():
        xs = [_xy(d, metric) for d in dfs if metric in d]
        if not xs:
            continue
        L = min(len(x) for x, _ in xs)
        X = xs[0][0][:L]
        Y = np.stack([y[:L] for _, y in xs])
        c = color(m)
        if len(Y) > 1:
            ax.fill_between(X, Y.min(0), Y.max(0), color=tint(c, 0.75), lw=0, zorder=1)
        ax.plot(X, Y.mean(0), color=c, lw=1.2, zorder=2)
        ends.append((m, X[-1], Y.mean(0)[-1]))
    ax.set_xlabel(t("epoch"))
    ax.set_ylabel(t(metric))
    if ylim:
        ax.set_ylim(*ylim)
    ygrid(ax)
    if ends:
        lo, hi = ax.get_ylim()
        xmax = max(e[1] for e in ends)
        ys = _end_labels([e[2] for e in ends], lo, hi)
        for (m, xe, ye), yl in zip(ends, ys):
            ax.annotate(label(m), (xe, ye), xytext=(xmax * 1.03, yl), va="center", ha="left",
                        fontsize=CTX.preset.small, color=INK, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=HAIR, lw=0.5, shrinkA=0, shrinkB=2))
    return fig


def sci(v: float) -> str:
    r"""3e-05 -> $3\times10^{-5}$ (mathtext)."""
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$" if int(e) != 0 else m


def lr_sweep(runs: Mapping[str, Mapping[float, pd.DataFrame]], metric: str = "val_psnr", best: Mapping | None = None,
             w: str | float = "full", h: float = 1.75):
    """Small panels per architecture; one line style per lr (dotted, dashed, solid = increasing lr); chosen lr bold."""
    archs = list(runs)
    ncol = len(archs) if len(archs) <= 6 else 5           # many architectures: 5-column grid, several rows
    nrow = math.ceil(len(archs) / ncol)
    gap = 0.42                                             # inches (before font scaling) between two panel rows
    h_all = h * (0.43 + 0.57 * nrow) + (nrow - 1) * gap
    fig = figure(w, h_all)
    gs = fig.add_gridspec(nrow, ncol, wspace=0.12, hspace=gap / (0.57 * h), left=0.075, right=0.995,
                          bottom=0.33 * h / h_all, top=1 - 0.1 * h / h_all)
    styles = [(0, (1, 1.6)), (0, (3.5, 1.6)), "-", (0, (6, 1.5, 1, 1.5))]
    lrs_all = sorted({lr for d in runs.values() for lr in d})
    names = [label(a) if nrow == 1 else SHORT.get(key_of(a), label(a)) for a in archs]
    tfs = CTX.preset.font if nrow == 1 else _fit_font(names, 0.92 * width(w) / (ncol * 1.12), CTX.preset.small)
    ax0 = None
    for ai, a in enumerate(archs):
        ax = fig.add_subplot(gs[ai // ncol, ai % ncol], sharey=ax0)
        ax0 = ax0 or ax
        c = color(a)
        for lr, d in sorted(runs[a].items()):
            x, y = _xy(d, metric)
            is_best = best is not None and np.isclose(best.get(a, -1), lr)
            ax.plot(x, y, color=c, lw=1.5 if is_best else 0.9, ls=styles[lrs_all.index(lr) % len(styles)],
                    alpha=1.0 if is_best else 0.85, zorder=3 if is_best else 2)
        if nrow == 1:
            panel_title(ax, ai, names[ai])
        else:
            ax.set_title(names[ai], loc="left", fontsize=tfs, pad=3.0)
        if ai + ncol >= len(archs):                        # last row of each column
            ax.set_xlabel(t("epoch"))
        ygrid(ax)
        if ai % ncol == 0:
            ax.set_ylabel(t(metric))
        else:
            ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=INK_2, lw=1.0, ls=styles[i % len(styles)], label=f"lr = {sci(lr)}")
               for i, lr in enumerate(lrs_all)]
    handles.append(Line2D([], [], color=INK_2, lw=1.8, label="selected" if CTX.lang == "en" else "được chọn"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.53, -0.02),
               frameon=False, handlelength=2.6)
    return fig


# ============================================================================ 8. learned per-stage parameters
PARAM_TEX = {"alphas": r"$\alpha_k$", "betas": r"$\beta_k$"}


def stage_params(params: Mapping[str, Mapping[str, Sequence[float]]], names: Sequence[str] = ("alphas", "betas"),
                 w: str | float = "full", h: float = 1.55):
    names = [n for n in names if any(n in p for p in params.values())]
    fig = figure(w, h)
    gs = fig.add_gridspec(1, len(names), wspace=0.32, left=0.05, right=0.995, bottom=0.25, top=0.78)
    for i, nm in enumerate(names):
        ax = fig.add_subplot(gs[0, i])
        for m, p in params.items():
            if nm not in p:
                continue
            v = np.asarray(p[nm], float)
            k = np.arange(1, len(v) + 1)
            ax.plot(k, v, color=color(m), lw=1.1, marker=marker(m), ms=_ms(m, 3.2), mfc=color(m), mec=WHITE,
                    mew=0.5, label=label(m))
        panel_title(ax, i, PARAM_TEX.get(nm, nm))
        ax.set_xlabel(t("stage"))
        ax.set_xticks([1, 5, 10, 14] if len(v) >= 14 else k)
        ygrid(ax)
    handles, labels_ = fig.axes[0].get_legend_handles_labels()
    if len(labels_) > 6:                            # multi-row legend: below the figure, columns fit the width
        fig.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.52, 0.03), frameon=False,
                   ncol=_legend_cols(labels_, fig.get_figwidth(), CTX.preset.small), columnspacing=1.0)
    else:
        fig.legend(handles, labels_, loc="upper center", ncol=len(labels_), bbox_to_anchor=(0.52, 1.02),
                   frameon=False)
    return fig
