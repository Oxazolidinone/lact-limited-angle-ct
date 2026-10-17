"""Layout helpers shared by the paper figures: method short names, font fitting, legend columns, labels at
line ends, method axes."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath

from ...style import ARCH_SHORT
from ..style import (
    CTX,
    INK_2,
    key_of,
    label,
    marker,
    spread,
)

SHORT = {r: s.replace("–", "-") for r, s in ARCH_SHORT.items()}


LEARN_GROUP = tuple(SHORT)


MANY = 9            # more methods than this: change the layout (wrap rows, legend instead of direct labels)


def _ms(m: str, base: float) -> float:
    """Stars (RegNet variants) look smaller at equal size: enlarged for visual balance."""
    return base * 1.45 if marker(m) == "*" else base


def _text_w(s: str, fs: float) -> float:
    """Width (inches) of the longest text line at ``fs`` pt, measured with the active font."""
    return max(TextPath((0, 0), line, size=fs).get_extents().width for line in s.split("\n")) / 72.0


def _fit_font(texts: Sequence[str], max_w: float, fs: float, fs_min: float = 5.5) -> float:
    """Largest font size (<= fs, 0.5 pt steps) at which every line fits within ``max_w`` inches."""
    while fs > fs_min and max(_text_w(x, fs) for x in texts) > max_w:
        fs -= 0.5
    return fs


def _legend_cols(labels: Sequence[str], W: float, fs: float, handle: float = 0.45) -> int:
    """Number of legend columns that fit within ``W`` inches."""
    widest = max(_text_w(x, fs) for x in labels) + handle
    return max(1, min(len(labels), int((W - 0.1) // widest)))


def _end_labels(values: Sequence[float], lo: float, hi: float) -> list[float]:
    """Vertical positions of line-end labels: spread to avoid overlap; many labels are packed tighter, kept inside."""
    n = len(values)
    gap = (hi - lo) * min(0.075, 0.98 / max(n, 1))
    ys = np.asarray(spread(list(values), gap))
    if n > MANY:
        ys = ys + max(lo + gap / 2 - ys.min(), 0.0)
        ys = ys - max(ys.max() - (hi - gap / 2), 0.0)
    return ys.tolist()


# ==================================================================== method axis (short labels + LEARN bracket)
def _method_axis(ax, methods: Sequence[str], bracket: bool = True) -> None:
    keys = [key_of(m) for m in methods]
    n_learn = sum(k in LEARN_GROUP for k in keys)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([SHORT.get(k, label(m)) if n_learn >= 2 else label(m) for k, m in zip(keys, methods)])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_xlim(-0.6, len(methods) - 0.4)
    idx = [i for i, k in enumerate(keys) if k in LEARN_GROUP]
    if not (bracket and idx and len(idx) >= 2 and idx == list(range(idx[0], idx[-1] + 1))):
        return
    fs = CTX.preset.small
    tr = transforms.offset_copy(transforms.blended_transform_factory(ax.transData, ax.transAxes), fig=ax.figure,
                                y=-(fs + 6.0), units="points")
    a, b = idx[0] - 0.35, idx[-1] + 0.35
    ax.add_line(Line2D([a, a, b, b], [0.012, 0, 0, 0.012], transform=tr, color=INK_2, lw=0.6, clip_on=False,
                       gid="learn_bracket"))
    ax.text((a + b) / 2, 0, "LEARN", transform=transforms.offset_copy(tr, fig=ax.figure, y=-1.5, units="points"),
            ha="center", va="top", fontsize=fs - 0.5, color=INK_2, gid="learn_bracket")
    ax._lact_methods = list(methods)  # lets _fit_ticklabels restore the full labels if they must be rotated


def _crowded(ax, r, gap_in: float = 0.1) -> bool:
    boxes = [tl.get_window_extent(r) for tl in ax.get_xticklabels() if tl.get_text() and tl.get_visible()]
    return any(b.x1 + gap_in * ax.figure.dpi > c.x0 for b, c in zip(boxes[:-1], boxes[1:]))


def _fit_ticklabels(fig, angle: float = 30.0) -> None:
    """Overlapping/crowded x tick labels (large text, narrow panel): first shrink by 1 pt; if still crowded, rotate."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        if _crowded(ax, r):
            ax.tick_params(axis="x", labelsize=CTX.preset.small - 1)
            fig.canvas.draw()
        if _crowded(ax, r):
            ms = getattr(ax, "_lact_methods", None)
            if ms:                                    # drop the LEARN bracket, use the full labels
                for art in [*ax.lines, *ax.texts]:
                    if art.get_gid() == "learn_bracket":
                        art.remove()
                ax.set_xticklabels([label(m) for m in ms])
            for tl in ax.get_xticklabels():
                tl.set_rotation(angle)
                tl.set_ha("right")
                tl.set_rotation_mode("anchor")
