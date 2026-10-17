"""Four-level layout of the encoder-decoder: set by hand (channels, depths) or by the RegNet rule.

Quantised linear rule (Radosavovic et al., 2020, "Designing Network Design Spaces"):

    u_j = w0 + wa * j  (j = 0..depth-1),   s_j = round(log_wm(u_j / w0)),   w_j = w0 * wm^{s_j}  (rounded to x8)

Blocks sharing a width w_j form a stage: the number of stages, their widths and depths all follow from four numbers
(w0, wa, wm, depth). The rule fixes only widths and depths, not the block type, so the dense block can be placed in
a "RegNet layout" through the ``regnet`` configuration entry. The RegNetY line also
adds squeeze-and-excitation to every block (``se_ratio`` = 0.25 in the configuration).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

# RegNetX-200MF and RegNetY-200MF share this rule (w = [24, 56, 152, 368], d = [1, 1, 4, 7]); Y = X + SE
REGNET_200MF = {"w0": 24, "wa": 36.44, "wm": 2.49, "depth": 13}
SE_RATIO_Y = 0.25                                                    # SE ratio of the RegNetY line


def regnet_stages(w0: float, wa: float, wm: float, depth: int, group_width: int | None = None, q: int = 8
                  ) -> tuple[list[int], list[int]]:
    """(width, depth) of each stage from the RegNet parameters; rounding as in the original (half to even).

    ``group_width``: snap widths to a multiple of the group width (X block, bottleneck ratio b = 1)."""
    if w0 <= 0 or wa < 0 or wm <= 1 or depth < 1:
        raise ValueError("need w0 > 0, wa >= 0, wm > 1, depth >= 1")
    widths: list[int] = []
    for j in range(int(depth)):
        s = round(math.log((w0 + wa * j) / w0) / math.log(wm))
        widths.append(max(q, int(round(w0 * wm ** s / q) * q)))
    stage_w = sorted(set(widths))
    stage_d = [widths.count(w) for w in stage_w]
    if group_width:
        stage_w = [int(round(w / min(group_width, w)) * min(group_width, w)) for w in stage_w]
    return stage_w, stage_d


def stage_layout(channels: Sequence[int] | None, depths: Sequence[int] | None, regnet: Mapping | None = None,
                 group_width: int | None = None) -> tuple[list[int], list[int]]:
    """Widths and depths of the levels. With ``regnet`` (w0, wa, wm, depth) the RegNet rule is used (it must yield
    exactly 4 stages); otherwise the hand-set ``channels``/``depths``."""
    if regnet is not None:
        unknown = sorted(set(regnet) - {"w0", "wa", "wm", "depth"})
        if unknown:
            raise ValueError(f"regnet: invalid keys {unknown}; expected w0, wa, wm, depth")
        w, d = regnet_stages(**{**REGNET_200MF, **regnet}, group_width=group_width)
        if len(w) != 4:
            raise ValueError(f"the encoder-decoder needs exactly 4 levels, the RegNet parameters give {len(w)}: {w}")
        return w, d
    if channels is None or depths is None:
        raise ValueError("need channels and depths, or a regnet entry")
    return list(channels), list(depths)
