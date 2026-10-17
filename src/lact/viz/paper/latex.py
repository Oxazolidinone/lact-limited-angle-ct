r"""Booktabs-style LaTeX result tables: mean ± std, bold = best, underline = second, † = significant difference.

    \usepackage{booktabs}  % in the preamble
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .style import CTX, geom_label, label

HIGHER_BETTER = {"psnr": True, "ssim": True, "psnr_body": True, "ssim_body": True, "rmse": False, "rmse_hu": False}
DIGITS = {"psnr": 2, "psnr_body": 2, "ssim": 4, "ssim_body": 4, "rmse_hu": 1, "rmse": 4}
HEAD = {"psnr": "PSNR (dB)", "ssim": "SSIM", "rmse_hu": "RMSE (HU)", "psnr_body": "PSNR (dB)", "ssim_body": "SSIM"}


def _tex(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def results_table(summary: Mapping[str, pd.DataFrame], geoms: Sequence[str], methods: Sequence[str],
                  sig: Mapping[tuple[str, str], bool] | None = None, caption: str = "", tab_label: str = "tab:main",
                  groups: Mapping[str, Sequence[str]] | None = None) -> str:
    """summary: {metric: DataFrame(geometry, method, mean, std)}. sig[(geometry, method)] = True => append †.
    groups: group name -> methods (e.g. {"Non-learned": [fbp, tv], "Unrolled": [...]}) to draw group rules."""
    metrics = list(summary)
    cols = "l" + "".join("c" * len(metrics) for _ in geoms)
    lines = [r"\begin{table*}[t]" if len(geoms) * len(metrics) > 2 else r"\begin{table}[t]", r"\centering",
             r"\small", r"\setlength{\tabcolsep}{4pt}"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
        lines.append(rf"\label{{{tab_label}}}")
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    method_head = "Method" if CTX.lang == "en" else "Phương pháp"
    head1 = [method_head] + [rf"\multicolumn{{{len(metrics)}}}{{c}}{{{geom_label(g).rstrip('°')}$^\circ$}}"
                             for g in geoms]
    lines.append(" & ".join(head1) + r" \\")
    rules = []
    for i in range(len(geoms)):
        a = 2 + i * len(metrics)
        rules.append(rf"\cmidrule(lr){{{a}-{a + len(metrics) - 1}}}")
    lines.append("".join(rules))
    lines.append(" & ".join([""] + [HEAD.get(m, m) + (r"$\uparrow$" if HIGHER_BETTER.get(m, True) else
                                                        r"$\downarrow$") for m in metrics] * len(geoms)) + r" \\")
    lines.append(r"\midrule")

    rank: dict[tuple[str, str], dict[str, int]] = {}
    for m in metrics:
        s = summary[m]
        for g in geoms:
            sg = s[(s.geometry == g) & s.method.isin(methods)].dropna(subset=["mean"])
            hb = HIGHER_BETTER.get(m, True)
            vals = sg.set_index("method")["mean"].round(DIGITS.get(m, 3))
            order = vals.sort_values(ascending=not hb).unique()
            rank[(m, g)] = {k: int(np.where(order == v)[0][0]) for k, v in vals.items()}

    def cell(m: str, g: str, meth: str) -> str:
        s = summary[m]
        r = s[(s.geometry == g) & (s.method == meth)]
        if r.empty or not np.isfinite(r["mean"].iloc[0]):
            return "--"
        d = DIGITS.get(m, 3)
        mu, sd = r["mean"].iloc[0], r["std"].iloc[0]
        txt = f"{mu:.{d}f}"
        rk = rank[(m, g)].get(meth, 99)
        if rk == 0:
            txt = rf"\textbf{{{txt}}}"
        elif rk == 1:
            txt = rf"\underline{{{txt}}}"
        if np.isfinite(sd):
            txt += rf"{{\scriptsize$\pm${sd:.{d}f}}}"
        if sig and sig.get((g, meth)):
            txt += r"$^\dagger$"
        return txt

    blocks = groups or {"": list(methods)}
    for bi, (gname, ms) in enumerate(blocks.items()):
        if bi:
            lines.append(r"\midrule")
        if gname:
            lines.append(rf"\multicolumn{{{1 + len(metrics) * len(geoms)}}}{{l}}{{\emph{{{_tex(gname)}}}}} \\")
        for meth in ms:
            row = [_tex(label(meth))] + [cell(m, g, meth) for g in geoms for m in metrics]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{table*}" if len(geoms) * len(metrics) > 2 else r"\end{table}"]
    return "\n".join(lines) + "\n"
