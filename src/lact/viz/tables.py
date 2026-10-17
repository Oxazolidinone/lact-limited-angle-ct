"""Export tables as CSV + Markdown + LaTeX (every number in the reports comes from these files, never by hand)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r.values) + " |" for _, r in df.iterrows()]
    return "\n".join(lines) + "\n"


def _tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("±", r"$\pm$")


def to_latex(df: pd.DataFrame, caption: str = "") -> str:
    cols = "l" + "c" * (len(df.columns) - 1)
    out = [r"\begin{table}[t]", r"\centering", rf"\caption{{{_tex_escape(caption)}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(_tex_escape(c) for c in df.columns) + r" \\",
           r"\midrule"]
    out += [" & ".join(_tex_escape(v) for v in r.values) + r" \\" for _, r in df.iterrows()]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def write_table(df: pd.DataFrame, out: str | Path, name: str, caption: str = "") -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(to_markdown(df), encoding="utf-8")
    (out / f"{name}.tex").write_text(to_latex(df, caption), encoding="utf-8")
    return out / f"{name}.csv"
