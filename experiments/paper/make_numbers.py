"""Generate the paper numbers (bao-cao/paper) from per-slice evaluation results; never copy numbers by hand.

    .venv/bin/python experiments/paper/make_numbers.py \
        --eval ../ket-qua/<batch>/outputs_joint/eval/per_slice ../ket-qua/<batch>/outputs_detect/eval/per_slice \
        --out ../bao-cao/paper/generated

Reads every ``<run_id>__test__<geometry>.csv`` in the --eval directories; each method keeps only the slices of the
SAME DOMAIN as its training dataset (slice ids: L... = Mayo, DL... = DeepLesion, DT... = 2DeteCT); table entries are
per-slice means of PSNR/SSIM (images normalised to [0, 1], data_range = 1), the thesis evaluation protocol.
Writes:
  numbers.tex      macros \\res{dataset}{angle}{method}{psnr|ssim} and \\gain{dataset}{angle} for the text
  tab_main.tex     main table: 3 datasets x 2 angles, best in bold, second underlined,
                   † = significantly better than the strongest competitor (paired per-slice Wilcoxon, p < 0.05)
  tab_qn.tex       data-step ablation: first order (gd) and L-BFGS on 4 regularizers (Mayo, DeepLesion)
Missing results show as "--", so the paper always compiles; re-run when more results arrive.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path

import numpy as np

DATASETS = {"mayo": ("Mayo", "L", "fan"), "dl": ("DeepLesion", "DL", "fan"), "dt": ("2DeteCT", "DT", "dt")}
ANGLES = {"a90": "90", "a120": "120"}
# (key, table label, run_id prefix pattern; {ds} = mayo | dl | dt, {g} = geometry)
MAIN = [
    ("fbp", "FBP", "fbp_{g}"),
    ("tv", "TV", "tv_{g}"),
    ("fbpconvnet", r"FBPConvNet~\cite{jin2017fbpconvnet}", "cmp_{ds}_fbpconvnet_{g}"),
    ("lpd", r"Learned Primal-Dual~\cite{adler2018lpd}", "cmp_{ds}_lpd_{g}"),
    ("learn", r"LEARN~\cite{chen2018learn}", "cmp_{ds}_learn_orig_{g}"),
    ("regformer", r"RegFormer~\cite{xia2024regformer}", "base_{ds}_regformer_{g}"),
    ("dpma", r"DPMA~\cite{wu2025dpma}", "dpma_{ds}_dpma_{g}"),
    ("cgglore", r"CG-GLORE~\cite{cgglore2026}", "cmp_{ds}_cg_glore_{g}"),
    ("ours", r"\oursname{}", "lbfgs_{ds}_learn_densenet_{g}"),
]
NONLEARNED = {"fbp", "tv"}
REGS = [("unet", "U-Net"), ("resnet", "ResNet"), ("densenet", "DenseNet"), ("regformer", "RegFormer")]


def geom(ds: str, a: str) -> str:
    return f"{DATASETS[ds][2]}_{a}"


def find(files: dict[str, Path], prefix: str, g: str) -> Path | None:
    # drop the auxiliary "<run_id>_norefine" results (DPMA without refinement,
    # experiments/analysis/dpma_norefine_eval.py): the comparison evaluates DPMA with its 300 refinement iterations, as
    # the original code and the CG-GLORE paper do
    hits = [p for n, p in files.items() if n.endswith(f"__test__{g}.csv") and "_norefine__" not in n
            and (n.startswith(prefix + "_lr") or n == f"{prefix}__test__{g}.csv")]
    return sorted(hits)[-1] if hits else None


def load(path: Path | None, pid_prefix: str) -> dict[tuple[str, str], tuple[float, float]] | None:
    """{(slice id, slice index): (psnr, ssim)} of the in-domain slices."""
    if path is None:
        return None
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["pid"].startswith(pid_prefix):
                out[(r["pid"], r["idx"])] = (float(r["psnr"]), float(r["ssim"]))
    return out or None


def wilcoxon_p(a: dict, b: dict) -> float:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return float("nan")
    keys = sorted(set(a) & set(b))
    if len(keys) < 10:
        return float("nan")
    return float(wilcoxon([a[k][0] for k in keys], [b[k][0] for k in keys]).pvalue)


def fmt(v: float | None, metric: str) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:.2f}" if metric == "psnr" else f"{v:.4f}"


def n_params(path: Path | None) -> int | None:
    if path is None:
        return None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            return int(r["n_params"]) if r.get("n_params", "").strip().isdigit() else None
    return None


def collect(files: dict[str, Path]) -> dict:
    """res[(ds, a, key)] = {'psnr', 'ssim', 'n', 'slices'}; res[('params', key)] = parameter count."""
    res = {}
    for ds, (_, pid, _) in DATASETS.items():
        for a in ANGLES:
            g = geom(ds, a)
            for key, _, pat in MAIN:
                path = find(files, pat.format(ds=ds, g=g), g)
                d = load(path, pid)
                if d:
                    v = np.array(list(d.values()))
                    res[(ds, a, key)] = {"psnr": float(v[:, 0].mean()), "ssim": float(v[:, 1].mean()), "n": len(d),
                                         "slices": d}
                    if n_params(path) and ("params", key) not in res:
                        res[("params", key)] = n_params(path)
            for r, _ in REGS:
                if r == "regformer":
                    gd = f"base_{ds}_regformer_{g}"
                    qn = f"lbfgs_{ds}_regformer_{g}"
                else:
                    gd = (f"e3_main_learn_{r}_{g}" if ds == "mayo" else f"base_{ds}_learn_{r}_{g}")
                    qn = f"lbfgs_{ds}_learn_{r}_{g}"
                for upd, pat in (("gd", gd), ("qn", qn)):
                    d = load(find(files, pat, g), pid)
                    if d:
                        v = np.array(list(d.values()))
                        res[(ds, a, f"{r}_{upd}")] = {"psnr": float(v[:, 0].mean()), "ssim": float(v[:, 1].mean()),
                                                     "n": len(d), "slices": d}
    return res


def ranks(res: dict, ds: str, a: str, metric: str, keys: list[str]) -> tuple[str | None, str | None]:
    vals = sorted(((res[(ds, a, k)][metric], k) for k in keys if (ds, a, k) in res), reverse=True)
    return (vals[0][1] if vals else None), (vals[1][1] if len(vals) > 1 else None)


def cell(res, ds, a, k, metric, best, second, mark="") -> str:
    v = res.get((ds, a, k), {}).get(metric)
    s = fmt(v, metric)
    if s == "--":
        return s
    s += mark
    return rf"\textbf{{{s}}}" if k == best else rf"\underline{{{s}}}" if k == second else s


def table_main(res: dict) -> tuple[str, dict]:
    keys = [k for k, _, _ in MAIN]
    learned = [k for k in keys if k not in NONLEARNED]
    sig, gains = {}, {}
    for ds in DATASETS:
        for a in ANGLES:
            comp = [k for k in learned if k != "ours" and (ds, a, k) in res]
            if (ds, a, "ours") in res and comp:
                rival = max(comp, key=lambda k: res[(ds, a, k)]["psnr"])
                p = wilcoxon_p(res[(ds, a, "ours")]["slices"], res[(ds, a, rival)]["slices"])
                gain = res[(ds, a, "ours")]["psnr"] - res[(ds, a, rival)]["psnr"]
                sig[(ds, a)] = p < 0.05 and gain > 0
                gains[(ds, a)] = (gain, rival, p)
    head = " & ".join(rf"\multicolumn{{4}}{{c}}{{{DATASETS[ds][0]}}}" for ds in DATASETS)
    mid = " & ".join(rf"\multicolumn{{2}}{{c}}{{{ANGLES[a]}$^\circ$}}" for ds in DATASETS for a in ANGLES)
    cm1 = " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3))
    cm2 = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(6))
    lines = [r"\begin{tabular}{l" + "cc" * 6 + "}", r"\toprule",
             rf"\multirow{{3}}{{*}}{{Method}} & {head} \\", cm1, rf" & {mid} \\", cm2,
             " & " + " & ".join(r"PSNR$\uparrow$ & SSIM$\uparrow$" for _ in range(6)) + r" \\", r"\midrule"]
    for k, label, _ in MAIN:
        if k == "ours":
            lines.append(r"\midrule")
        row = [label]
        for ds in DATASETS:
            for a in ANGLES:
                for m in ("psnr", "ssim"):
                    b, s = ranks(res, ds, a, m, learned)
                    mark = r"$^\dagger$" if (k == "ours" and m == "psnr" and sig.get((ds, a))) else ""
                    row.append(cell(res, ds, a, k, m, b, s, mark))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", gains


def table_qn(res: dict) -> str:
    cols = [(ds, a) for ds in ("mayo", "dl") for a in ANGLES]
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASETS[ds][0]}}}" for ds in ("mayo", "dl"))
    lines = [r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
             rf"Regularizer & Data step & {head} \\", r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
             " & & " + " & ".join(rf"{ANGLES[a]}$^\circ$" for _, a in cols) + r" \\", r"\midrule"]
    for r, name in REGS:
        for upd, lab in (("gd", "first order"), ("qn", "damped L-BFGS")):
            row = [name if upd == "gd" else "", lab]
            for ds, a in cols:
                v = res.get((ds, a, f"{r}_{upd}"), {}).get("psnr")
                s = fmt(v, "psnr")
                if upd == "qn" and s != "--" and (ds, a, f"{r}_gd") in res:
                    d = v - res[(ds, a, f"{r}_gd")]["psnr"]
                    s += rf" \textcolor{{{'green!50!black' if d > 0 else 'red!70!black'}}}{{\scriptsize({d:+.2f})}}"
                row.append(s)
            lines.append(" & ".join(row) + r" \\")
        if r != REGS[-1][0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def numbers(res: dict, gains: dict) -> str:
    out = [f"% Generated by kltn-lact/experiments/paper/make_numbers.py "
           f"({_dt.datetime.now().isoformat(timespec='minutes')}); "
           "do not edit by hand.",
           r"\providecommand{\res}[4]{\ifcsname r@#1@#2@#3@#4\endcsname\csname r@#1@#2@#3@#4\endcsname\else--\fi}",
           r"\providecommand{\gain}[2]{\ifcsname g@#1@#2\endcsname\csname g@#1@#2\endcsname\else--\fi}",
           r"\providecommand{\nslices}[1]{\ifcsname n@#1\endcsname\csname n@#1\endcsname\else--\fi}",
           ]
    out.append(r"\providecommand{\nparams}[1]{\ifcsname p@#1\endcsname\csname p@#1\endcsname\else--\fi}")
    for key, v in sorted((k[1], v) for k, v in res.items() if k[0] == "params"):
        out.append(rf"\expandafter\def\csname p@{key}\endcsname{{{v / 1e6:.2f}\,M}}")
    for (ds, a, k), v in sorted((k, v) for k, v in res.items() if k[0] != "params"):
        for m in ("psnr", "ssim"):
            out.append(rf"\expandafter\def\csname r@{ds}@{a}@{k}@{m}\endcsname{{{fmt(v[m], m)}}}")
    for (ds, a), (g, rival, p) in sorted(gains.items()):
        out.append(rf"\expandafter\def\csname g@{ds}@{a}\endcsname{{{g:+.2f}}}")
        out.append(f"% {ds} {a}: strongest competitor {rival}, Wilcoxon p = {p:.3g}")
    for ds in DATASETS:
        n = max((v["n"] for k, v in res.items() if k[0] == ds), default=0)
        if n:
            out.append(rf"\expandafter\def\csname n@{ds}\endcsname{{{n}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", nargs="+", required=True, help="eval/per_slice directories")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = {}
    for d in a.eval:
        for p in Path(d).glob("*.csv"):
            files[p.name] = p
    res = collect(files)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main_tab, gains = table_main(res)
    (out / "tab_main.tex").write_text(main_tab, encoding="utf-8")
    (out / "tab_qn.tex").write_text(table_qn(res), encoding="utf-8")
    (out / "numbers.tex").write_text(numbers(res, gains), encoding="utf-8")
    print(f"{len(files)} result files, {len(res)} filled cells. Wrote {out}/numbers.tex, tab_main.tex, tab_qn.tex")
    for (ds, a), (g, rival, p) in sorted(gains.items()):
        print(f"  {ds} {a}: ours - {rival} = {g:+.2f} dB (Wilcoxon p = {p:.3g})")
    missing = [f"{ds}/{a}/{k}" for ds in DATASETS for a in ANGLES for k, _, _ in MAIN if (ds, a, k) not in res]
    if missing:
        print(f"  {len(missing)} cells still missing: {missing[:12]}{' ...' if len(missing) > 12 else ''}")


if __name__ == "__main__":
    main()
