"""Draw the thesis/paper figure and table set from the compact folder written by experiments/figures/export_results.py.

    python experiments/figures/make_figures.py --data <folder> --out <figure folder> --lang vi --preset thesis
    python experiments/figures/make_figures.py --data <folder> --out <figure folder> --lang en --preset ieee

Each figure skips itself when data are missing and uses only the methods present. Outputs: <name>.pdf (vector, for
LaTeX) + <name>.png (300 dpi, for Word) + table_main.tex. Figure labels follow --lang.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from lact.config import load_geometry
from lact.core.analysis.visibility import visibility_gates
from lact.viz import paper as pp
from lact.viz.paper.figures import roi_by_structure
from lact.viz.style import ARCH_TO_REG

ARCH_OF = ARCH_TO_REG
LEARN = list(ARCH_TO_REG.values())
ALL = ["fbp", "tv", *LEARN]
GEOMS = ["fan_a90", "fan_a120", "fan_a150"]
MAIN_GEOMS = ["fan_a90", "fan_a120"]


class Skip(Exception):
    pass


def has(df: pd.DataFrame, methods, geom=None) -> list[str]:
    d = df if geom is None else df[df.geometry == geom]
    s = set(d.method)
    return [m for m in methods if m in s]


def load_curves(runs_dir: Path, exp: str) -> dict:
    """{(method, geometry, lr): [metrics per seed]}"""
    out: dict = {}
    for d in sorted((runs_dir / exp).glob(f"{exp}_*")):
        f = d / "metrics.csv"
        if not f.exists() or "_fan_" not in d.name:
            continue
        model, rest = d.name[len(exp) + 1:].split("_fan_")
        lr = float(rest.split("_lr")[1].rsplit("_s", 1)[0])
        out.setdefault((ARCH_OF.get(model, model), "fan_" + rest.split("_lr")[0], lr), []).append(pd.read_csv(f))
    return out


class Data:
    def __init__(self, root: Path):
        self.root = root
        self.sl = pd.read_csv(root / "slices.csv")
        self.meta = json.loads((root / "meta.json").read_text())
        self.params = json.loads((root / "params.json").read_text()) if (root / "params.json").exists() else {}
        self.npz = np.load(root / "images.npz")
        self.costs = pd.read_csv(root / "costs.csv") if (root / "costs.csv").exists() else pd.DataFrame()
        self.runs = root / "runs"
        self.mu = (self.meta["mu_ref"], self.meta["mu_water"])
        self.img_size = int(self.meta.get("img_size", 256))
        # main results: baselines + E3 (90°/120°) + E5 (150°); E4 only enters the pipeline's parameter-matched table
        self.main = self.sl[self.sl.exp.isin(["baseline", "e3_main", "e5_a150"])]

    def images(self, geom: str) -> dict:
        out: dict = {}
        for k in self.npz.files:
            g, m, pid, idx = k.split("/")
            if g == geom:
                out.setdefault((pid, int(idx)), {})[m] = self.npz[k]
        return out

    def block(self, geom: str, key, methods, lab: str) -> dict:
        d = self.images(geom)[key]
        row = self.sl[(self.sl.geometry == geom) & (self.sl.pid == key[0]) & (self.sl.idx == key[1])]
        if row.groupby("method").seed.nunique().max() > 1:        # several seeds left: illustrations use seed 0
            row = row[(row.seed == 0) | row.method.isin(["fbp", "tv"])]
        row = row.groupby("method")[["psnr", "ssim"]].mean()
        return {"label": lab, "ref": d["gt"], "images": {m: d[m] for m in methods},
                "roi": roi_by_structure(d["gt"], *self.mu, size=44),
                "metrics": {m: (row.loc[m, "psnr"], row.loc[m, "ssim"]) for m in methods if m in row.index}}


# ============================================================================ figures
def fig_qualitative(D: Data, out, fmt, lang):
    geoms = [g for g in MAIN_GEOMS if D.images(g)]
    common = [m for m in ALL if all(any(m in v for v in D.images(g).values()) for g in geoms)]
    if len(geoms) == 2 and len(common) >= 5:          # block = scan range, same slice
        keys = [sorted(set(D.images(geoms[0])) & set(D.images(g))) for g in geoms]
        key = keys[1][len(keys[1]) // 2]
        blocks = [D.block(g, key, common, pp.geom_label(g)) for g in geoms]
        methods = common
    else:                                              # block = two slices of the scan range with the most methods
        g = max(geoms, key=lambda g_: len(has(D.main, ALL, g_)))
        methods = [m for m in has(D.main, ALL, g) if all(m in v for v in D.images(g).values())]
        ks = sorted(D.images(g))
        sA, sB = ("Slice A", "Slice B") if lang == "en" else ("Lát A", "Lát B")
        blocks = [D.block(g, ks[len(ks) // 2], methods, sA), D.block(g, ks[-1], methods, sB)]
    if len(methods) < 3:
        raise Skip("images missing")
    return pp.save(pp.qualitative(blocks, methods, *D.mu, window=(400, 40), err_max=150), out,
                   "fig_qualitative", fmt)


def fig_strip(D: Data, out, fmt, lang):
    geoms = [g for g in MAIN_GEOMS if len(has(D.main, LEARN, g))]
    made = []
    for metric in ("psnr", "ssim"):
        made += pp.save(pp.metric_strip(D.main, metric, geoms, ALL), out, f"fig_{metric}_by_method", fmt)
    return made


def fig_forest(D: Data, out, fmt, lang):
    geoms = [g for g in MAIN_GEOMS if len(has(D.main, LEARN, g)) >= 2]
    if not geoms:
        raise Skip("methods missing")
    ms = ["fbp", *LEARN]
    pairs = pd.concat([pp.stats.paired_vs(D.main, "psnr", "tv", ms, g) for g in geoms])
    return pp.save(pp.paired_forest(pairs, "psnr", geoms, has(D.main, ms), ref="tv",
                                    w="full" if len(geoms) > 1 else "col"), out, "fig_delta_vs_tv", fmt)


def fig_angle(D: Data, out, fmt, lang):
    summ = pp.stats.summarize(D.main, "psnr")
    ms = [m for m in ALL if summ[summ.method == m].geometry.nunique() >= 2]
    if not ms:
        raise Skip("needs >= 2 scan ranges")
    return pp.save(pp.angle_lines(summ, ms, "PSNR (dB)"), out, "fig_psnr_vs_angle", fmt)


def fig_curves(D: Data, out, fmt, lang):
    made = []
    e3 = load_curves(D.runs, "e3_main")
    for g in MAIN_GEOMS:
        cur = {m: dfs for (m, gg, _lr), dfs in e3.items() if gg == g}
        if cur:
            made += pp.save(pp.curves(cur), out, f"fig_curves_{g.replace('fan_', '')}", fmt)
    if not made:
        raise Skip("no metrics.csv")
    return made


def fig_lr(D: Data, out, fmt, lang):
    sweep: dict = {}
    for (m, _g, lr), dfs in load_curves(D.runs, "e2_sweep").items():
        sweep.setdefault(m, {})[lr] = dfs[0]
    if not sweep:
        raise Skip("no E2")
    bf = D.runs / "e2_sweep/best_lr.json"
    best = json.loads(bf.read_text())["best_lr"] if bf.exists() else None
    return pp.save(pp.lr_sweep({m: sweep[m] for m in LEARN if m in sweep}, best=best), out, "fig_lr_sweep", fmt)


def fig_visibility(D: Data, out, fmt, lang):
    maps, gates, srcs, n = {}, {}, {}, D.img_size
    for g in GEOMS:
        geom = load_geometry(g)
        G = visibility_gates(geom, n, 16)
        gates[g], maps[g] = G, G.mean(0)
        px = geom.pixel_size_mm(n)
        s = geom.vectors()[:, :2]
        srcs[g] = np.stack([s[:, 0] / px + n / 2 - 0.5, n / 2 - 0.5 - s[:, 1] / px], axis=1)
    return pp.save(pp.visibility(maps, gates, srcs), out, "fig_visibility", fmt)


def fig_error_split(D: Data, out, fmt, lang):
    if "err_inv" not in D.sl or D.sl.err_inv.isna().all():
        raise Skip("no err_vis/err_inv yet")
    made = []
    for g in MAIN_GEOMS:
        ms = has(D.main, ALL, g)
        if len(ms) >= 3:
            made += pp.save(pp.error_split(D.main, ms, g, D.img_size ** 2), out,
                            f"fig_error_split_{g.replace('fan_', '')}", fmt)
    return made


def fig_stage(D: Data, out, fmt, lang):
    made = []
    for g in MAIN_GEOMS:
        sp = {m: D.params[f"{g}/{m}"] for m in LEARN if f"{g}/{m}" in D.params}
        if sp:
            made += pp.save(pp.stage_params(sp), out, f"fig_stage_params_{g.replace('fan_', '')}", fmt)
    if not made:
        raise Skip("no params.json yet")
    return made


def fig_pareto(D: Data, out, fmt, lang):
    made = []
    for g in ["fan_a120", "fan_a90"]:
        summ = pp.stats.summarize(D.main[D.main.geometry == g], "psnr")
        summ = summ[summ.method.isin(LEARN)]
        if len(summ) < 2:
            continue
        n_par = {}
        for k, info in D.meta.get("runs", {}).items():
            gm, m = k.split("/")
            if gm == g:
                s = info.get("summary", {})
                n_par[m] = s.get("n_params_regularizer") or s.get("n_params")
        d = summ.assign(params_m=summ.method.map(n_par) / 1e6).dropna(subset=["params_m"])
        if len(d) >= 2:
            made += pp.save(pp.pareto(d, "params_m", pp.t("params"), "PSNR (dB)", logx=True), out,
                            f"fig_pareto_params_{g.replace('fan_', '')}", fmt)
        if not D.costs.empty and "median_ms" in D.costs:
            c = D.costs[D.costs.geometry_test == g].copy()
            c["method"] = c.arch.map(lambda a: ARCH_OF.get(a, a))
            d2 = summ.merge(c[["method", "median_ms"]].drop_duplicates("method"), on="method")
            if len(d2) >= 2:
                made += pp.save(pp.pareto(d2, "median_ms", pp.t("time"), "PSNR (dB)"), out,
                                f"fig_pareto_time_{g.replace('fan_', '')}", fmt)
    if not made:
        raise Skip("parameter counts missing")
    return made


def table_main(D: Data, out, fmt, lang):
    geoms = [g for g in MAIN_GEOMS if len(has(D.main, LEARN, g))]
    ms = has(D.main, ALL)
    summ = {m: pp.stats.summarize(D.main, m) for m in ("psnr", "ssim")}
    ref = "tv"
    sig = {}
    for g in geoms:
        for _, r in pp.stats.paired_vs(D.main, "psnr", ref, ms, g).iterrows():
            sig[(g, r.method)] = bool(r.p_holm < 0.05)
    groups = {("Non-learned" if lang == "en" else "Không học"): [m for m in ["fbp", "tv"] if m in ms],
              "LEARN (unrolled)": [m for m in LEARN if m in ms]}
    split = D.meta.get("split", "test")
    cap = (f"Image quality on the {split} set (mean$\\pm$std; std over seeds when available). Bold: best; "
           f"underline: second; $^\\dagger$: paired Wilcoxon vs.\\ {pp.label(ref)}, Holm-corrected $p<0.05$."
           if lang == "en" else
           f"Chất lượng ảnh trên tập {split} (trung bình$\\pm$độ lệch chuẩn). "
           f"Đậm: tốt nhất; gạch chân: thứ nhì; "
           f"$^\\dagger$: Wilcoxon ghép cặp so với {pp.label(ref)}, hiệu chỉnh Holm $p<0.05$.")
    tex = pp.results_table(summ, geoms, ms, sig=sig, caption=cap, groups=groups)
    p = Path(out) / "table_main.tex"
    p.write_text(tex, encoding="utf-8")
    return [p]


FIGS = [fig_qualitative, fig_strip, fig_forest, fig_angle, fig_curves, fig_lr, fig_visibility, fig_error_split,
        fig_stage, fig_pareto, table_main]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="en", choices=["en", "vi"])
    ap.add_argument("--preset", default="ieee", choices=sorted(pp.PRESETS))
    ap.add_argument("--family", default="sans", choices=["sans", "serif"])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--only", nargs="*", default=[], help="draw only figures whose name contains these strings")
    a = ap.parse_args()
    pp.use(a.preset, lang=a.lang, family=a.family)
    D = Data(Path(a.data))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in FIGS:
        if a.only and not any(s in fn.__name__ for s in a.only):
            continue
        try:
            for p in fn(D, out, a.formats, a.lang):
                if str(p).endswith((".pdf", ".tex")):
                    print("  ", p.name)
        except Skip as e:
            print(f"  skipping {fn.__name__}: {e}")
        except Exception:                                # one failing figure does not block the others
            print(f"  ERROR {fn.__name__}:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
