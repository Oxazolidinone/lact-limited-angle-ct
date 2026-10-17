"""Phase-2 figure set (two datasets, L-BFGS data step) from the compact folder written by
experiments/figures/export_phase2.py.

    python experiments/figures/make_figures_phase2.py --data <figdata> --out <figure folder> --lang vi --preset thesis

For each test dataset (mayo, dl):
  fig_qual_<ds>            reconstructions of models trained on THAT dataset: FBP, TV, each first-order model and its
                           L-BFGS variant; block = scan range, same slice; with zoomed ROI and |error| maps
  fig_qual_cross_<ds>      as above with models trained on the OTHER dataset (cross-dataset)
  fig_psnr_<ds>, fig_ssim_<ds>   per-slice metrics of each method, one panel per scan range
  fig_lbfgs_delta_<ds>     per-slice ΔPSNR of the L-BFGS run versus the first-order run of the same model
  fig_lbfgs_vs_ctrl_<ds>   per-slice ΔPSNR of the L-BFGS run versus the control run (only after --control)
  fig_curves_<ds>_<angle>  validation PSNR per epoch: first-order run, then the L-BFGS run continuing it
  fig_stage_<ds>_<angle>   α_k, β_k per stage

Each figure skips itself when data are missing. Outputs: <name>.pdf (vector) and/or <name>.png (300 dpi).
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from lact.viz import paper as pp
from lact.viz.paper import style as ps
from lact.viz.paper.figures import roi_by_structure

MODELS = ["unet_ed", "resnet_ed", "densenet_ed", "regformer"]      # display order; other models (if any) come after
GEOMS = ["fan_a90", "fan_a120"]                                     # display order; other geometries come after
DATASET = {"mayo": "Mayo", "dl": "DeepLesion"}
OTHER = {"mayo": "dl", "dl": "mayo"}
SRC = "__source__"          # pseudo key for the "first-order run" label on the axis of the Δ figure


class Skip(Exception):
    pass


def register_methods(models) -> None:
    """Add RegFormer and the variants (+lbfgs, +ctrl) to the lact.viz.paper colour/label table: same colour as the
    original model, different marker."""
    ps.METHODS.setdefault("regformer", ("#0e7c86", "8", "RegFormer", "RegFormer"))
    for m in models:
        c, _, en, vi = ps.METHODS.get(m, (ps.INK_2, "o", m, m))
        ps.METHODS[f"{m}+lbfgs"] = (c, "D", f"{en} + L-BFGS", f"{vi} + L-BFGS")
        ps.METHODS[f"{m}+ctrl"] = (c, "X", f"{en} (longer)", f"{vi} (train thêm)")
    ps.METHODS[SRC] = (ps.INK, "", "the first-order run", "lượt train thường")
    ps.METHODS[SRC + "ctrl"] = (ps.INK, "", "its control run", "lượt đối chứng")


class Data:
    def __init__(self, root: Path):
        self.root = root
        self.sl = pd.read_csv(root / "slices.csv")
        self.sl["init_from"] = self.sl["init_from"].fillna("")
        self.meta = json.loads((root / "meta.json").read_text())
        self.params = json.loads((root / "params.json").read_text()) if (root / "params.json").exists() else {}
        self.npz = np.load(root / "images.npz")
        self.runs = root / "runs"
        self.mu = (self.meta["mu_ref"], self.meta["mu_water"])
        have = set(self.sl.geometry)
        self.geoms = [g for g in GEOMS if g in have] + sorted(have - set(GEOMS))
        learned = {str(m).split("+")[0] for m in self.sl[self.sl.train != "none"].method}
        self.models = [m for m in MODELS if m in learned] + sorted(learned - set(MODELS))

    def subset(self, dataset: str, train: str) -> pd.DataFrame:
        """Slices of test dataset ``dataset`` for methods trained on ``train`` (+ non-learned baselines)."""
        d = self.sl[self.sl.dataset == DATASET[dataset]]
        return d[d.train.isin([train, "none"])]

    def methods(self, df: pd.DataFrame, variants=("", "+lbfgs")) -> list[str]:
        have = set(df.method)
        learned = [m + v for m in self.models for v in variants if m + v in have]
        return [m for m in ("fbp", "tv") if m in have] + learned

    def images(self, geom: str, train: str) -> dict:
        """{(pid, idx): {method: image}} with gt, baselines and the methods trained on ``train``."""
        out: dict = {}
        for k in self.npz.files:
            parts = k.split("/")
            if parts[0] != geom:
                continue
            if parts[1] == "gt":
                out.setdefault((parts[2], int(parts[3])), {})["gt"] = self.npz[k]
            elif parts[1] in (train, "none"):
                out.setdefault((parts[3], int(parts[4])), {})[parts[2]] = self.npz[k]
        return out


def _block(D: Data, geom: str, train: str, key, methods, label: str) -> dict:
    d = D.images(geom, train)[key]
    row = D.sl[(D.sl.geometry == geom) & (D.sl.pid == key[0]) & (D.sl.idx == key[1])
               & D.sl.train.isin([train, "none"])].groupby("method")[["psnr", "ssim"]].mean()
    ref = d["gt"].astype(np.float32)
    return {"label": label, "ref": ref, "images": {m: d[m].astype(np.float32) for m in methods},
            "roi": roi_by_structure(ref, *D.mu, size=44),
            "metrics": {m: (row.loc[m, "psnr"], row.loc[m, "ssim"]) for m in methods if m in row.index}}


def _qualitative(D: Data, dataset: str, train: str, name: str, out, fmt):
    """Block = scan range, same slice of ``dataset``; columns = reference + methods trained on ``train``."""
    geoms = [g for g in D.geoms if D.images(g, train)]
    if not geoms:
        raise Skip("images missing")
    pids = set(D.sl[D.sl.dataset == DATASET[dataset]].pid)
    methods = D.methods(D.subset(dataset, train))
    keys = None
    for g in geoms:
        ok = {k for k, v in D.images(g, train).items() if k[0] in pids and "gt" in v and all(m in v for m in methods)}
        keys = ok if keys is None else keys & ok
    if not keys or len(methods) < 3:
        raise Skip(f"images missing for {dataset} (trained on {train})")
    keys = sorted(keys)
    key = keys[len(keys) // 2]
    blocks = [_block(D, g, train, key, methods, pp.geom_label(g)) for g in geoms]
    return pp.save(pp.qualitative(blocks, methods, *D.mu, window=(400, 40), err_max=150, max_cols=6), out, name, fmt)


def fig_qual(D: Data, out, fmt):
    made = []
    for ds in DATASET:
        try:
            made += _qualitative(D, ds, ds, f"fig_qual_{ds}", out, fmt)
        except Skip as e:
            print(f"  skipping fig_qual_{ds}: {e}")
    if not made:
        raise Skip("no dataset has enough images")
    return made


def fig_qual_cross(D: Data, out, fmt):
    made = []
    for ds in DATASET:
        try:
            made += _qualitative(D, ds, OTHER[ds], f"fig_qual_cross_{ds}", out, fmt)
        except Skip as e:
            print(f"  skipping fig_qual_cross_{ds}: {e}")
    if not made:
        raise Skip("no cross-dataset results")
    return made


def fig_strip(D: Data, out, fmt):
    made = []
    for ds in DATASET:
        d = D.subset(ds, ds)
        ms = D.methods(d, ("", "+lbfgs", "+ctrl"))
        geoms = [g for g in D.geoms if len(set(d[d.geometry == g].method) - {"fbp", "tv"})]
        if not geoms:
            continue
        for metric in ("psnr", "ssim"):
            made += pp.save(pp.metric_strip(d, metric, geoms, ms), out, f"fig_{metric}_{ds}", fmt)
    if not made:
        raise Skip("no results yet")
    return made


def _delta(D: Data, variant: str, ref_variant: str, ref_key: str, name: str, out, fmt):
    """Per-slice ΔPSNR of ``model + variant`` versus ``model + ref_variant`` (same model, same training data)."""
    made = []
    for ds in DATASET:
        d = D.subset(ds, ds)
        rows = []
        for g in D.geoms:
            for m in D.models:
                a, b = m + variant, m + ref_variant
                r = pp.stats.paired_vs(d[d.method.isin([a, b])], "psnr", b, [a], g)
                if not r.empty:
                    rows.append(r)
        if not rows:
            continue
        pairs = pd.concat(rows, ignore_index=True)
        geoms = [g for g in D.geoms if g in set(pairs.geometry)]
        ms = [m + variant for m in D.models if m + variant in set(pairs.method)]
        made += pp.save(pp.paired_forest(pairs, "psnr", geoms, ms, ref=ref_key, w="full" if len(geoms) > 1 else "col"),
                        out, f"{name}_{ds}", fmt)
    if not made:
        raise Skip("no pair of runs to compare yet")
    return made


def fig_lbfgs_delta(D: Data, out, fmt):
    return _delta(D, "+lbfgs", "", SRC, "fig_lbfgs_delta", out, fmt)


def fig_lbfgs_vs_ctrl(D: Data, out, fmt):
    return _delta(D, "+lbfgs", "+ctrl", SRC + "ctrl", "fig_lbfgs_vs_ctrl", out, fmt)


def _metrics(D: Data, run_id: str) -> pd.DataFrame | None:
    info = D.meta["runs"].get(run_id)
    f = D.runs / info["exp"] / run_id / "metrics.csv" if info else None
    return pd.read_csv(f) if f is not None and f.exists() else None


def fig_curves(D: Data, out, fmt):
    """Validation PSNR per epoch. From-scratch runs start at epoch 0; runs loading a checkpoint (train.init_from, if
    any) are drawn CONTINUING after their source run."""
    made = []
    runs = D.meta["runs"]
    for ds in DATASET:
        for g in D.geoms:
            cur: dict = {}
            for run_id, info in runs.items():
                if info["train"] != ds or info["geometry"] != g:
                    continue
                df = _metrics(D, run_id)
                if df is None or "val_psnr" not in df:
                    continue
                src = _metrics(D, info["init_from"]) if info.get("init_from") else None
                if src is not None and len(src):
                    e = df["epoch"].to_numpy(float)
                    df = df.assign(epoch=(e + 1 if e.size and e.min() == 0 else e) + float(len(src)))
                cur[info["method"]] = [df]
            cur = {m: cur[m] for m in D.methods(pd.DataFrame({"method": list(cur)}), ("", "+lbfgs", "+ctrl"))}
            if cur:
                made += pp.save(pp.curves(cur, w="full", h=2.6), out, f"fig_curves_{ds}_{g.replace('fan_', '')}", fmt)
    if not made:
        raise Skip("no metrics.csv")
    return made


def fig_stage(D: Data, out, fmt):
    made = []
    for ds in DATASET:
        for g in D.geoms:
            sp = {info["method"]: D.params[run_id] for run_id, info in D.meta["runs"].items()
                  if info["train"] == ds and info["geometry"] == g and run_id in D.params
                  and "alphas" in D.params[run_id]}
            sp = {m: sp[m] for m in D.methods(pd.DataFrame({"method": list(sp)}), ("", "+lbfgs", "+ctrl"))}
            if sp:
                made += pp.save(pp.stage_params(sp), out, f"fig_stage_{ds}_{g.replace('fan_', '')}", fmt)
    if not made:
        raise Skip("no params.json yet")
    return made


FIGS = [fig_qual, fig_qual_cross, fig_strip, fig_lbfgs_delta, fig_lbfgs_vs_ctrl, fig_curves, fig_stage]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="vi", choices=["en", "vi"])
    ap.add_argument("--preset", default="thesis", choices=sorted(pp.PRESETS))
    ap.add_argument("--family", default="sans", choices=["sans", "serif"])
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--only", nargs="*", default=[], help="draw only figures whose name contains these strings")
    a = ap.parse_args()
    pp.use(a.preset, lang=a.lang, family=a.family)
    D = Data(Path(a.data))
    register_methods(D.models)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fn in FIGS:
        if a.only and not any(s in fn.__name__ for s in a.only):
            continue
        try:
            for p in fn(D, out, a.formats):
                n += 1
                print("  ", p.name)
        except Skip as e:
            print(f"  skipping {fn.__name__}: {e}")
        except Exception:                                # one failing figure does not block the others
            print(f"  ERROR {fn.__name__}:\n{traceback.format_exc()}")
    print(f"{n} figure files in {out}")


if __name__ == "__main__":
    main()
