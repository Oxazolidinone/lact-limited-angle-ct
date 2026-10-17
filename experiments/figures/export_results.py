"""Collect the EVALUATION results (test split) of one outputs tree into a compact folder for the thesis/paper figures.

Run after the pipeline has finished (read-only), from the project directory:

    python experiments/figures/export_results.py --outputs outputs --out outputs/figdata

Outputs (inputs of make_figures.py):
  slices.csv  : exp, method, geometry, seed, pid, idx, psnr, ssim, rmse_hu, psnr_body, ssim_body, err_vis, err_inv
  images.npz  : <geometry>/<method>/<pid>/<idx> (seed 0) + <geometry>/gt/<pid>/<idx> for the illustrative slices
  params.json : per-stage alpha_k, beta_k (seed 0) ; meta.json ; costs.csv
  runs/<exp>/<run_id>/{metrics.csv, summary.json}
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lact.config import load_config_file, load_geometry
from lact.core.analysis.visibility import VisibilityOperator
from lact.infra.io.hdf5_store import ImageStore
from lact.infra.stores.results_store import ResultsStore
from lact.infra.stores.run_store import RunStore
from lact.services.factories import load_trained
from lact.services.reporting.seeds import POLICIES, keep_chosen, seed_selection
from lact.viz.style import ARCH_TO_REG

MODEL_KEY = ARCH_TO_REG
KEEP = ["psnr", "ssim", "rmse_hu", "psnr_body", "ssim_body"]


def method_of(row: pd.Series) -> str:
    """Method key from a run_id (<exp>_<model>_<geom>_lr.._s..); baselines -> regularizer (fbp/tv)."""
    if row.exp == "baseline":
        return str(row.regularizer)
    model = str(row.run_id)[len(row.exp) + 1:].split("_fan_")[0]
    return MODEL_KEY.get(model, model)


def load_store(rs: ResultsStore, split: str) -> pd.DataFrame:
    df = rs.load_slices(split=split)
    if df.empty:
        return df
    df = df[df.geometry_train == df.geometry_test].copy()
    df["method"] = df.apply(method_of, axis=1)
    df["geometry"] = df.geometry_test
    return df


def _tag(rs: ResultsStore, run_id: str, geom: str, split: str) -> str | None:
    tags = [t for t, info in rs.index().items() if info.get("run_id") == run_id
            and info.get("geometry_test") == geom and info.get("split") == split]
    return tags[0] if tags else None


@torch.no_grad()
def error_split(df: pd.DataFrame, rs: ResultsStore, img: ImageStore, img_size: int, dev) -> pd.DataFrame:
    """err_vis / err_inv = <e, P e> per slice, from saved predictions (Parseval frame: sum = ||e||^2)."""
    out = []
    for g, dg in df.groupby("geometry"):
        vop = VisibilityOperator(load_geometry(g), img_size, 16, device=dev)
        for run_id, d in dg.groupby("run_id"):
            tag = _tag(rs, run_id, g, d.split.iloc[0])
            if tag is None:
                continue
            for pid, idxs, preds in rs.iter_preds(tag):
                ref = np.stack([img.read(pid, "img256", int(i)) for i in idxs]).astype(np.float32)
                e = torch.from_numpy(np.clip(preds, 0, None) - ref)[:, None].to(dev)
                ev = (e * vop.vis(e)).flatten(1).sum(1).cpu().numpy()
                ei = (e * vop.inv(e)).flatten(1).sum(1).cpu().numpy()
                out += [{"run_id": run_id, "geometry": g, "pid": pid, "idx": int(i), "err_vis": float(a),
                         "err_inv": float(b)} for i, a, b in zip(idxs, ev, ei)]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs", help="the pipeline outputs directory (with eval/, runs/, tables/)")
    ap.add_argument("--images", default=None, help="images.h5 (default: from the config.yaml of an E3 run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep", type=int, default=2, help="illustrative slices per patient")
    ap.add_argument("--seeds", default="best_val", choices=list(POLICIES),
                    help="configurations with several seeds: best_val = seed with the best validation PSNR | first | "
                         "all")
    a = ap.parse_args()
    root, out = Path(a.outputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rs = ResultsStore(root / "eval")
    df = load_store(rs, a.split)
    if df.empty:
        raise SystemExit("No evaluation results yet")
    # one run per configuration (as in the pipeline tables/figures); "all" keeps every seed and uses seed 0 for images
    exps = sorted(set(df.exp) - {"baseline"})
    df = keep_chosen(df, seed_selection(root / "runs", exps, a.seeds), exps)
    base_cfg = load_config_file(next(iter(sorted((root / "runs/e3_main").glob("*/config.yaml")))))
    img = ImageStore(Path(a.images) if a.images else base_cfg.data.images_h5)
    es = error_split(df, rs, img, base_cfg.data.img_size_rec, dev)
    df = df.merge(es, on=["run_id", "geometry", "pid", "idx"], how="left")
    one = (df.exp == "baseline") | ((df.seed == 0) if a.seeds == "all" else True)    # runs used for the illustrations
    cols = ["exp", "method", "geometry", "seed", "pid", "idx", "run_id", *[c for c in KEEP if c in df],
            "err_vis", "err_inv"]
    df[cols].to_csv(out / "slices.csv", index=False)

    # ---- illustrations: per patient, slices around the median FBP PSNR (model-independent), seed 0
    images = {}
    for g, dg in df.groupby("geometry"):
        fbp = dg[dg.method == "fbp"]
        for pid, fp in fbp.groupby("pid"):
            order = fp.sort_values("psnr").idx.to_numpy()
            picks = [int(order[int(q * (len(order) - 1))]) for q in np.linspace(0.35, 0.65, a.keep)]
            for idx in picks:
                images[f"{g}/gt/{pid}/{idx}"] = img.read(pid, "img256", idx).astype(np.float32)
                for run_id, r in dg[one.loc[dg.index]].groupby("run_id"):
                    tag = _tag(rs, run_id, g, a.split)
                    p = rs.load_pred(tag, pid, idx) if tag else None
                    if p is not None:
                        images[f"{g}/{r.method.iloc[0]}/{pid}/{idx}"] = p
    np.savez_compressed(out / "images.npz", **images)

    # ---- per-stage parameters (seed 0) + meta
    params, meta = {}, {"mu_ref": img.mu_ref, "mu_water": img.mu_water, "split": a.split,
                        "img_size": base_cfg.data.img_size_rec, "runs": {}}
    for (g, m, run_id), _ in df[one & (df.exp != "baseline")].groupby(["geometry", "method", "run_id"]):
        exp = df[df.run_id == run_id].exp.iloc[0]
        run = RunStore(root / "runs" / exp / run_id)
        if not run.config_file.exists():
            continue
        model, _, _ = load_trained(run, "best" if run.checkpoint("best").exists() else "last", "cpu")
        params[f"{g}/{m}"] = {k: [float(x) for x in v.detach()] for k, v in model.stage_params().items()}
        meta["runs"][f"{g}/{m}"] = {"dir": str(run.dir), "exp": exp, "summary": run.summary()}
    (out / "params.json").write_text(json.dumps(params, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))

    # ---- cost + learning curves
    costs = [pd.read_csv(f) for f in sorted((root / "tables").glob("cost_*.csv"))]
    if costs:
        pd.concat(costs, ignore_index=True).to_csv(out / "costs.csv", index=False)
    for f in (root / "runs").glob("*/*/metrics.csv"):
        dst = out / "runs" / f.parent.parent.name / f.parent.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / "metrics.csv")
        if (f.parent / "summary.json").exists():
            shutil.copy2(f.parent / "summary.json", dst / "summary.json")
    b = root / "runs/e2_sweep/best_lr.json"
    if b.exists():
        (out / "runs/e2_sweep").mkdir(parents=True, exist_ok=True)
        shutil.copy2(b, out / "runs/e2_sweep/best_lr.json")
    print("DONE", out, len(df), "rows")


if __name__ == "__main__":
    main()
