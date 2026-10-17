"""Result figures F1-F6, F10 (PDF + PNG) from the evaluation store, saved predictions and training logs.

Illustrative slices and ROIs are chosen by fixed, model-independent rules: the slice whose FBP PSNR is the median
within each patient, and the ROI where the local FBP error is largest.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Config, load_config_file
from ...infra.io.hdf5_store import ImageStore
from ...infra.stores.results_store import ResultsStore, make_tag
from ...infra.stores.run_store import list_runs
from ...viz import (
    color_of,
    interaction_plot,
    label_of,
    line_profile,
    metric_by_arch,
    pareto_plot,
    per_slice_plot,
    recon_grid,
    roi_by_fbp_error,
    training_curves,
)
from ...viz.style import ARCH_ORDER
from ..factories import image_store, results_store
from .aggregate import per_method
from .seeds import chosen_run_ids
from .tables import ReportSpec, load_costs, load_selected

log = logging.getLogger("lact")
EXTS = ("pdf", "png")


def save_both(fig_fn: Callable, out: Path, name: str, **kw) -> None:
    for ext in EXTS:
        fig_fn(out_path=out / f"{name}.{ext}", **kw)


def fig_metrics(df: pd.DataFrame, pm: pd.DataFrame, spec: ReportSpec, out: Path) -> None:
    """F1 (metrics per architecture) and F2 (PSNR versus scan range)."""
    save_both(metric_by_arch, out, "F1_psnr_by_arch", method_df=pm, metric="psnr", geometries=spec.geoms)
    save_both(metric_by_arch, out, "F1_ssim_by_arch", method_df=pm, metric="ssim", geometries=spec.geoms,
              ylabel="SSIM")
    geoms_all = [g for g in ["fan_a150", "fan_a120", "fan_a90"] if g in set(df.geometry_test)]
    pm_all = per_method(df[df.exp.isin([spec.main_exp, spec.a150_exp, "baseline"]) &
                           df.geometry_test.isin(geoms_all)])
    save_both(interaction_plot, out, "F2_psnr_vs_angle", method_df=pm_all, metric="psnr", order_geoms=geoms_all)


def fig_pareto(cost: pd.DataFrame, pm: pd.DataFrame, spec: ReportSpec, out: Path) -> None:
    """F3: quality versus cost (time, parameters, memory)."""
    if cost.empty:
        return
    for g in spec.geoms:
        m = cost[cost.geometry_test == g].drop_duplicates("arch").merge(
            pm[pm.geometry_test == g][["arch", "psnr_mean", "psnr_std"]], on="arch")
        if m.empty:
            continue
        m = m.assign(n_params_m=m.n_params / 1e6)
        save_both(pareto_plot, out, f"F3_pareto_time_{g}", df=m, x="median_ms",
                  xlabel="Thời gian tái tạo trung vị (ms/lát)")
        save_both(pareto_plot, out, f"F3_pareto_params_{g}", df=m, x="n_params_m", xlabel="Số tham số (M)", logx=True)
        if "peak_infer_mem_mb" in m and m.peak_infer_mem_mb.notna().any():
            save_both(pareto_plot, out, f"F3_pareto_mem_{g}", df=m, x="peak_infer_mem_mb",
                      xlabel="Bộ nhớ GPU đỉnh khi suy luận (MB)")


def _median_fbp_slice(fbp_rows: pd.DataFrame) -> int:
    return int(fbp_rows.iloc[(fbp_rows.psnr - fbp_rows.psnr.median()).abs().argsort().iloc[0]].idx)


def fig_qualitative(dmain: pd.DataFrame, spec: ReportSpec, seed: int, images: ImageStore, results: ResultsStore,
                    out: Path) -> None:
    """F5 (image grid + ROI) and F6 (line profile) for each patient's median-FBP slice."""
    for g in spec.geoms:
        dg = dmain[dmain.geometry_test == g]
        fbp_rows = dg[dg.regularizer == "fbp"]
        if fbp_rows.empty:
            log.warning("No FBP results for %s; skipping the qualitative figure", g)
            continue
        for pid, fp in fbp_rows.groupby("pid"):
            idx = _median_fbp_slice(fp)
            ref = images.read(pid, "img256", idx).astype(np.float32)
            imgs, mets, cols = {}, {}, {}
            for reg in ARCH_ORDER:
                rr = dg[dg.regularizer == reg]
                if rr.seed.nunique() > 1:             # several seeds left (seed_policy = all): take the requested one
                    rr = rr[rr.seed == seed]
                if rr.empty:
                    continue
                tag = make_tag({"run_id": rr.run_id.iloc[0], "split": spec.split, "geometry_test": g})
                pred = results.load_pred(tag, pid, idx)
                if pred is None:
                    continue
                lab = label_of(reg)
                imgs[lab], cols[lab] = pred, color_of(reg)
                row = rr[(rr.pid == pid) & (rr.idx == idx)].iloc[0]
                mets[lab] = (row.psnr, row.ssim)
            if "FBP" not in imgs:
                continue
            roi = roi_by_fbp_error(imgs["FBP"], ref, size=max(16, ref.shape[0] // 6))
            for ext in EXTS:
                recon_grid(imgs, ref, images.mu_ref, images.mu_water, "soft", roi, metrics=mets,
                           title=f"{pid}, lát {idx}, {g}", out_path=out / f"F5_qual_{g}_{pid}.{ext}")
            r0, c0, h, w = roi
            learned = {k: v for k, v in imgs.items() if k != "FBP"}
            for ext in EXTS:
                line_profile(learned, ref, images.mu_ref, images.mu_water, r0 + h // 2,
                             (max(0, c0 - w), min(ref.shape[1], c0 + 2 * w)), cols,
                             out_path=out / f"F6_profile_{g}_{pid}.{ext}", title=f"{pid}, hàng {r0 + h // 2}")


def training_curves_by_geometry(cfg: Config, exp: str, keep: set[str] | None = None
                                ) -> dict[str, dict[str, list[pd.DataFrame]]]:
    """{geometry: {regularizer: [metrics.csv per seed]}}. ``keep``: only runs whose run_id is in this set."""
    curves: dict[str, dict[str, list[pd.DataFrame]]] = {}
    for run in list_runs(cfg.paths.runs, exp):
        if not (run.metrics_csv.exists() and run.config_file.exists()):
            continue
        if keep is not None and run.status == "done" and run.run_id not in keep:
            continue
        rc = load_config_file(run.config_file)
        curves.setdefault(rc.require_geometry().name, {}).setdefault(rc.require_model().regularizer.name, []).append(
            pd.read_csv(run.metrics_csv))
    return curves


def make_figures(cfg: Config, spec: ReportSpec, seed: int = 0) -> Path:
    out = cfg.paths.figures
    out.mkdir(parents=True, exist_ok=True)
    results = results_store(cfg)
    df, selection = load_selected(cfg, spec)
    if df.empty:
        raise FileNotFoundError("No evaluation results yet: run `lact eval runs` / `lact eval baselines`")
    keep = None if spec.seed_policy == "all" else chosen_run_ids(selection)
    df = df[df.geometry_train == df.geometry_test]
    dmain = df[df.exp.isin([spec.main_exp, "baseline"]) & df.geometry_test.isin(spec.geoms)]
    if dmain.empty:
        raise FileNotFoundError("No evaluation results for the main experiment yet")
    pm = per_method(dmain)
    fig_metrics(df, pm, spec, out)
    fig_pareto(load_costs(cfg), pm, spec, out)
    for g in spec.geoms:
        save_both(per_slice_plot, out, f"F4_per_slice_{g}", df=dmain[dmain.geometry_test == g])
    fig_qualitative(dmain, spec, seed, image_store(cfg), results, out)
    for g, curves in training_curves_by_geometry(cfg, spec.main_exp, keep).items():
        if g in spec.geoms:
            save_both(training_curves, out, f"F10_training_{g}", curves=curves, title=g)
    log.info("Figures written to %s", out)
    return out
