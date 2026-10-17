"""In-depth analyses (step 32): error energy by frequency-coverage region (F7), effective receptive field (F9),
per-stage analysis (F8)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ...config import Config, load_config_file, load_geometry
from ...core.analysis import (
    coverage_map,
    effective_receptive_field,
    erf_radius,
    error_energy_by_region,
    region_masks,
    stagewise_metrics,
    summarize_stagewise,
)
from ...core.imaging.geometry import FanBeamGeometry
from ...core.models import UnrolledLEARN
from ...infra.io.hdf5_store import ImageStore
from ...infra.stores.results_store import ResultsStore
from ...infra.stores.run_store import list_runs
from ...viz import heatmaps, label_of, stagewise_plot, wedge_fraction_plot
from ..factories import dataset_for, image_store, load_trained, projector_for, results_store
from .seeds import chosen_run_ids, seed_selection
from .tables import ReportSpec, arch_rank

log = logging.getLogger("lact")
EXTS = ("pdf", "png")


@dataclass
class AnalysisResult:
    coverage: dict[str, np.ndarray] = field(default_factory=dict)             # geometry -> coverage map
    wedge_rows: list[dict] = field(default_factory=list)
    stagewise: dict[str, dict[str, pd.DataFrame]] = field(default_factory=dict)  # geometry -> reg -> summary
    erf: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)       # (reg, geometry) -> ERF
    erf_r90: dict[tuple[str, str], float] = field(default_factory=dict)


def _use_run(run_id: str, run_seed: int, seed: int, keep: set[str] | None) -> bool:
    """``keep`` (the runs chosen for reporting) when given; otherwise the runs of ``seed``."""
    return run_id in keep if keep is not None else int(run_seed) == seed


def wedge_errors(geom: FanBeamGeometry, masks: dict, spec: ReportSpec, seed: int, results: ResultsStore,
                 images: ImageStore, keep: set[str] | None = None) -> list[dict]:
    """Decompose the error energy by frequency region on the saved predictions."""
    rows = []
    for tag, info in results.find(split=spec.split, geometry_test=geom.name, geometry_train=geom.name).items():
        if info["exp"] not in (spec.main_exp, "baseline"):
            continue
        if info["exp"] != "baseline" and not _use_run(info["run_id"], info["seed"], seed, keep):
            continue
        for pid, idx, pred in results.iter_preds(tag):
            ref = images.read(pid, "img256")[idx]
            for i in range(len(idx)):
                rows.append({"geometry": geom.name, "regularizer": info["regularizer"], "pid": pid,
                             "idx": int(idx[i]), **error_energy_by_region(pred[i], ref[i], masks)})
    return rows


def model_probes(cfg: Config, geom: FanBeamGeometry, P, spec: ReportSpec, seed: int, n_erf: int, n_stage: int,
                 device, tables_dir, res: AnalysisResult, keep: set[str] | None = None) -> None:
    """ERF and per-stage analysis for the ``seed`` model of each architecture at geometry ``geom``."""
    ds_val = dataset_for(cfg, "val", geom.name, preload=False, augment=False)
    sel = np.unique(np.linspace(0, len(ds_val) - 1, min(n_erf, len(ds_val))).round().astype(int))
    imgs = torch.stack([ds_val[int(i)]["x0"] for i in sel])
    for run in list_runs(cfg.paths.runs, spec.main_exp):
        if run.status != "done":
            continue
        rc = load_config_file(run.config_file)
        if rc.require_geometry().name != geom.name or not _use_run(run.run_id, rc.seed, seed, keep):
            continue
        model, _, _ = load_trained(run, "best", device)
        if not isinstance(model, UnrolledLEARN):      # the ERF of one shared regularizer is meaningful only for LEARN
            continue
        model.set_projector(P)
        reg = rc.require_model().regularizer.name
        key = (reg, geom.name)
        res.erf[key] = effective_receptive_field(model.regularizer(0), imgs.clamp(0, 1), device)
        res.erf_r90[key] = erf_radius(res.erf[key])
        sw = stagewise_metrics(model, ds_val, device, n_slices=n_stage)
        sw.to_csv(tables_dir / f"stagewise_{reg}_{geom.name}.csv", index=False)
        res.stagewise.setdefault(geom.name, {})[reg] = summarize_stagewise(sw)
        log.info("%s|%s: ERF r90 = %.1f px", reg, geom.name, res.erf_r90[key])


def _write_outputs(cfg: Config, res: AnalysisResult) -> None:
    out_t, out_f = cfg.paths.tables, cfg.paths.figures
    if res.wedge_rows:
        w = pd.DataFrame(res.wedge_rows)
        w.to_csv(out_t / "wedge_error_by_region_slices.csv", index=False)
        s = w.groupby(["geometry", "regularizer"])[[c for c in w.columns if c.startswith("frac_")]].mean()
        s = s.reset_index()
        s.to_csv(out_t / "wedge_error_by_region.csv", index=False)
        for gname, d in s.groupby("geometry"):
            for ext in EXTS:
                wedge_fraction_plot(d, out_path=out_f / f"F7_wedge_{gname}.{ext}")
    if res.coverage:
        for ext in EXTS:
            heatmaps(res.coverage, titles={k: f"Độ phủ tần số {k}" for k in res.coverage}, vmax=1.0,
                     cbar_label="độ phủ", out_path=out_f / f"F7_coverage.{ext}")
    for gname, sums in res.stagewise.items():
        for ext in EXTS:
            stagewise_plot(sums, out_path=out_f / f"F8_stagewise_{gname}.{ext}")
    if not res.erf:
        return
    pd.DataFrame([{"regularizer": r, "geometry": g, "erf_r90_px": v} for (r, g), v in res.erf_r90.items()]
                 ).to_csv(out_t / "erf_radius.csv", index=False)
    crop = min(64, cfg.data.img_size_rec // 2)
    for gname in sorted({g for _, g in res.erf}):
        regs = sorted([r for r, g in res.erf if g == gname], key=arch_rank)
        maps = {r: res.erf[(r, gname)] for r in regs}
        for ext in EXTS:
            heatmaps(maps, titles={r: label_of(r) for r in maps}, radius={r: res.erf_r90[(r, gname)] for r in regs},
                     crop=crop, vmax=max(float(m.max()) for m in maps.values()), cbar_label="|∂R/∂x| (chuẩn hóa)",
                     out_path=out_f / f"F9_erf_{gname}.{ext}")


def run_analysis(cfg: Config, spec: ReportSpec, seed: int = 0, n_erf: int = 50, n_stage: int = 50,
                 device: torch.device | str = "cpu") -> AnalysisResult:
    device = torch.device(device)
    cfg.paths.tables.mkdir(parents=True, exist_ok=True)
    cfg.paths.figures.mkdir(parents=True, exist_ok=True)
    results, images = results_store(cfg), image_store(cfg)
    res = AnalysisResult()
    keep = None if spec.seed_policy == "all" else chosen_run_ids(
        seed_selection(cfg.paths.runs, spec.trained_exps, spec.seed_policy))
    for name in spec.geoms:
        geom = load_geometry(name)
        P = projector_for(cfg, geom, device=device)
        res.coverage[geom.name] = cov = coverage_map(P)
        res.wedge_rows += wedge_errors(geom, region_masks(cov), spec, seed, results, images, keep)
        model_probes(cfg, geom, P, spec, seed, n_erf, n_stage, device, cfg.paths.tables, res, keep)
    _write_outputs(cfg, res)
    log.info("In-depth analysis finished.")
    return res
