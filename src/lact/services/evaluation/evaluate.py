"""Evaluation use cases: training runs, baselines, cross-geometry inference (E6); access control for the test split.

Results are written through ``ResultsStore`` (per-slice CSV + predictions + index.json).
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ...config import Config, ExperimentSpec, load_config_file, load_geometry
from ...core.imaging.geometry import FanBeamGeometry
from ...core.metrics import all_metrics, body_mask
from ...core.operators import SparseFanBeam
from ...infra.data.dataset import LACTDataset, collate
from ...infra.stores.results_store import ResultsStore, make_tag
from ...infra.stores.run_store import RunStore, list_runs
from ...infra.system.runtime import apply_precision, select_device
from ..factories import dataset_for, load_trained, projector_for
from .baselines import Method, baseline_method

log = logging.getLogger("lact")

LOCK_FILE = "configs_frozen.lock"


# ---------------------------------------------------------------------------- test-split lock
def lock_file(cfg: Config) -> Path:
    return cfg.paths.outputs / LOCK_FILE


def check_test_access(cfg: Config, split: str, force: bool = False) -> None:
    """The test split opens only once the configuration is frozen (lock file present) or with ``force``; every
    opening is logged."""
    if split != "test":
        return
    lock = lock_file(cfg)
    if not lock.exists() and not force:
        raise PermissionError(f"Configuration not frozen ({lock} does not exist). Run `lact eval freeze`, "
                              "or use --force-test if you are sure.")
    ResultsStore(cfg.paths.eval).log_test_access(forced=force and not lock.exists())


def freeze_configs(cfg: Config, note: str = "") -> Path:
    lock = lock_file(cfg)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"frozen_at": _dt.datetime.now().isoformat(timespec="seconds"), "note": note}),
                    encoding="utf-8")
    return lock


def _device(device) -> torch.device:
    return select_device(device) if isinstance(device, str) else device


# ---------------------------------------------------------------------------- core
@torch.no_grad()
def evaluate_method(method: Method, dataset: LACTDataset, projector: SparseFanBeam, device, info: dict,
                    batch_size: int = 4, keep_preds: bool = True, use_body_mask: bool = True
                    ) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]] | None]:
    """Run ``method(y, x0) -> x_hat`` over the dataset. Returns (per-slice metric table, predictions or None)."""
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    rows = []
    preds: dict[str, dict[int, np.ndarray]] = {}
    for b in dl:
        y, x0, x = b["y"].to(device), b["x0"].to(device), b["x"].to(device)
        out = method(y, x0)
        mask = body_mask(x, dataset.mu_ref, dataset.mu_water) if use_body_mask else None
        m = all_metrics(out, x, dataset.mu_ref, dataset.mu_water, mask=mask)
        dc = projector.data_residual(out, y)
        for i in range(out.shape[0]):
            pid, idx = b["pid"][i], int(b["idx"][i])
            rows.append({**info, "pid": pid, "idx": idx, "z": float(b["z"][i]), "dc_rel": float(dc[i]),
                         **{k: float(v[i]) for k, v in m.items()}})
            if keep_preds:
                preds.setdefault(pid, {})[idx] = out[i, 0].float().cpu().numpy()
    return pd.DataFrame(rows), (preds if keep_preds else None)


def _run_info(cfg: Config, run: RunStore, g_test: FanBeamGeometry, split: str, which: str) -> dict:
    m = cfg.require_model()
    return {"run_id": cfg.run_id or run.run_id, "exp": cfg.exp, "arch": m.arch, "regularizer": m.regularizer.name,
            "geometry_train": cfg.require_geometry().name, "geometry_test": g_test.name, "seed": int(cfg.seed),
            "lr": float(cfg.train.lr), "split": split, "checkpoint": which,
            "status": run.summary().get("status", "unknown"), "n_params": cfg.derived.get("n_params"),
            # update rule and split used for TRAINING (tells which dataset the model learned from)
            "update": m.update_tag, "metric": m.metric if m.uses_cg else "", "train_split": cfg.split.name,
            # source run whose checkpoint this run loaded at start (run_id), empty when trained from scratch
            "init_from": run.summary().get("init_from") or ""}


# ---------------------------------------------------------------------------- training runs
def evaluate_run(run: RunStore, split: str = "test", which: str = "best", device="auto",
                 test_geometry: FanBeamGeometry | None = None, save_pred: bool = True, batch_size: int = 4,
                 limit: int | None = None, force_test: bool = False, overwrite: bool = False,
                 results: ResultsStore | None = None, data_cfg: Config | None = None) -> pd.DataFrame | None:
    """Evaluate a run's checkpoint. ``test_geometry`` != None => cross-geometry inference (E6).

    ``data_cfg`` != None => cross-DATASET inference: model and geometry come from the run, while images, sinograms,
    split, test-split lock and results store come from ``data_cfg`` (e.g. a Mayo-trained run on the joint
    Mayo + DeepLesion store). Both must share the reconstruction grid; the caller guarantees the intensity scale (see
    configs/data/joint.yaml).

    Returns None when the result exists and ``overwrite`` is not set.
    """
    device = _device(device)
    cfg = load_config_file(run.config_file)
    dcfg = data_cfg or cfg
    if dcfg.data.img_size_rec != cfg.data.img_size_rec:
        raise ValueError(f"{run.run_id}: reconstruction grid {cfg.data.img_size_rec} differs from the evaluation data "
                         f"{dcfg.data.img_size_rec}")
    g_test = test_geometry or cfg.require_geometry()
    which = which if run.checkpoint(which).exists() else "last"
    info = _run_info(cfg, run, g_test, split, which)
    results = results or ResultsStore(dcfg.paths.eval)
    if results.exists(info, require_pred=save_pred) and not overwrite:
        log.info("Already have %s", make_tag(info))
        return None
    check_test_access(dcfg, split, force_test)
    # numerical mode from the EVALUATION configuration (not each run's): every model in a table is computed the same way
    apply_precision(dcfg.precision.tf32)
    model, _, state = load_trained(run, which, device)
    P = projector_for(dcfg, g_test, device=device)
    model.set_projector(P).eval()
    info["epoch"] = int(state.get("epoch", -1))
    ds = dataset_for(dcfg, split, g_test.name, preload=False, limit=limit, augment=False)
    df, preds = evaluate_method(lambda y, x0: model(y, x0), ds, P, device, info, batch_size, save_pred)
    results.save(info, df, preds)
    log.info("%s [%s, %s]: PSNR %.3f | SSIM %.4f | RMSE %.4f (%d slices)", info["run_id"], split, g_test.name,
             df.psnr.mean(), df.ssim.mean(), df.rmse.mean(), len(df))
    return df


def evaluate_experiments(cfg: Config, exps: Sequence[str] = (), run_dirs: Sequence[str | Path] = (),
                         split: str = "val", which: str = "best", device="auto", **kw) -> list[pd.DataFrame]:
    """Evaluate every finished run (DONE/DIVERGED) of the experiments ``exps`` and/or the directories ``run_dirs``."""
    runs = [RunStore(p) for p in run_dirs]
    for exp in exps:
        runs += list_runs(cfg.paths.runs, exp)
    out = []
    for run in runs:
        if not run.finished:
            log.warning("Skipping %s: not finished", run.run_id)
            continue
        df = evaluate_run(run, split, which, device, **kw)
        if df is not None:
            out.append(df)
    return out


def cross_geometry(cfg: Config, spec_file: str, split: str = "test", device="auto", **kw) -> list[pd.DataFrame]:
    """E6: models trained at geometry g1 run on data of g2 != g1 (no retraining)."""
    spec = ExperimentSpec.load(spec_file)
    if not spec.source_exp:
        raise ValueError(f"{spec_file}: missing source_exp")
    test_geoms = [load_geometry(g) for g in spec.test_geometries]
    out = []
    for run in list_runs(cfg.paths.runs, spec.source_exp):
        if run.status != "done":
            continue
        g_train = load_config_file(run.config_file).require_geometry()
        if g_train.name not in spec.train_geometries:
            continue
        for g in test_geoms:
            if g.name == g_train.name:
                continue  # same geometry = the E3 result
            df = evaluate_run(run, split, "best", device, test_geometry=g, **kw)
            if df is not None:
                out.append(df)
    return out


# ---------------------------------------------------------------------------- baseline
def evaluate_baseline(cfg: Config, name: str, geom: FanBeamGeometry, split: str = "val", device="auto",
                      save_pred: bool = True, batch_size: int = 4, limit: int | None = None,
                      force_test: bool = False, overwrite: bool = False) -> pd.DataFrame | None:
    """Evaluate a baseline with the same data, metrics and result format as the learned models."""
    device = _device(device)
    info = {"run_id": f"{name}_{geom.name}", "exp": "baseline", "arch": name, "regularizer": name,
            "geometry_train": geom.name, "geometry_test": geom.name, "seed": 0, "lr": float("nan"),
            "split": split, "epoch": -1, "status": "done", "n_params": 0}
    results = ResultsStore(cfg.paths.eval)
    if results.exists(info, require_pred=save_pred) and not overwrite:
        log.info("Already have %s", make_tag(info))
        return None
    check_test_access(cfg, split, force_test)
    apply_precision(cfg.precision.tf32)
    factory, params = baseline_method(cfg, name, geom)
    info["checkpoint"] = json.dumps(params)
    P = projector_for(cfg, geom, device=device)
    ds = dataset_for(cfg, split, geom.name, preload=False, limit=limit, augment=False)
    df, preds = evaluate_method(factory(P), ds, P, device, info, batch_size, save_pred)
    results.save(info, df, preds)
    log.info("%s %s [%s]: PSNR %.3f | SSIM %.4f", name, geom.name, split, df.psnr.mean(), df.ssim.mean())
    return df


def evaluate_baselines(cfg: Config, names: list[str], geoms: list[FanBeamGeometry], split: str = "val",
                       device="auto", **kw) -> list[pd.DataFrame]:
    out = []
    for g in geoms:
        for name in names:
            df = evaluate_baseline(cfg, name, g, split, device, **kw)
            if df is not None:
                out.append(df)
    return out
