"""Cross-check trained runs against the data on the current machine.

When runs move to another machine, the matrix A and the sinograms are rebuilt there. Re-evaluating the ``best``
checkpoint on the validation split must reproduce the ``best_val_psnr`` recorded during training; a larger deviation
means the images, sinograms or matrix on this machine DIFFER from training, and old and new runs must not be mixed.
Writes nothing to outputs.
"""
from __future__ import annotations

import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ...config import Config
from ...core.metrics import psnr
from ...core.models import val_forward
from ...infra.data.dataset import collate
from ...infra.stores.run_store import RunStore, list_runs
from ...infra.system.runtime import apply_precision
from ..factories import dataset_for, load_trained, projector_for

log = logging.getLogger("lact")
MAIN_EXPS = ("base_mayo", "base_dl", "lbfgs_mayo", "lbfgs_dl")


@torch.no_grad()
def val_psnr(run: RunStore, device: torch.device | str = "cpu", data_cfg: Config | None = None) -> float:
    """Mean per-slice validation PSNR of the ``best`` checkpoint, computed as during training.

    ``data_cfg``: take images, sinograms, matrix and validation split from this configuration instead of the run's
    (e.g. a Mayo-trained run checked against the joint store, which has the same validation patients)."""
    device = torch.device(device)
    model, cfg, _ = load_trained(run, "best", device)
    apply_precision(cfg.precision.tf32)              # as when the run was trained (GPUs without TF32 still use FP32)
    dcfg = data_cfg or cfg
    model.set_projector(projector_for(dcfg, cfg.require_geometry(), device=device))
    ds = dataset_for(dcfg, "val", cfg.require_geometry().name, limit=cfg.train.limit_val, augment=False,
                     preload=False)
    dl = DataLoader(ds, batch_size=int(cfg.train.val_batch_size), shuffle=False, collate_fn=collate, num_workers=0)
    infer = val_forward(model)          # as in validation during training (see trainer.val_forward)
    vals = [psnr(infer(b["y"].to(device), b["x0"].to(device)), b["x"].to(device)).cpu() for b in dl]
    return float(torch.cat(vals).mean())


def pick_runs(cfg: Config, exps: tuple[str, ...] | list[str] = MAIN_EXPS, per_geometry: int = 2) -> list[RunStore]:
    """Per geometry, up to ``per_geometry`` DONE runs of different architectures (smaller seeds first)."""
    by_geom: dict[str, dict[str, RunStore]] = {}
    runs = [r for exp in exps for r in list_runs(cfg.paths.runs, exp)]
    for run in sorted(runs, key=lambda r: (r.summary().get("seed", 0), r.run_id)):
        s = run.summary()
        if run.status != "done" or s.get("best_val_psnr") is None or not run.checkpoint("best").is_file():
            continue
        archs = by_geom.setdefault(str(s.get("geometry")), {})
        if len(archs) < per_geometry:
            archs.setdefault(str(s.get("arch")), run)
    return [r for archs in by_geom.values() for r in archs.values()]


def recheck(cfg: Config, exps: tuple[str, ...] | list[str] = MAIN_EXPS, per_geometry: int = 2,
            tol_db: float = 0.02, device: torch.device | str = "cpu", on_config_data: bool = False) -> pd.DataFrame:
    """Table: run_id, geometry, arch, recorded, now, diff, ok (|diff| <= ``tol_db``). Empty when there are no runs.

    ``on_config_data``: check on the data and validation split of ``cfg`` instead of each run's own."""
    rows = []
    for run in pick_runs(cfg, exps, per_geometry):
        s = run.summary()
        now = val_psnr(run, device, cfg if on_config_data else None)
        diff = now - float(s["best_val_psnr"])
        rows.append({"run_id": run.run_id, "geometry": s.get("geometry"), "arch": s.get("arch"),
                     "recorded": float(s["best_val_psnr"]), "now": now, "diff": diff, "ok": abs(diff) <= tol_db})
        log.info("%s: at training %.4f dB, now %.4f dB (difference %+.4f) %s", run.run_id, s["best_val_psnr"], now,
                 diff, "OK" if rows[-1]["ok"] else "MISMATCH")
    return pd.DataFrame(rows)
