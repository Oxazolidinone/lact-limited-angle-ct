"""Check CG convergence of ``update = cg | admm`` runs.

Models are trained with a cap of ``cg_iters`` CG iterations per stage and a stopping threshold ``cg_tol``. Here the
best checkpoint is re-run on the validation split twice: with the training settings, and with CG solved almost exactly
(a cap ``factor`` times larger and a threshold 100 times smaller). The PSNR difference between the two is the error
due to stopping CG early: "within 0.01 dB" means the difference does not exceed 0.01 dB. Writes nothing to the
evaluation store.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from ...config import Config
from ...core.metrics import psnr
from ...core.models import UnrolledLEARN
from ...infra.data.dataset import collate
from ...infra.stores.run_store import list_runs
from ...infra.system.runtime import apply_precision
from ..factories import dataset_for, load_trained, projector_for

log = logging.getLogger("lact")


def _score(model, loader, device) -> tuple[np.ndarray, float]:
    """(per-slice PSNR, mean CG iterations per stage)."""
    vals, iters = [], []
    for b in loader:
        out = model(b["y"].to(device), b["x0"].to(device))
        vals.append(psnr(out, b["x"].to(device)).cpu())
        iters.append(model.cg_iters_mean())
    return torch.cat(vals).numpy(), float(np.mean(iters))


@torch.no_grad()
def cg_check(cfg: Config, exps: list[str], device: torch.device | str = "cpu", n_slices: int = 64,
             factor: int = 5, tol_db: float = 0.01, batch_size: int = 4) -> pd.DataFrame:
    """One row per DONE run with CG: validation PSNR with the training settings (``psnr``) and with CG solved almost
    exactly (``psnr_ref``), ``diff`` = psnr_ref - psnr, ``ok`` = |diff| <= ``tol_db``. ``n_slices`` evenly spaced."""
    device = torch.device(device)
    rows = []
    for exp in exps:
        for run in list_runs(cfg.paths.runs, exp):
            if run.status != "done" or not run.checkpoint("best").is_file():
                continue
            model, rcfg, _ = load_trained(run, "best", device)
            if not (isinstance(model, UnrolledLEARN) and model.uses_cg):
                continue
            geom = rcfg.require_geometry()
            apply_precision(rcfg.precision.tf32)
            model.set_projector(projector_for(rcfg, geom, device=device))
            ds = dataset_for(rcfg, "val", geom.name, augment=False, preload=False)
            idx = sorted(set(np.linspace(0, len(ds) - 1, min(n_slices, len(ds))).round().astype(int).tolist()))
            loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, collate_fn=collate)
            iters, tol = model.cg_iters, model.cg_tol
            base, used = _score(model, loader, device)
            model.cg_iters, model.cg_tol = iters * int(factor), tol / 100.0
            ref, used_ref = _score(model, loader, device)
            diff = float(ref.mean() - base.mean())
            m = rcfg.require_model()
            rows.append({"run_id": run.run_id, "exp": exp, "arch": m.arch, "geometry": geom.name, "update": m.update,
                         "metric": m.metric, "cg_iters": iters, "cg_tol": tol, "cg_used": used,
                         "cg_used_ref": used_ref, "n_slices": len(idx), "psnr": float(base.mean()),
                         "psnr_ref": float(ref.mean()), "diff": diff,
                         "max_abs_slice_diff": float(np.abs(ref - base).max()), "ok": abs(diff) <= tol_db})
            log.info("%s: CG %.1f iterations/stage (cap %d) -> %.3f dB; near-exact solve (%.1f iterations) -> %.3f dB; "
                     "difference %+.4f dB %s", run.run_id, used, iters, base.mean(), used_ref, ref.mean(), diff,
                     "OK" if rows[-1]["ok"] else f"> {tol_db} dB")
    return pd.DataFrame(rows)
