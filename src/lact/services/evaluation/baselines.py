"""Non-learned baselines: FBP (the networks' x^0) and TV-PDHG; TV tuning on the validation split (step 22)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...config import Config
from ...core.imaging.geometry import FanBeamGeometry
from ...core.metrics import psnr
from ...core.operators import SparseFanBeam
from ...core.solvers import tv_pdhg
from ...infra.io.files import load_json, save_json
from ..factories import dataset_for, projector_for

log = logging.getLogger("lact")

Method = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]      # (raw y, FBP x0) -> x_hat
MethodFactory = Callable[[SparseFanBeam], Method]
BASELINES = ("fbp", "tv")


def fbp_method(_: SparseFanBeam) -> Method:
    return lambda y, x0: x0


def tv_method(lam: float, iters: int) -> MethodFactory:
    return lambda P: (lambda y, x0: tv_pdhg(P, y, lam, iters, x_init=x0))


def tv_best_file(cfg: Config, geom_name: str) -> Path:
    return cfg.paths.baselines / f"tv_best_{geom_name}.json"


def baseline_method(cfg: Config, name: str, geom: FanBeamGeometry) -> tuple[MethodFactory, dict]:
    """(factory, parameters to record with the results) for baseline ``name``."""
    if name == "fbp":
        return fbp_method, {"filter": cfg.baseline.fbp.filter}
    if name == "tv":
        f = tv_best_file(cfg, geom.name)
        if not f.exists():
            raise FileNotFoundError(f"TV is not tuned for {geom.name} yet: run `lact eval tune-tv`")
        params = load_json(f)
        return tv_method(float(params["lambda"]), int(params["iters"])), params
    raise ValueError(f"Unsupported baseline: {name} (choose from {BASELINES})")


def _val_indices(n: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))


@torch.no_grad()
def tune_tv(cfg: Config, geom: FanBeamGeometry, device="cpu") -> dict:
    """Sweep (lambda, iterations) on fixed validation slices; write tv_grid_<geom>.csv and tv_best_<geom>.json."""
    tv = cfg.baseline.tv
    out = cfg.paths.baselines
    out.mkdir(parents=True, exist_ok=True)
    ds = dataset_for(cfg, "val", geom.name, preload=False, augment=False)
    idx = _val_indices(len(ds), int(tv.n_val_slices))
    P = projector_for(cfg, geom, device=device)
    iters = sorted(int(i) for i in tv.iters)
    rows = []
    for lam in tv.lambdas:
        acc: dict[int, list[torch.Tensor]] = {it: [] for it in iters}
        for s in range(0, len(idx), int(tv.batch_size)):
            items = [ds[int(i)] for i in idx[s: s + int(tv.batch_size)]]
            y = torch.stack([b["y"] for b in items]).to(device)
            x0 = torch.stack([b["x0"] for b in items]).to(device)
            x = torch.stack([b["x"] for b in items]).to(device)
            _, rec = tv_pdhg(P, y, float(lam), iters[-1], x_init=x0 if tv.init == "fbp" else None, record_at=iters)
            for it in iters:
                acc[it].append(psnr(rec[it], x).cpu())
        for it in iters:
            rows.append({"geometry": geom.name, "lambda": float(lam), "iters": it,
                         "val_psnr": float(torch.cat(acc[it]).mean())})
        log.info("%s lambda=%g: %s", geom.name, lam,
                 ", ".join(f"{r['iters']}:{r['val_psnr']:.2f}" for r in rows[-len(iters):]))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"tv_grid_{geom.name}.csv", index=False)
    best = df.sort_values("val_psnr", ascending=False).iloc[0]
    lams = sorted(float(v) for v in tv.lambdas)
    at_edge = best["lambda"] in (lams[0], lams[-1])
    note = "best lambda is on the grid boundary; consider widening the grid" if at_edge else ""
    result = {"geometry": geom.name, "lambda": float(best["lambda"]), "iters": int(best["iters"]),
              "val_psnr": float(best["val_psnr"]), "n_val_slices": len(idx), "note": note}
    save_json(result, tv_best_file(cfg, geom.name))
    log.info("==> %s: lambda=%g, iters=%d, validation PSNR %.2f dB %s", geom.name, result["lambda"], result["iters"],
             result["val_psnr"], note)
    return result
