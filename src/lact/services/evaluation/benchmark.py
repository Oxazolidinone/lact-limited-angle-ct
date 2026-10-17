"""Compute-cost measurements (step 21).

pilot: run N training steps per architecture to measure seconds/step and memory and to estimate the grid's GPU hours.
runs:  reconstruction time (FBP + K stages, batch 1; median, P95), memory, parameters and FLOPs of trained runs,
       plus the training GPU hours from summary.json.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
import pandas as pd
import torch

from ...config import Config
from ...core.analysis import benchmark_fbp, benchmark_reconstruction, model_cost
from ...core.analysis.cost import time_data_loading
from ...core.models import count_params
from ...infra.stores.run_store import list_runs
from ...infra.system.runtime import apply_precision, collect_env, seed_everything
from ..factories import dataset_for, fbp_for, load_trained, model_for, projector_for
from ..training.optim import build_optimizer

log = logging.getLogger("lact")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def _time_steps(model, opt, ds, steps: int, device: torch.device) -> float:
    """Median seconds per step (warm-up steps excluded)."""
    times = []
    for s in range(steps):
        b = ds[s % len(ds)]
        y, x0, x = b["y"][None].to(device), b["x0"][None].to(device), b["x"][None].to(device)
        _sync(device)
        t0 = time.perf_counter()
        own_loss = getattr(model, "training_loss", None)           # as in the trainer: time the model's own loss
        loss = own_loss(y, x0, x) if own_loss is not None else torch.mean((model(y, x0) - x) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return float(np.median(times[min(10, len(times) // 4):]))


def pilot(cfgs: dict[str, Config], steps: int = 200, planned_runs: int | None = None,
          device: torch.device | str = "cpu") -> pd.DataFrame:
    """``cfgs``: {model name: full configuration (same geometry)}. Returns seconds/step and GPU hours per run.

    On GPU out-of-memory the model is measured again with gradient checkpointing, exactly as a real training run would
    do by itself (column ``grad_checkpoint``); if memory is still short, the row is NaN and the next model follows."""
    device = torch.device(device)
    rows = []
    first = next(iter(cfgs.values()))
    P = projector_for(first, device=device)
    ds = dataset_for(first, "train", limit=max(steps, 8), preload=True)
    n_train_full = len(dataset_for(first, "train", preload=False))
    for name, cfg in cfgs.items():
        t, peak, used_ckpt, cg_used = math.nan, math.nan, False, math.nan
        apply_precision(cfg.precision.tf32)        # measure in the numerical mode the real training run will use
        for ckpt in (False, True):
            seed_everything(0, False)
            model = model_for(cfg, P).to(device)
            model.grad_checkpoint = model.grad_checkpoint or ckpt
            opt = build_optimizer(model, cfg.train)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                t = _time_steps(model, opt, ds, steps, device)
                cg_used = model.cg_iters_mean()        # of the last measured step
                peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else math.nan
                used_ckpt = bool(model.grad_checkpoint)
                break
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                log.warning("%s: GPU out of memory%s", name,
                            "" if ckpt else "; measuring again with gradient checkpointing")
            finally:
                del model, opt
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        epochs = int(cfg.train.epochs)
        hours = t * n_train_full * epochs / 3600.0
        mc = cfg.require_model()
        rows.append({"model": name, "K": mc.K, "sec_per_step": t, "peak_mem_mb": peak,
                     "n_params": count_params(model_for(cfg, P), trainable_only=False), "n_train_slices": n_train_full,
                     "epochs": epochs, "gpu_hours_per_run": hours, "grad_checkpoint": used_ckpt,
                     # cg/admm: CG iterations per stage AT INITIALISATION (with a cg_tol stop this changes during
                     # training with lam_k; the cap is cg_iters)
                     "update": mc.update_tag, "metric": mc.metric if mc.uses_cg else "", "cg_iters_cap":
                     mc.cg_iters if mc.uses_cg else 0, "cg_iters_used": cg_used})
        log.info("%s: %.3f s/step, %.1f GPU-hours/run, peak %.1f GB%s%s", name, t, hours, peak / 1024,
                 " (gradient checkpointing)" if used_ckpt else "",
                 f", CG {cg_used:.1f}/{mc.cg_iters} iterations per stage" if mc.uses_cg else "")
    df = pd.DataFrame(rows)
    mean_run = df.gpu_hours_per_run.mean()
    if planned_runs:        # rough estimate; `lact train plan` computes it exactly from the grid's remaining runs
        log.info("Estimate for %d equivalent runs: %.0f GPU-hours (mean %.1f hours/run, %s)", planned_runs,
                 mean_run * planned_runs, mean_run, collect_env().get("gpu", "cpu"))
    return df


@torch.no_grad()
def cost_of_runs(cfg: Config, exps: list[str], seed: int = 0, warmup: int = 10, iters: int = 100,
                 device: torch.device | str = "cpu") -> pd.DataFrame:
    device = torch.device(device)
    rows = []
    gpu = collect_env().get("gpu", "cpu")
    for exp in exps:
        for run in list_runs(cfg.paths.runs, exp):
            if run.status != "done":
                continue
            model, rcfg, _ = load_trained(run, "best", device)
            if int(rcfg.seed) != seed:
                continue
            apply_precision(rcfg.precision.tf32)
            geom = rcfg.require_geometry()
            n_rec = rcfg.data.img_size_rec
            model.set_projector(projector_for(rcfg, geom, device=device))
            fbp = fbp_for(rcfg, geom, device)
            ds = dataset_for(rcfg, "val", preload=False, augment=False, limit=64)
            y = ds[0]["y"][None].to(device)
            bench = benchmark_reconstruction(model, fbp, y, warmup, iters, include_fbp=True)
            cost = model_cost(model, n_rec, "cpu")
            model.to(device)
            summ = run.summary()
            steps = pd.read_csv(run.steps_csv) if run.steps_csv.exists() else pd.DataFrame()
            m = rcfg.require_model()
            row = {"run_id": rcfg.run_id or run.run_id, "exp": exp, "arch": m.arch,
                   "regularizer": m.regularizer.name, "geometry_test": geom.name,
                   **bench, **cost, **benchmark_fbp(fbp, y), **time_data_loading(ds, 30),
                   "train_hours": summ.get("train_seconds", float("nan")) / 3600.0,
                   "peak_train_mem_mb": summ.get("peak_train_mem_mb"),
                   "train_gpus": ", ".join(summ.get("gpus", [])) or None,
                   "train_grad_checkpoint": summ.get("grad_checkpoint"),
                   "train_jobs_per_gpu": summ.get("jobs_per_gpu", 1),
                   "train_sec_per_step": float(steps.sec_per_step.median()) if "sec_per_step" in steps else math.nan,
                   "gpu": gpu}
            rows.append(row)
            log.info("%s: %.1f ms (P95 %.1f), %.2f GFLOPs x K", row["run_id"], row["median_ms"], row["p95_ms"],
                     row["gflops_regularizer_total"])
    return pd.DataFrame(rows)
