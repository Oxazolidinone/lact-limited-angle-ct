"""Queue order and remaining-work estimates of a grid, from the measured seconds per step of each regularizer."""
from __future__ import annotations

import logging
from pathlib import Path

from ...config import load_config
from ...infra.stores.run_store import RunStore, list_runs
from ..training.trainer import run_store_for
from .jobs import Job

log = logging.getLogger("lact")


def sweep_sec_per_step(runs_root: Path, sweep_exp: str = "e2_sweep") -> dict[str, float]:
    """Median seconds per step per regularizer, measured from finished lr-sweep runs (to queue long runs first)."""
    per: dict[str, list[float]] = {}
    for run in list_runs(runs_root, sweep_exp):
        s = run.summary()
        if s.get("steps") and s.get("train_seconds") and s.get("regularizer"):
            per.setdefault(s["regularizer"], []).append(float(s["train_seconds"]) / float(s["steps"]))
    return {k: float(sorted(v)[len(v) // 2]) for k, v in per.items()}


def order_jobs(jobs: list[Job], stores: dict[str, RunStore], sec_per_step: dict[str, float] | None = None) -> list[Job]:
    """Sort runs into the order of the parallel-run queue.

    1. Partial runs first, so results complete early and partial checkpoints are released.
    2. Then the runs with the longest ESTIMATED duration first (LPT): long runs started early mean less waiting at the
       end of the queue. Estimate = the regularizer's seconds/step (measured in E2) x the remaining steps. Without a
       measurement the grid order is kept (E3 -> E4 -> E5), with a stable sort.
    """
    sec = sec_per_step or {}
    known = sorted(sec.values())
    fallback = known[len(known) // 2] if known else 0.0

    beats = {j.run_id: stores[j.run_id].heartbeat() for j in jobs}
    # Every run uses the same training split, so steps per epoch are equal; taken from a partial run when available.
    spe = max((int(b.get("steps_per_epoch") or 0) for b in beats.values()), default=0) or 1

    def remaining_hours(job: Job) -> float:
        cfg = load_config(job.files, job.overrides, use_defaults=False)
        reg = cfg.require_model().regularizer.name
        beat = beats[job.run_id]
        done = int(beat.get("step") or 0) if stores[job.run_id].status == "partial" else 0
        total = int(cfg.train.epochs) * int(beat.get("steps_per_epoch") or spe)
        return sec.get(reg, fallback) * max(0, total - done) / 3600.0

    est = {j.run_id: remaining_hours(j) for j in jobs} if sec else {}
    return sorted(jobs, key=lambda j: (stores[j.run_id].status != "partial", -est.get(j.run_id, 0.0)))


def remaining_work(jobs: list[Job], sec_per_step: dict[str, float], n_train: int) -> list[dict]:
    """Estimated GPU hours still to train for each unfinished run, from per-architecture seconds/step
    (``lact train pilot``).

    ``sec_per_step``: {model.arch: seconds/step when running alone on a GPU}; unmeasured architectures take the median
    of the measured ones (column ``measured`` = False). Partial runs count only their remaining steps."""
    known = sorted(sec_per_step.values())
    fallback = known[len(known) // 2] if known else float("nan")
    rows = []
    for job in jobs:
        if not job.pending:
            continue
        cfg = load_config(job.files, job.overrides, use_defaults=False)
        total = int(cfg.train.epochs) * -(-int(n_train) // int(cfg.train.batch_size))
        if cfg.train.max_steps:
            total = min(total, int(cfg.train.max_steps))
        done = int(run_store_for(cfg).heartbeat().get("step") or 0) if job.status == "partial" else 0
        sec = sec_per_step.get(job.arch, fallback)
        rows.append({"run_id": job.run_id, "exp": cfg.exp, "arch": job.arch, "steps_left": max(0, total - done),
                     "gpu_hours": sec * max(0, total - done) / 3600.0, "measured": job.arch in sec_per_step})
    return rows
