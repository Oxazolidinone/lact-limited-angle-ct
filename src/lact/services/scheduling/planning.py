"""Queue order of a grid: unfinished runs first, then the longest estimated runs first."""
from __future__ import annotations

import logging

from ...config import load_config
from ...infra.stores.run_store import RunStore
from .jobs import Job

log = logging.getLogger("lact")


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
