"""Run training jobs: one child process ``python -m lact train run ...`` per job, sequentially or spread over GPUs.

A child process per run isolates GPU memory and keeps the command list portable (``lact train grid ... --emit``).
"""
from __future__ import annotations

import logging
import signal
import subprocess
import time

from ...config import ConfigError, load_config
from ...config.loader import REPO_ROOT
from ...infra.system.processes import ProcessRunner
from ..training.trainer import EXIT_INTERRUPTED, find_init_run, run_store_for
from .jobs import Job
from .planning import order_jobs

log = logging.getLogger("lact")


def gpu_shares(jobs: list[Job], job_mem_mb: dict[str, float] | None, gpu_mem_mb: float | None, per_gpu: int,
               budget: float = 0.85) -> dict[str, float]:
    """{run_id: GPU share} = peak memory of the architecture / (``budget`` x GPU memory); unmeasured architectures => 1
    (run alone). Empty when not running several runs per GPU or without measurements."""
    if not (per_gpu > 1 and job_mem_mb and gpu_mem_mb):
        return {}
    cap = budget * float(gpu_mem_mb)
    return {j.run_id: min(1.0, job_mem_mb[j.arch] / cap) if j.arch in job_mem_mb else 1.0 for j in jobs}


def run_jobs(jobs: list[Job], max_hours: float | None = None, gpus: list[str] | None = None,
             per_gpu: int = 1, job_mem_mb: dict[str, float] | None = None, gpu_mem_mb: float | None = None,
             sec_per_step: dict[str, float] | None = None) -> list[int]:
    """Run the unfinished runs (todo/partial). DIVERGED/OOM runs are not restarted automatically.

    - Without ``gpus``: sequentially on the configured device (as before).
    - With ``gpus`` (UUIDs): in parallel, one GPU per run, at most ``per_gpu`` runs per GPU. Runs with
      ``train.init_from`` wait until their source run is DONE, so a source grid and a dependent grid can share one
      command.
    - ``job_mem_mb`` {model.arch: peak memory}, ``gpu_mem_mb``: memory-aware packing (runs needing more than half a GPU
      run alone). ``sec_per_step`` {model.arch: seconds/step}: the longest runs are queued first, so each long run gets
      its own GPU before short runs fill in. All three come from ``--mem-table``.
    - A run interrupted mid-way (exit code 3, checkpoint saved) => the whole grid stops; run the command again to
    resume.
    - ``max_hours``: start no new runs after that many hours (running runs continue).
    """
    if gpus:
        return _run_jobs_parallel(jobs, max_hours, gpus, per_gpu, job_mem_mb, gpu_mem_mb, sec_per_step)
    codes = []
    t0 = time.time()
    for j in jobs:
        if not j.pending:
            continue
        if max_hours is not None and (time.time() - t0) / 3600 >= max_hours:
            log.warning("Ran %.1f hours >= --max-hours: stopping before %s", (time.time() - t0) / 3600, j.run_id)
            codes.append(EXIT_INTERRUPTED)
            break
        log.info("==> %s", j.run_id)
        code = _run_child(j.argv)
        codes.append(code)
        if code in (EXIT_INTERRUPTED, -2, 130):   # stopped (SIGINT/SIGTERM): do not continue with other runs
            log.warning("%s was interrupted (checkpoint saved). Run the command again to continue.", j.run_id)
            break
    return codes


def _run_jobs_parallel(jobs: list[Job], max_hours: float | None, gpus: list[str], per_gpu: int,
                       job_mem_mb: dict[str, float] | None = None, gpu_mem_mb: float | None = None,
                       sec_per_step: dict[str, float] | None = None) -> list[int]:
    """Run in parallel on several GPUs: one GPU per run, partial runs first, longest estimated runs first; runs with
    ``train.init_from`` start only once their source run is DONE."""
    todo = [j for j in jobs if j.pending]
    if not todo:
        return []
    cfgs = {j.run_id: load_config(j.files, j.overrides, use_defaults=False) for j in todo}
    stores = {k: run_store_for(c) for k, c in cfgs.items()}
    ordered = order_jobs(todo, stores)
    if sec_per_step:         # per-architecture measurements (all runs have the same steps): stable sort
        ordered = sorted(ordered, key=lambda j: (stores[j.run_id].status != "partial",
                                                 -float(sec_per_step.get(j.arch, 0.0))))
    weights = gpu_shares(todo, job_mem_mb, gpu_mem_mb, per_gpu)
    if weights:
        alone = sum(w > 0.5 for w in weights.values())
        log.info("Packing runs by measured memory: %d runs need a GPU of their own, %d share up to %d runs per GPU",
                 alone, len(weights) - alone, per_gpu)

    def ready(run_id: str) -> bool:
        """From-scratch runs can start at once; runs loading a checkpoint wait for their source run to be DONE."""
        if not cfgs[run_id].train.init_from:
            return True
        try:
            find_init_run(cfgs[run_id])
            return True
        except ConfigError:
            return False

    waiting = [j.run_id for j in ordered if not ready(j.run_id)]
    if waiting:
        log.info("%d runs wait for their source run (train.init_from) to finish", len(waiting))
    with ProcessRunner(REPO_ROOT, gpus=gpus, per_gpu=per_gpu, start_hours=max_hours) as runner:
        codes = runner.run([(j.run_id, j.argv) for j in ordered], weights=weights or None, ready=ready)
    out = list(codes.values())
    if runner.interrupted or runner.stopped_starting:
        out.append(EXIT_INTERRUPTED)
    return out


def _run_child(argv: list[str], grace_s: float = 600.0) -> int:
    """Run one training run in a child process. On a stop request (Ctrl-C / Colab's stop button) SIGTERM is forwarded
    to the running run and we WAIT for its checkpoint instead of killing it as ``subprocess.run`` would."""
    proc = subprocess.Popen(argv, cwd=REPO_ROOT)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        log.warning("Stopping: waiting for the current run to save its checkpoint (up to %.0f seconds)...", grace_s)
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=grace_s)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            proc.kill()
            proc.wait()
        return EXIT_INTERRUPTED
