"""Process queue: limits the number of jobs, assigns GPUs, forwards stop signals and waits for checkpoints.

One ProcessRunner shares a single deadline across all pipeline stages.

Multiple GPUs: pass ``gpus`` (a list of UUIDs) and ``per_gpu``. Each child job sees exactly one GPU through
``CUDA_VISIBLE_DEVICES=<UUID>`` (inside the job the GPU is ``cuda:0``), and no GPU runs more than ``per_gpu`` jobs at
once. A new job always goes to the GPU with the fewest jobs. Without ``gpus`` the old behaviour is kept: every job
uses the device chosen by the configuration.

Memory-aware packing: ``run(..., weights={job: GPU share})`` gives each job a "GPU share" in (0, 1] (e.g. peak memory
divided by the GPU memory budget); the shares of the jobs on one GPU never exceed 1. Heavy jobs (share > 1/2)
therefore run alone and light jobs share a GPU, at most ``per_gpu`` of them.

Dependencies between jobs: ``run(..., ready=fn)``. A job starts only when ``ready(name)`` returns True (e.g. an L-BFGS
run waiting for the first-order run whose checkpoint it loads); the condition is re-evaluated after each job ends.
Remaining jobs that no running job can unblock receive the ``BLOCKED`` code.
"""
from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

log = logging.getLogger("lact")
INTERRUPTED = 3
BLOCKED = 4          # the job never ran because its ``ready`` condition can never hold (e.g. its source run failed)
# SIGHUP: the SSH connection dropped outside tmux. Treat it as a graceful stop request instead of letting the parent die
# and leaving child jobs (in their own sessions) writing into the run directories.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class ProcessRunner:
    def __init__(self, cwd: Path, max_hours: float | None = None, grace_s: float = 600,
                 gpus: Sequence[str] | None = None, per_gpu: int = 1, start_hours: float | None = None,
                 solo_gpus: Sequence[str] | None = None):
        """``max_hours``: at the deadline, gracefully stop EVERY job (waiting for checkpoints). ``start_hours``: at the
        deadline, only stop launching new jobs and let running ones finish (the meaning of ``train grid --max-hours``).
        ``solo_gpus``: GPUs that take ONE job at a time even when ``per_gpu`` > 1 (e.g. GPUs on a slow PCIe link,
        where two jobs together run slower than one after the other)."""
        if per_gpu < 1:
            raise ValueError("per_gpu must be >= 1")
        if gpus is not None and len(set(gpus)) != len(list(gpus)):
            raise ValueError("The GPU list contains duplicates")
        self.cwd = cwd
        self.deadline = None if max_hours is None else time.monotonic() + max_hours * 3600
        self.start_deadline = None if start_hours is None else time.monotonic() + start_hours * 3600
        self.stopped_starting = False
        self.grace_s = grace_s
        self.gpus = list(gpus or [])
        self.per_gpu = per_gpu
        self.solo = set(solo_gpus or []) & set(self.gpus)
        self.interrupted = False
        self._handlers: dict = {}

    @property
    def slots(self) -> int:
        """Maximum number of concurrent jobs when a GPU list is given."""
        return len(self.gpus) * self.per_gpu if self.gpus else 1

    def __enter__(self):
        for sig in STOP_SIGNALS:
            self._handlers[sig] = signal.signal(sig, self._interrupt)
        return self

    def __exit__(self, *_):
        for sig, handler in self._handlers.items():
            signal.signal(sig, handler)

    def _interrupt(self, *_):
        self.interrupted = True

    def expired(self) -> bool:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.interrupted = True
        return self.interrupted

    def _env(self, gpu: str | None, share: float = 0.0) -> dict[str, str] | None:
        if gpu is None:
            return None
        together = max(1, min(self.per_gpu, int(1.0 / share + 1e-9))) if share > 0 else self.per_gpu
        return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu,
                "LACT_GPU": gpu, "LACT_JOBS_PER_GPU": str(together)}

    def _share(self, key: str, weights: Mapping[str, float] | None) -> float:
        """GPU share of a job: at least 1/per_gpu (job-count limit), at most 1 (a whole GPU)."""
        w = 1.0 / self.per_gpu if weights is None or key not in weights else float(weights[key])
        return min(1.0, max(1.0 / self.per_gpu, w))

    def _pick(self, queue: Sequence[tuple[str, list[str]]], load: Mapping[str, float],
              weights: Mapping[str, float] | None, eligible: Sequence[int] | None = None) -> tuple[int, str] | None:
        """(queue position, GPU) of the next job to launch; None when no job fits yet.

        The job at the head of the queue goes to the least loaded GPU with enough room. When it fits nowhere yet (a
        heavy job needs a whole GPU), the least loaded GPU is RESERVED for it and later jobs may only fill the other
        GPUs, so light jobs cannot starve a heavy one. ``eligible``: positions of the jobs allowed to run now
        (None = all); jobs waiting for a dependency reserve nothing."""
        idx = list(range(len(queue))) if eligible is None else list(eligible)
        if not idx:
            return None
        order = sorted(self.gpus, key=lambda g: (load[g], self.gpus.index(g)))
        fits = lambda g, w: load[g] + (1.0 if g in self.solo else w) <= 1.0 + 1e-9
        head = self._share(queue[idx[0]][0], weights)
        for g in order:
            if fits(g, head):
                return idx[0], g
        for i in idx[1:]:
            w = self._share(queue[i][0], weights)
            for g in order[1:]:
                if fits(g, w):
                    return i, g
        return None

    def run(self, commands: Sequence[tuple[str, list[str]]], workers: int | None = None,
            on_exit: Callable[[str, int], None] | None = None,
            weights: Mapping[str, float] | None = None,
            ready: Callable[[str], bool] | None = None) -> dict[str, int]:
        """Run at most ``workers`` jobs (default: the number of GPU slots, or 1). Ordinary failures are recorded;
        exit code 3 / a signal stops the whole queue. With ``gpus``, ``workers`` never exceeds the slot count.
        ``weights``: GPU share per job (see the module docstring); by default every job takes 1/per_gpu.
        ``ready``: a job starts only when ``ready(name)`` is True (see the module docstring)."""
        workers = self.slots if workers is None else workers
        if self.gpus:
            workers = min(workers, self.slots)
        workers = max(1, workers)
        queue = list(commands)
        active: dict[str, subprocess.Popen] = {}
        on_gpu: dict[str, str] = {}
        share: dict[str, float] = {}
        load = {g: 0.0 for g in self.gpus}
        codes: dict[str, int] = {}
        exhausted = False
        stopping_at = None
        ready_now: set[str] = set()
        recheck = True                   # re-evaluate ``ready`` at start and after each job ends (not every loop)
        try:
            while active or not exhausted:
                if self.expired() and stopping_at is None:
                    stopping_at = time.monotonic()
                    log.warning("Stopping the pipeline: waiting for jobs to save checkpoints (up to %.0fs).",
                                self.grace_s)
                    for proc in active.values():
                        if proc.poll() is None:
                            proc.terminate()
                if (not exhausted and self.start_deadline is not None
                        and time.monotonic() >= self.start_deadline):
                    exhausted = self.stopped_starting = True
                    log.warning("Launch window over: no new jobs are started, waiting for running jobs.")
                while not exhausted and not self.interrupted and len(active) < workers:
                    if not queue:
                        exhausted = True
                        break
                    eligible = None
                    if ready is not None:
                        if recheck:
                            ready_now |= {key for key, _ in queue if key not in ready_now and ready(key)}
                            recheck = False
                        eligible = [k for k, (key, _) in enumerate(queue) if key in ready_now]
                        if not eligible:
                            if not active:       # no running job is left to unblock the rest
                                log.error("Cannot run %d jobs because their dependencies never completed: %s",
                                          len(queue), ", ".join(key for key, _ in queue))
                                for key, _ in queue:
                                    codes[key] = BLOCKED
                                    if on_exit:
                                        on_exit(key, BLOCKED)
                                queue.clear()
                                exhausted = True
                            break
                    gpu, i = None, (eligible[0] if eligible else 0)
                    if self.gpus:
                        pick = self._pick(queue, load, weights, eligible)
                        if pick is None:
                            break
                        i, gpu = pick
                    key, argv = queue.pop(i)
                    w = 0.0 if gpu is None else 1.0 if gpu in self.solo else self._share(key, weights)
                    log.info("==> %s%s", key, f"  [GPU {gpu}]" if gpu else "")
                    active[key] = subprocess.Popen(argv, cwd=self.cwd, start_new_session=True,
                                                   env=self._env(gpu, w))
                    if gpu is not None:
                        share[key] = w
                        load[gpu] += w
                        on_gpu[key] = gpu
                for key, proc in list(active.items()):
                    code = proc.poll()
                    if code is None:
                        if stopping_at is not None and time.monotonic() - stopping_at >= self.grace_s:
                            _kill_group(proc)
                        continue
                    codes[key] = code
                    del active[key]
                    recheck = True
                    if key in on_gpu:
                        g = on_gpu.pop(key)
                        load[g] = max(0.0, round(load[g] - share.pop(key), 9))
                    if on_exit:
                        on_exit(key, code)
                    if code in (INTERRUPTED, -signal.SIGINT, -signal.SIGTERM, -signal.SIGHUP, 130, 143):
                        self.interrupted = True
                if self.interrupted and not active:
                    break
                if active:
                    time.sleep(0.1)
        finally:
            # I/O, launch or callback errors must not leave orphan jobs writing to disk either.
            for proc in active.values():
                if proc.poll() is None:
                    proc.terminate()
            for proc in active.values():
                try:
                    proc.wait(timeout=self.grace_s)
                except subprocess.TimeoutExpired:
                    _kill_group(proc)
                    proc.wait()
        return codes


def _kill_group(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):   # the process group may have ended between two checks
        os.killpg(proc.pid, signal.SIGKILL)
