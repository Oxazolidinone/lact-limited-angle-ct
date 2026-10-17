"""Read GPU status with nvidia-smi and pick idle GPUs (shared servers).

Standard library only and Python 3.9 compatible, so ``scripts/server/pick_gpus.py`` can load this file with the host's
Python without installing the project. Never initialises CUDA, sets environment variables or occupies a GPU.

GPUs are identified by UUID (``GPU-...``) rather than index: CUDA's order (FASTEST_FIRST by default) can differ from
nvidia-smi's (PCI) order, whereas UUIDs are the same on the host, inside containers and in ``CUDA_VISIBLE_DEVICES``.
"""
from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

GPU_QUERY = "index,uuid,name,memory.total,memory.used,utilization.gpu"


class GpuSelectionError(RuntimeError):
    """Not enough GPUs satisfy the conditions; the message includes the status table."""


@dataclass
class GpuStatus:
    index: int
    uuid: str
    name: str
    mem_total_mb: int | None
    mem_used_mb: int | None
    util: int | None          # % (maximum over the samples); None = unreadable
    procs: int | None = None  # number of compute processes; None when unreadable (e.g. inside a container)

    @property
    def mem_free_mb(self) -> int | None:
        if (self.mem_total_mb is None or self.mem_used_mb is None
                or self.mem_total_mb <= 0 or not 0 <= self.mem_used_mb <= self.mem_total_mb):
            return None
        return self.mem_total_mb - self.mem_used_mb


def _int(value: str) -> int | None:
    value = value.strip()
    try:
        return int(float(value))
    except (ValueError, OverflowError):  # "[N/A]", "[Not Supported]", nan/inf
        return None


def _peak(a: int | None, b: int | None) -> int | None:
    # A sample with missing information cannot confirm that a GPU is idle.
    return max(a, b) if a is not None and b is not None else None


def parse_gpu_csv(text: str) -> list[GpuStatus]:
    """Parse ``nvidia-smi --query-gpu=<GPU_QUERY> --format=csv,noheader,nounits`` output."""
    out = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        index = _int(parts[0])
        if index is None or index < 0 or not parts[1].startswith("GPU-"):
            continue
        out.append(GpuStatus(index=index, uuid=parts[1], name=parts[2], mem_total_mb=_int(parts[3]),
                             mem_used_mb=_int(parts[4]), util=_int(parts[5])))
    return out


def parse_apps_csv(text: str) -> dict:
    """``--query-compute-apps=gpu_uuid,pid`` -> {uuid: number of processes}."""
    counts: dict = {}
    for line in text.strip().splitlines():
        uuid = line.split(",")[0].strip()
        if uuid.startswith("GPU-"):
            counts[uuid] = counts.get(uuid, 0) + 1
    return counts


def _smi(args: Sequence[str]) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True, timeout=30).stdout


def query_gpus(samples: int = 3, interval_s: float = 1.0,
               run: Callable[[Sequence[str]], str] = _smi) -> list[GpuStatus]:
    """Status of every visible GPU, sampled ``samples`` times keeping the LARGEST memory use and load (conservative)."""
    merged: dict = {}
    for i in range(max(1, samples)):
        if i:
            time.sleep(interval_s)
        for g in parse_gpu_csv(run([f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])):
            old = merged.get(g.uuid)
            if old is not None:
                g.mem_used_mb, g.util = _peak(g.mem_used_mb, old.mem_used_mb), _peak(g.util, old.util)
                if old.mem_total_mb is None:
                    g.mem_total_mb = None
            merged[g.uuid] = g
    try:
        apps = parse_apps_csv(run(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]))
        for g in merged.values():
            g.procs = apps.get(g.uuid, 0)
    except (OSError, subprocess.SubprocessError):
        pass
    return sorted(merged.values(), key=lambda g: g.index)


def format_table(gpus: Sequence[GpuStatus], chosen: Sequence[str] = ()) -> str:
    rows = ["  idx  used/total (MiB)  load  processes  UUID"]
    for g in gpus:
        mark = "*" if g.uuid in chosen else " "
        procs = "?" if g.procs is None else str(g.procs)
        used, total, util = ("?" if v is None else str(v) for v in (g.mem_used_mb, g.mem_total_mb, g.util))
        rows.append(f"{mark} {g.index:>3}  {used:>6}/{total:<6}   {util:>3}%  {procs:>10}  "
                    f"{g.uuid}  {g.name}")
    return "\n".join(rows)


def is_idle(g: GpuStatus, max_used_mb: int = 1024, max_util: int = 10) -> bool:
    """A GPU counts as idle only when memory, load and the compute-process list were all read."""
    return (g.mem_free_mb is not None and g.mem_used_mb is not None and g.mem_used_mb <= max_used_mb
            and g.util is not None and 0 <= g.util <= max_util and g.procs == 0)


def pick_free_gpus(n: int, gpus: Sequence[GpuStatus], max_used_mb: int = 1024, max_util: int = 10,
                   allow_shared: bool = False, min_free_mb: int = 9216,
                   allowed: Sequence[str] | None = None) -> list[str]:
    """Pick the ``n`` most idle GPUs and return their UUIDs. By default only unused GPUs qualify (``is_idle``).

    ``allow_shared``: also accept busy GPUs with at least ``min_free_mb`` free memory (not recommended on shared
    servers). ``allowed``: choose only among these UUIDs/indices (e.g. from CUDA_VISIBLE_DEVICES). With too few GPUs
    an error with the status table is raised; busy GPUs are NEVER taken silently.
    """
    if n < 1:
        raise ValueError("The number of GPUs to pick must be >= 1")
    pool = list(gpus)
    if allowed is not None:
        allow = {str(a) for a in allowed}
        pool = [g for g in pool if g.uuid in allow or str(g.index) in allow]
    ok = [g for g in pool if is_idle(g, max_used_mb, max_util)
          or (allow_shared and g.mem_free_mb is not None and g.mem_free_mb >= min_free_mb)]
    ok.sort(key=lambda g: (g.procs if g.procs is not None else float("inf"),
                          g.util if g.util is not None else float("inf"), g.mem_used_mb, g.index))
    if len(ok) < n:
        rule = (f"memory used <= {max_used_mb} MiB, load <= {max_util}%, no processes"
                + (f", or >= {min_free_mb} MiB free" if allow_shared else ""))
        raise GpuSelectionError(f"Only {len(ok)}/{n} GPUs satisfy the conditions ({rule}).\n{format_table(gpus)}")
    return [g.uuid for g in ok[:n]]


def visible_gpus(gpus: Sequence[GpuStatus]) -> list[GpuStatus]:
    """Respect the CUDA_VISIBLE_DEVICES limit set by the shell/scheduler, even when it is an empty string.

    Abbreviated UUIDs must match uniquely. An invalid index ends the list, as in CUDA; other GPUs are never added.
    nvidia-smi may see more GPUs than CUDA applications do.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is None:
        return list(gpus)
    out: list[GpuStatus] = []
    for token in env.split(","):
        token = token.strip()
        matches = [g for g in gpus if (g.uuid.startswith(token) if token.startswith("GPU-")
                                      else str(g.index) == token)]
        if len(matches) != 1 or matches[0] in out:
            break
        out.append(matches[0])
    return out


def resolve_gpus(spec: Sequence[str], n_auto: int = 2, max_used_mb: int = 1024, max_util: int = 10,
                 allow_shared: bool = False, min_free_mb: int = 9216,
                 query: Callable[[], list[GpuStatus]] = query_gpus) -> list[str]:
    """``spec``: ["auto"] (the ``n_auto`` most idle GPUs), ["all"] (every visible GPU, e.g. in a container), or a
    list of nvidia-smi indices / UUIDs. Returns a list of unique UUIDs."""
    spec = [s.strip() for s in spec if s.strip()]
    if not spec:
        return []
    gpus = visible_gpus(query())
    if spec == ["auto"]:
        return pick_free_gpus(n_auto, gpus, max_used_mb, max_util, allow_shared, min_free_mb)
    if spec == ["all"]:
        if not gpus:
            raise GpuSelectionError("No GPU within the CUDA_VISIBLE_DEVICES/nvidia-smi limits")
        return [g.uuid for g in gpus]
    by_index = {str(g.index): g.uuid for g in gpus}
    known = {g.uuid for g in gpus}
    out: list[str] = []
    for s in spec:
        uuid = s if s.startswith("GPU-") else by_index.get(s)
        if uuid is None or uuid not in known:
            raise GpuSelectionError(f"No GPU '{s}'.\n{format_table(gpus)}")
        if uuid in out:
            raise GpuSelectionError(f"GPU '{s}' is listed twice")
        out.append(uuid)
    return out
