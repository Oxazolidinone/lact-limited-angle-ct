"""lact pipeline: one command for the main workflow, resuming automatically in later sessions."""
from __future__ import annotations

import json

from ...config import ConfigError
from ...infra.io.files import load_json
from ...services.orchestration.pipeline import Pipeline
from ...services.orchestration.stages import UNTIL, describe, state_file
from ..common import add_gpu_args, config_overrides, load_cfg, resolve_gpus


def cmd_run(a) -> int:
    if a.no_defaults:
        raise ConfigError("the pipeline uses each experiment's base configuration; customise with -c/-s.")
    cfg = load_cfg(a)
    gpus = resolve_gpus(a, [cfg])
    job_mem, gpu_mem = _memory(a.mem_table, gpus) if a.mem_table and gpus and a.per_gpu > 1 else (None, None)
    solo = _solo(a.solo_gpus, gpus) if a.solo_gpus and gpus else []
    pipeline = Pipeline(cfg, a.config, config_overrides(a), a.workers, gpus=gpus, per_gpu=a.per_gpu,
                        extend=a.extend, models=a.models, job_mem_mb=job_mem, gpu_mem_mb=gpu_mem, solo_gpus=solo)
    for line in describe():
        print(line)
    if a.dry_run:
        where = (f"{len(gpus)} GPU x {a.per_gpu} job" if gpus else f"workers={a.workers} on one device")
        print(f"{where}; until={a.until}; state: {state_file(cfg)}")
        return 0
    return pipeline.run(a.until, a.max_hours)


def _memory(table: str, gpus: list[str]) -> tuple[dict[str, float], float | None]:
    """({architecture: peak memory MB} from the `lact train pilot` table, memory of the smallest GPU used)."""
    import pandas as pd

    from ...infra.system import gpus as G

    df = pd.read_csv(table).dropna(subset=["peak_mem_mb"])
    totals = [g.mem_total_mb for g in G.query_gpus(samples=1) if g.uuid in gpus and g.mem_total_mb]
    return dict(zip(df.model, df.peak_mem_mb.astype(float))), (float(min(totals)) if totals else None)


def _solo(spec: list[str], gpus: list[str]) -> list[str]:
    """nvidia-smi indices or UUIDs -> UUIDs of the GPUs that run one job at a time (must be among ``gpus``)."""
    from ...infra.system import gpus as G

    by_index = {str(g.index): g.uuid for g in G.query_gpus(samples=1)}
    solo = [by_index.get(s, s) for s in spec]
    unknown = [s for s, u in zip(spec, solo) if u not in gpus]
    if unknown:
        raise ConfigError(f"--solo-gpus {unknown}: not among the GPUs in use")
    print(f"GPUs running one job at a time: {', '.join(solo)}")
    return solo


def cmd_status(a) -> int:
    path = state_file(load_cfg(a))
    if not path.exists():
        print("The pipeline has not run yet.")
        return 0
    print(json.dumps(load_json(path), ensure_ascii=False, indent=2))
    return 0


def register(sub, parent) -> None:
    p = sub.add_parser("pipeline", help="Run E0–E6 in one go, respecting dependencies and resuming automatically")
    s = p.add_subparsers(dest="cmd", required=True)
    c = s.add_parser("run", parents=[parent], help="Prepare -> E2 -> E3/E4/E5 -> E6 -> reporting")
    c.add_argument("--workers", type=int, default=1,
                   help="Concurrent training runs on ONE device (ignored with --gpus)")
    c.add_argument("--max-hours", type=float, default=None, help="Gracefully stop running jobs after N hours + save "
                                                                 "time")
    c.add_argument("--until", choices=UNTIL, default="all", help="Run up to prepare/sweep/train/all")
    c.add_argument("--models", nargs="+", default=None, metavar="ARCH",
                   help="Only these models (names in configs/model, e.g. learn_unet learn_unet_regnet); "
                        "keep the same list when resuming")
    c.add_argument("--extend", action="store_true",
                   help="Accept existing outputs after EXTENDING the grid (e.g. new architectures); old runs are kept")
    c.add_argument("--mem-table", default=None, metavar="CSV",
                   help="Table from `lact train pilot` (model, peak_mem_mb). With --per-gpu > 1: runs needing more "
                        "than half the GPU memory run alone, light runs share")
    c.add_argument("--solo-gpus", nargs="+", default=None, metavar="GPU",
                   help="GPUs (nvidia-smi index or UUID) that run ONE job at a time even with --per-gpu > 1")
    c.add_argument("--dry-run", action="store_true", help="Show the plan; write no files and use no GPU")
    add_gpu_args(c)
    c.set_defaults(func=cmd_run)
    c = s.add_parser("status", parents=[parent], help="Read the saved pipeline state")
    c.set_defaults(func=cmd_status)
