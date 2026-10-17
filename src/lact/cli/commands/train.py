"""lact train {run, grid, status}: train one run, run experiment grids, show their progress."""
from __future__ import annotations

import logging

import pandas as pd

from ...config import ConfigError, load_config
from ...config.defaults import geometry_file, model_file
from ...services import scheduling
from ...services.training import monitor
from ...services.training import trainer as training
from ..common import add_gpu_args, config_overrides, load_cfg, mem_tables, resolve_gpus

log = logging.getLogger("lact")


def cmd_run(a) -> int:
    extra = ([model_file(a.model)] if a.model else []) + ([geometry_file(a.geometry)] if a.geometry else [])
    cfg = load_cfg(a, extra)
    if a.run_id:
        cfg = cfg.replace(run_id=a.run_id)
    cfg.require_model(), cfg.require_geometry()
    summary = training.train(cfg)
    log.info("Result: %s", summary)
    if summary.get("status") == "interrupted":
        return training.EXIT_INTERRUPTED
    return 0 if summary.get("status") in ("done", "skipped") else 1


def cmd_grid(a) -> int:
    # Several grid files at once (e.g. base_mayo lbfgs_mayo cmp_mayo) => one shared queue, run in parallel with --gpus.
    bad = [r for r in a.exp_rename or [] if r.count("=") != 1 or not all(r.split("="))]
    if bad:
        raise ConfigError(f"--exp-rename expects OLD=NEW, got {bad}")
    rename = dict(r.split("=") for r in a.exp_rename or [])
    jobs = scheduling.filter_jobs([j for spec in a.experiment for j in
                                    scheduling.expand(spec, a.config, config_overrides(a), require_lr=a.run,
                                                       models=a.models, exp_rename=rename)],
                                   a.match, a.exclude)
    if len({j.run_id for j in jobs}) != len(jobs):
        raise ConfigError("The grids produce duplicate run_ids; they cannot run together.")
    if a.index is not None:
        jobs = [jobs[a.index]]
    for i, j in enumerate(jobs):
        print(f"[{i:02d}] {j.status:8s} {j.run_id}")
    if a.emit:
        print(f"Wrote {scheduling.emit_script(jobs, a.emit)}")
    if a.dry_run:
        for j in jobs:
            print(j.command)
    if a.run:
        cfgs = [load_config(j.files, j.overrides, use_defaults=False) for j in jobs] if a.gpus else None
        gpus = resolve_gpus(a, cfgs) or None
        mem, sec, gpu_mem = mem_tables(a.mem_table, gpus) if a.mem_table and gpus else (None, None, None)
        codes = scheduling.run_jobs(jobs, a.max_hours, gpus, a.per_gpu, mem, gpu_mem, sec)
        if training.EXIT_INTERRUPTED in codes:
            return training.EXIT_INTERRUPTED
        return 0 if all(c == 0 for c in codes) else 1
    return 0


def cmd_status(a) -> int:
    df = monitor.run_status(load_cfg(a), a.exp or None)
    if df.empty:
        print("No training runs yet.")
        return 0
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df.to_string(index=False))
    return 0


def register(sub, parent) -> None:
    p = sub.add_parser("train", help="Training: one run, experiment grids, progress")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    c = s.add_parser("run", parents=[parent], help="Train one run (resumes from last.pt; skipped when DONE)",
                     description="Example: lact train run -m learn_densenet -g fan_a90 -s exp=dev seed=0")
    c.add_argument("-m", "--model", default=None, help="Name in configs/model or a .yaml path")
    c.add_argument("-g", "--geometry", default=None, help="Name in configs/geometry or a .yaml path")
    c.add_argument("--run-id", default=None, help="Run name (default: derived from the configuration)")
    c.set_defaults(func=cmd_run)

    c = s.add_parser("grid", parents=[parent], help="Expand/run configs/experiment/*.yaml grids",
                     description="DONE runs are skipped; DIVERGED runs are listed but not restarted.")
    c.add_argument("experiment", nargs="+", help="One or more grid files, e.g. configs/experiment/lbfgs_mayo.yaml")
    c.add_argument("--dry-run", action="store_true", help="Print the command of each run")
    c.add_argument("--run", action="store_true",
                   help="Run the unfinished runs: sequentially, or in parallel with one GPU per run when --gpus is "
                        "given")
    c.add_argument("--emit", default=None, metavar="SH", help="Write the command list to a .sh file")
    c.add_argument("--index", type=int, default=None, help="Take only run number i (SLURM array)")
    c.add_argument("--match", nargs="+", default=None, metavar="TEXT",
                   help="Only runs whose run_id contains one of these strings, e.g. --match _fan_a90_")
    c.add_argument("--models", nargs="+", default=None, metavar="ARCH",
                   help="Only these models (names in configs/model, e.g. learn_densenet regformer)")
    c.add_argument("--exclude", nargs="+", default=None, metavar="TEXT",
                   help="Drop runs whose run_id contains one of these strings (e.g. --exclude regformer to run them "
                        "later)")
    c.add_argument("--mem-table", nargs="+", default=None, metavar="CSV",
                   help="CSV with columns model, peak_mem_mb, sec_per_step: runs needing more than half "
                        "the GPU memory run alone, the longest runs are queued first (needs --gpus)")
    c.add_argument("--exp-rename", nargs="+", default=None, metavar="OLD=NEW",
                   help="Rename the experiment prefix, e.g. lbfgs_=ctrl_: re-run the same grid with -s overrides, "
                        "with the runs in another runs directory")
    c.add_argument("--max-hours", type=float, default=None, help="Start no new runs after N hours")
    add_gpu_args(c)
    c.set_defaults(func=cmd_grid)

    c = s.add_parser("status", parents=[parent], help="Progress table of the runs (from heartbeat.json, summary.json)")
    c.add_argument("--exp", nargs="*", default=[])
    c.set_defaults(func=cmd_status)
