"""lact train {run, grid, pilot, plan}: train one run, run experiment grids, measure trial cost and estimate the
remaining GPU hours (steps 10, 11, 21)."""
from __future__ import annotations

import logging

import pandas as pd

from ...config import ConfigError, load_config, load_geometry
from ...config.defaults import geometry_file, model_file
from ...infra.system.runtime import select_device
from ...services import scheduling
from ...services.evaluation import benchmark
from ...services.training import monitor
from ...services.training import trainer as training
from ..common import MODELS_MAIN, add_gpu_args, config_overrides, load_cfg, pilot_tables, resolve_gpus

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
    # Several grid files at once (e.g. e3 e4 e5) => one shared queue, run in parallel with --gpus.
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
        mem, sec, gpu_mem = pilot_tables(a.mem_table, gpus) if a.mem_table and gpus else (None, None, None)
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


def cmd_pilot(a) -> int:
    if a.spec:
        # measure exactly the grid's run configurations (data, split, epochs, update rule, overrides)
        geom = load_geometry(a.geometry).name
        jobs = [(j, load_config(j.files, j.overrides, use_defaults=False))
                for j in scheduling.filter_jobs(scheduling.expand(a.spec, a.config, config_overrides(a),
                                                                    models=a.models), exclude=a.exclude)]
        jobs = [(j, c) for j, c in jobs if c.require_geometry().name == geom]
        # grids with an lr/seed axis have several runs per architecture: key by run name to count every run
        repeated = len({j.arch for j, _ in jobs}) < len(jobs)
        cfgs = {(j.run_id if repeated else j.arch): c for j, c in jobs}
        if not cfgs:
            raise ConfigError(f"{a.spec}: no run at geometry {geom} with --models {a.models}")
        name = f"pilot_{next(iter(cfgs.values())).exp}_{geom}{a.tag}.csv"
    else:
        cfgs = {m: load_cfg(a, [model_file(m), geometry_file(a.geometry)]) for m in a.models or MODELS_MAIN}
        name = f"pilot_{a.geometry}{a.tag}.csv"
    first = next(iter(cfgs.values()))
    # device from the fully merged configuration: `device(a)` loads a config without a model file and fails with -s
    # model.*
    df = benchmark.pilot(cfgs, a.steps, a.planned_runs, select_device(first.device))
    out = first.paths.tables / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Wrote %s", out)
    return 0


PLAN_EXPS = ["e2_sweep", "e3_main", "e4_matched", "e5_a150"]


def cmd_plan(a) -> int:
    """GPU hours still to train = remaining steps x the seconds/step the pilot measured for each architecture."""
    pilot = pd.read_csv(a.pilot)
    # pilot models that could not be measured (OOM) are dropped
    ok = pilot[pilot.sec_per_step.notna()]
    sec = dict(zip(ok.model, ok.sec_per_step.astype(float)))
    jobs = [j for e in a.exp for j in scheduling.expand(f"configs/experiment/{e}.yaml", a.config,
                                                         config_overrides(a), models=a.models)]
    rows = pd.DataFrame(scheduling.remaining_work(jobs, sec, int(pilot.n_train_slices.iloc[0])))
    done = sum(not j.pending for j in jobs)
    if rows.empty:
        print(f"No run left to train ({done} runs finished).")
        return 0
    by = rows.groupby("exp", sort=False).agg(runs=("run_id", "size"), gpu_hours=("gpu_hours", "sum"))
    print("Still to train (speed measured at 1 run/GPU; E3-E5 runs without a chosen lr are included):")
    for exp, r in by.iterrows():
        print(f"  {exp:11s} {int(r.runs):4d} runs  {r.gpu_hours:7.1f} GPU-hours")
    total = float(rows.gpu_hours.sum())
    print(f"  {'TOTAL':11s} {len(rows):4d} runs  {total:7.1f} GPU-hours   (finished: {done} runs)")
    slow = rows.groupby("arch").gpu_hours.sum().sort_values(ascending=False).head(3)
    print("  Heaviest: " + ", ".join(f"{k} {v:.0f} GPU-hours" for k, v in slow.items()))
    if not rows.measured.all():
        print("  NOTE: no measurement for", sorted(rows[~rows.measured].arch.unique()), "(the median was used).")
    if a.n_gpus:
        wall = total / (a.n_gpus * a.gain)
        print(f"  With {a.n_gpus} GPUs (throughput x{a.gain:.2f}/GPU): about {wall:.0f} hours = {wall / 24:.1f} days. "
              f"Rental cost ~ {wall:.0f} x the machine's hourly price.")
    return 0


def register(sub, parent) -> None:
    p = sub.add_parser("train", help="Training: one run, experiment grids, trial cost measurement")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    c = s.add_parser("run", parents=[parent], help="Train one run (resumes from last.pt; skipped when DONE)",
                     description="Example: lact train run -m learn_convnext -g fan_a120 -s exp=e3_main seed=0")
    c.add_argument("-m", "--model", default=None, help="Name in configs/model or a .yaml path")
    c.add_argument("-g", "--geometry", default=None, help="Name in configs/geometry or a .yaml path")
    c.add_argument("--run-id", default=None, help="Run name (default: derived from the configuration)")
    c.set_defaults(func=cmd_run)

    c = s.add_parser("grid", parents=[parent], help="Expand/run configs/experiment/*.yaml grids",
                     description="DONE runs are skipped; DIVERGED runs are listed but not restarted.")
    c.add_argument("experiment", nargs="+", help="One or more grid files, e.g. e3_main.yaml e4_matched.yaml")
    c.add_argument("--dry-run", action="store_true", help="Print the command of each run")
    c.add_argument("--run", action="store_true",
                   help="Run the unfinished runs: sequentially, or in parallel with one GPU per run when --gpus is "
                        "given")
    c.add_argument("--emit", default=None, metavar="SH", help="Write the command list to a .sh file")
    c.add_argument("--index", type=int, default=None, help="Take only run number i (SLURM array)")
    c.add_argument("--match", nargs="+", default=None, metavar="TEXT",
                   help="Only runs whose run_id contains one of these strings, e.g. --match convnext")
    c.add_argument("--models", nargs="+", default=None, metavar="ARCH",
                   help="Only these models (names in configs/model, e.g. learn_cnn learn_resnet)")
    c.add_argument("--exclude", nargs="+", default=None, metavar="TEXT",
                   help="Drop runs whose run_id contains one of these strings (e.g. --exclude regformer to run them "
                        "later)")
    c.add_argument("--mem-table", nargs="+", default=None, metavar="CSV",
                   help="Table from `lact train pilot` (model, peak_mem_mb, sec_per_step): runs needing more than half "
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

    c = s.add_parser("pilot", parents=[parent], help="Gate 1: measure seconds/step and memory, estimate GPU hours")
    c.add_argument("--models", nargs="+", default=None,
                   help=f"Default: {' '.join(MODELS_MAIN)}; with --spec the default is every model of the grid")
    c.add_argument("--spec", default=None, metavar="YAML",
                   help="Measure exactly the runs of a configs/experiment/*.yaml grid (instead of the defaults)")
    c.add_argument("--exclude", nargs="+", default=None, metavar="TEXT",
                   help="With --spec: drop runs whose run_id contains one of these strings")
    c.add_argument("--tag", default="", help="Suffix of the result file name, e.g. _cap")
    c.add_argument("--geometry", default="fan_a120")
    c.add_argument("--steps", type=int, default=200)
    c.add_argument("--planned-runs", type=int, default=None, help="Rough estimate for N runs (see `train plan`)")
    c.set_defaults(func=cmd_pilot)

    c = s.add_parser("plan", parents=[parent], help="Estimate the GPU hours still to train from pilot measurements")
    c.add_argument("--pilot", required=True, metavar="CSV", help="Table written by `lact train pilot` (model, "
                                                                 "sec_per_step)")
    c.add_argument("--exp", nargs="+", default=PLAN_EXPS)
    c.add_argument("--models", nargs="+", default=None, metavar="ARCH", help="Only these models")
    c.add_argument("--n-gpus", type=int, default=0, help="Number of GPUs to use (to convert into wall-clock hours)")
    c.add_argument("--gain", type=float, default=1.0, help="Per-GPU throughput relative to 1 run/GPU (measured by "
                                                           "pilot)")
    c.set_defaults(func=cmd_plan)
