"""Experiment grids: expand ``configs/experiment/*.yaml`` into training jobs, filter them, emit a shell script."""
from __future__ import annotations

import itertools
import logging
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ...config import ConfigError, ExperimentSpec, load_config
from ...config.defaults import DEFAULT_FILES, LOCAL_FILE
from ...config.loader import REPO_ROOT, resolve_path
from ...infra.io.files import load_json
from ..training.trainer import make_run_id, run_store_for

log = logging.getLogger("lact")


@dataclass
class Job:
    run_id: str
    status: str                                  # todo | partial | done | diverged | oom
    files: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)
    arch: str = ""                               # model configuration name (model.arch), e.g. learn_resnet_regnet

    @property
    def pending(self) -> bool:
        return self.status in ("todo", "partial")

    @property
    def argv(self) -> list[str]:
        return [sys.executable, "-m", "lact", "train", "run", "--no-defaults", "-c", *self.files,
                "-s", *self.overrides]

    @property
    def command(self) -> str:
        return " ".join(shlex.quote(x) for x in self.argv)


def _base_files(spec: ExperimentSpec, extra_files: list[str]) -> list[str]:
    files = list(spec.configs or DEFAULT_FILES)
    if resolve_path(LOCAL_FILE).exists():
        files.append(LOCAL_FILE)
    return files + list(extra_files)


def load_best_lr(runs_root: Path, lr_from: str) -> dict[str, float]:
    """{regularizer name: lr} from <runs>/<lr_from>/best_lr.json (written by ``lact report select-lr``)."""
    p = Path(runs_root) / lr_from / "best_lr.json"
    if not p.exists():
        log.warning("%s does not exist yet; using the default train.lr", p)
        return {}
    return load_json(p).get("best_lr", {})


def renamed(exp: str, rename: dict[str, str] | None) -> str:
    """Rename experiments by prefix, e.g. {"lbfgs_": "ctrl_"}: the same grid with runs in another runs directory (used
    when re-running a grid with -s overrides, e.g. control runs that keep the old data step: -s model.update=gd)."""
    for old, new in (rename or {}).items():
        if exp.startswith(old):
            return new + exp[len(old):]
    return exp


def expand(spec_file: str, extra_files: list[str] | None = None, overrides: list[str] | None = None,
           require_lr: bool = False, models: list[str] | None = None,
           exp_rename: dict[str, str] | None = None) -> list[Job]:
    """``models``: keep only runs whose model.arch is listed (learning rates of other models are not required).
    ``exp_rename``: see ``renamed``."""
    spec = ExperimentSpec.load(spec_file)
    exp = renamed(spec.exp, exp_rename)
    base = _base_files(spec, extra_files or [])
    best_lr = {}
    if spec.lr_from:
        # Only paths.runs is needed here: drop model.* / geometry.* overrides, which have no model file to attach to yet
        # (e.g. `train grid e3_main.yaml -s model.K=3` from `make smoke`).
        path_sets = [o for o in (overrides or []) if not o.strip().startswith(("model.", "geometry."))]
        runs_root = load_config(base, path_sets, use_defaults=False).paths.runs
        best_lr = load_best_lr(runs_root, spec.lr_from)
    keys = [k for k in ("model", "geometry", "seed", "lr") if k in spec.grid]
    combos = [dict(zip(keys, combo)) for combo in itertools.product(*[spec.grid[k] for k in keys])]
    # (combination, only-if-finished): main seeds first, then the seeds kept only for runs already DONE
    items = [(c, False) for c in combos]
    rest = [k for k in keys if k != "seed"]
    for combo in itertools.product(*[spec.grid[k] for k in rest]):
        items += [({**dict(zip(rest, combo)), "seed": s}, True) for s in spec.keep_finished_seeds]
    jobs = []
    for c, finished_only in items:
        files = base + ["configs/baseline/fbp.yaml", c["model"], c["geometry"]]
        sets = [f"exp={exp}", f"seed={c.get('seed', 0)}"] + list(spec.overrides)
        model = load_config(files, sets + list(overrides or []), use_defaults=False).require_model()
        if models and model.arch not in models:
            continue
        reg = model.regularizer.name
        if reg not in spec.lr_table:                 # precedence: grid.lr > lr_table > lr_from
            if spec.lr_table and not spec.lr_from:
                raise ConfigError(f"{spec_file}: lr_table has no lr for {reg} ({model.arch})")
            if spec.lr_from and reg not in best_lr:
                if finished_only:                    # without a chosen lr there cannot be a finished run
                    continue
                if require_lr:
                    raise ConfigError(f"No lr chosen yet for {reg} from {spec.lr_from}; finish that lr sweep "
                                      "first.")
        if "lr" in c:
            sets.append(f"train.lr={c['lr']}")
        elif reg in spec.lr_table:
            sets.append(f"train.lr={spec.lr_table[reg]}")
        elif reg in best_lr:
            sets.append(f"train.lr={best_lr[reg]}")
        sets += list(overrides or [])
        cfg = load_config(files, sets, use_defaults=False)
        status = run_store_for(cfg).status
        if finished_only and status != "done":
            continue
        jobs.append(Job(make_run_id(cfg), status, files, sets, model.arch))
    return jobs


def filter_jobs(jobs: list[Job], match: list[str] | None = None, exclude: list[str] | None = None) -> list[Job]:
    """Keep runs whose run_id contains ONE of the ``match`` strings (e.g. ['convnext'] to run ConvNeXt alone) and none
    of the ``exclude`` strings (e.g. ['regformer'] to run those later)."""
    if match:
        jobs = [j for j in jobs if any(m in j.run_id for m in match)]
    if exclude:
        jobs = [j for j in jobs if not any(x in j.run_id for x in exclude)]
    return jobs


def emit_script(jobs: list[Job], path: str | Path) -> Path:
    lines = ["#!/usr/bin/env bash", "set -e", f"cd {shlex.quote(str(REPO_ROOT))}"]
    lines += [j.command for j in jobs if j.pending]
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
