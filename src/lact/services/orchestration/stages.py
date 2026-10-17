"""Stages of the main workflow: geometry and experiment names, the stage record and the state-file digest."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...config import Config

MAIN_GEOMS = ["fan_a90", "fan_a120", "fan_a150"]
CHECK_GEOMS = ["fan_full", "fan_short"]
HALF_GEOMS = [f"{g}_half" for g in MAIN_GEOMS]
SWEEP = "e2_sweep"
MAIN_EXPS = ["e3_main", "e4_matched", "e5_a150"]
UNTIL = ("prepare", "sweep", "train", "all")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _scientific_config(cfg: Config) -> dict:
    """Allow changing GPUs/cache/checkpoint settings on resume while keeping the experiment configuration."""
    data = cfg.to_dict()
    for k in ("device", "cache", "derived", "run_id", "verify_data"):
        data.pop(k, None)
    for k in ("checkpoint_every_min", "keep_prev_checkpoint", "tensorboard", "require_gpu",
              "oom_retry_checkpoint", "checkpoint_grad"):
        data["train"].pop(k, None)
    return data


def state_file(cfg: Config) -> Path:
    return cfg.paths.outputs / "pipeline" / "main.json"


def describe() -> list[str]:
    return [
        "prepare: DICOM -> images.h5 -> A (shared) -> E0 -> sinograms -> E1 -> tune TV",
        "sweep: E2 (45 runs) -> check results are complete -> choose the lr of each architecture",
        "train: freeze the configuration -> E3 (30), E4 (3), E5 (15); one seed per configuration",
        "all: test evaluation -> baselines -> E6 (cross-geometry) -> cost -> tables/figures/analyses",
    ]


@dataclass
class Stage:
    key: str
    args: list[str]
    outputs: list[Path] = field(default_factory=list)
    # Accept existing caches only for commands that check/skip files that already exist.
    always: bool = False
    scan_dir: Path | None = None
    inputs: list[Path] = field(default_factory=list)
