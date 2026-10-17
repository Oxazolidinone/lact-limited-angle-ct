"""Experiment grid specification (configs/experiment/*.yaml)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .loader import load_yaml
from .schema import ConfigError

GRID_KEYS = ("model", "geometry", "seed", "lr")


@dataclass
class ExperimentSpec:
    exp: str
    configs: list[str] = field(default_factory=list)       # shared configuration files; empty => defaults
    grid: dict[str, list[Any]] = field(default_factory=dict)
    lr_from: str | None = None                             # name of an lr-sweep experiment with best_lr.json
    # Learning rates given per regularizer name (same format as best_lr.json), e.g. the Mayo E2 choice reused on another
    # dataset without a new sweep. Models missing from the table take their lr from lr_from (when declared); without
    # lr_from the table must cover every model of the grid. Cannot be combined with grid.lr.
    lr_table: dict[str, float] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)
    # Seeds that are NOT trained anew: their runs belong to the experiment only if already DONE (e.g. seeds 1 and 2 of
    # the first four architectures on Mayo, trained while the grid still had 3 seeds). Reports pick one seed per setup.
    keep_finished_seeds: list[int] = field(default_factory=list)
    # cross-geometry inference only (E6)
    source_exp: str | None = None
    train_geometries: list[str] = field(default_factory=list)
    test_geometries: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> ExperimentSpec:
        d = load_yaml(path)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ConfigError(f"{path}: invalid keys {unknown}")
        spec = cls(**d)
        bad = sorted(set(spec.grid) - set(GRID_KEYS))
        if bad:
            raise ConfigError(f"{path}: grid accepts only {GRID_KEYS}, got {bad}")
        if set(spec.keep_finished_seeds) & set(spec.grid.get("seed", [0])):
            raise ConfigError(f"{path}: keep_finished_seeds overlaps grid.seed")
        if spec.lr_table and "lr" in spec.grid:
            raise ConfigError(f"{path}: lr_table cannot be combined with grid.lr")
        bad_lr = {k: v for k, v in spec.lr_table.items() if isinstance(v, bool) or not isinstance(v, (int, float))
                  or v <= 0}
        if bad_lr:
            raise ConfigError(f"{path}: lr_table needs positive numbers, got {bad_lr}")
        return spec
