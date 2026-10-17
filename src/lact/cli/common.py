"""Shared command-line options and how configurations are loaded from them.

Merge order: default files -> configs/local.yaml (if present) -> -c files -> -s key=value overrides.
"""
from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence

import torch

from ..config import Config, ConfigError, load_config
from ..config.defaults import load_geometry
from ..core.imaging.geometry import FanBeamGeometry
from ..infra.system.runtime import select_device

GEOMS_MAIN = ["fan_a90", "fan_a120"]


def config_parser() -> argparse.ArgumentParser:
    """Parser cha: -c/--config, -s/--set, --no-defaults, --device."""
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("configuration")
    g.add_argument("-c", "--config", nargs="*", default=[], metavar="YAML",
                   help="Extra YAML files (merged in order)")
    g.add_argument("-s", "--set", nargs="*", default=[], dest="overrides", metavar="KEY=VALUE",
                   help="Override values, e.g. -s train.lr=1e-4 seed=1")
    g.add_argument("--no-defaults", action="store_true", help="Do not load the default configuration files")
    g.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:1 (default: from the configuration)")
    return p


def load_cfg(args: argparse.Namespace, extra: list[str] | None = None) -> Config:
    return load_config(list(extra or []) + list(args.config), config_overrides(args),
                       use_defaults=not args.no_defaults)


def config_overrides(args: argparse.Namespace) -> list[str]:
    return list(args.overrides) + ([f"device={args.device}"] if args.device is not None else [])


def device(args: argparse.Namespace) -> torch.device:
    return select_device(load_cfg(args).device)


def geometries(names: list[str]) -> list[FanBeamGeometry]:
    """Accept a short name ('fan_a120') or a .yaml path."""
    return [load_geometry(n) for n in names]


def add_geometries(p: argparse.ArgumentParser, default: list[str], nargs: str = "+") -> None:
    p.add_argument("-g", "--geometries", nargs=nargs, default=default, metavar="GEOM",
                   help=f"Geometry names in configs/geometry (default: {' '.join(default)})")


def add_gpu_args(p: argparse.ArgumentParser) -> None:
    """Options for running many jobs on many GPUs (train grid --run)."""
    g = p.add_argument_group("multiple GPUs")
    g.add_argument("--gpus", nargs="+", default=None, metavar="GPU",
                   help="auto (pick the --n-gpus most idle GPUs) | all (every visible GPU, e.g. in a container) | "
                        "nvidia-smi indices or UUIDs, e.g. --gpus 3 5")
    g.add_argument("--n-gpus", type=int, default=2, help="Number of GPUs with --gpus auto (default 2)")
    g.add_argument("--per-gpu", type=int, default=1, help="Concurrent jobs on EACH GPU (default 1)")
    g.add_argument("--max-used-mb", type=int, default=1024,
                   help="--gpus auto: a GPU counts as idle when its used memory is <= this (MiB)")
    g.add_argument("--max-util", type=int, default=10, help="--gpus auto: maximum load (%%) to count as idle")
    g.add_argument("--allow-shared", action="store_true",
                   help="--gpus auto: also accept busy GPUs with >= --min-free-mb free memory")
    g.add_argument("--min-free-mb", type=int, default=9216, help="Minimum free memory with --allow-shared")


def mem_tables(tables: Sequence[str], gpus: Sequence[str]
                 ) -> tuple[dict[str, float], dict[str, float], float | None]:
    """Merge cost tables (CSV: model, peak_mem_mb, sec_per_step): ({architecture: peak memory MB},
    {architecture: seconds/step}, memory of the smallest GPU used). An architecture present in several tables (grids,
    geometries) takes the LARGEST value."""
    import pandas as pd

    from ..infra.system import gpus as G

    df = pd.concat([pd.read_csv(t) for t in tables], ignore_index=True)
    mem = df.dropna(subset=["peak_mem_mb"]).groupby("model").peak_mem_mb.max().astype(float).to_dict()
    sec = df.dropna(subset=["sec_per_step"]).groupby("model").sec_per_step.max().astype(float).to_dict()
    totals = [g.mem_total_mb for g in G.query_gpus(samples=1) if g.uuid in gpus and g.mem_total_mb]
    return mem, sec, (float(min(totals)) if totals else None)


def resolve_gpus(args: argparse.Namespace, configs: Sequence[Config] | None = None) -> list[str]:
    """UUIDs of the GPUs to use (empty = single-device mode as before). Prints the status table when auto-picking."""
    if not args.gpus:
        return []
    from ..infra.system import gpus as G

    if args.per_gpu < 1 or args.n_gpus < 1:
        raise ConfigError("--per-gpu and --n-gpus must be >= 1")
    for cfg in configs if configs is not None else [load_cfg(args)]:
        if cfg.device not in ("auto", "cuda", "cuda:0"):
            raise ConfigError(f"With --gpus, the device after merging YAML/-s/--device must be auto, cuda or "
                              f"cuda:0 (it is {cfg.device!r}): each job sees only the GPU assigned to it.")
    try:
        chosen = G.resolve_gpus(args.gpus, args.n_gpus, args.max_used_mb, args.max_util, args.allow_shared,
                                args.min_free_mb)
        print(G.format_table(G.query_gpus(samples=1), chosen))
    except (OSError, subprocess.SubprocessError, G.GpuSelectionError) as exc:
        raise ConfigError(f"Cannot select GPUs: {exc}") from exc
    print(f"Using {len(chosen)} GPUs, at most {args.per_gpu} jobs/GPU: {', '.join(chosen)}")
    return chosen
