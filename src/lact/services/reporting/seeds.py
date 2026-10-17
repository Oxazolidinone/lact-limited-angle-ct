"""Pick the run (seed) to report when a configuration has several trained seeds.

Convention since 2026-10-05: each configuration (experiment, architecture, geometry) reports ONE run. A configuration
with a single seed uses it; a configuration with several seeds (the first four architectures on Mayo, trained while
the grid still had 3 seeds) uses the seed with the best VALIDATION PSNR. Never selected on the test split.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...infra.stores.run_store import list_runs

POLICIES = ("best_val", "first", "all")
GROUP = ["exp", "arch", "geometry"]


def run_table(runs_root: str | Path, exps: list[str]) -> pd.DataFrame:
    """DONE runs of ``exps``: exp, arch, regularizer, geometry, seed, run_id, best_val_psnr."""
    rows = []
    for exp in exps:
        for run in list_runs(runs_root, exp):
            s = run.summary()
            if run.status != "done" or s.get("best_val_psnr") is None:
                continue
            rows.append({"exp": exp, "arch": s.get("arch"), "regularizer": s.get("regularizer"),
                         "geometry": s.get("geometry"), "seed": int(s.get("seed", 0)), "run_id": run.run_id,
                         "best_val_psnr": float(s["best_val_psnr"])})
    return pd.DataFrame(rows, columns=["exp", "arch", "regularizer", "geometry", "seed", "run_id", "best_val_psnr"])


def choose(table: pd.DataFrame, policy: str = "best_val") -> pd.DataFrame:
    """Add the columns ``chosen`` (bool) and ``n_seeds``. ``all``: keep every seed (mean ± standard deviation)."""
    if policy not in POLICIES:
        raise ValueError(f"seed_policy must be one of {POLICIES}, got {policy!r}")
    t = table.copy()
    if t.empty:
        return t.assign(chosen=pd.Series(dtype=bool), n_seeds=pd.Series(dtype=int))
    t["n_seeds"] = t.groupby(GROUP)["seed"].transform("size")
    if policy == "all":
        return t.assign(chosen=True)
    by = ["best_val_psnr", "seed"] if policy == "best_val" else ["seed"]
    asc = [False, True] if policy == "best_val" else [True]          # ties go to the smaller seed
    best = t.sort_values(by, ascending=asc).groupby(GROUP, sort=False).head(1).index
    return t.assign(chosen=t.index.isin(best))


def seed_selection(runs_root: str | Path, exps: list[str], policy: str = "best_val") -> pd.DataFrame:
    return choose(run_table(runs_root, exps), policy)


def keep_chosen(df: pd.DataFrame, selection: pd.DataFrame, exps: list[str]) -> pd.DataFrame:
    """Drop result rows of the runs of ``exps`` that were NOT chosen; baselines and other experiments are kept."""
    if df.empty or selection.empty or selection.chosen.all():
        return df
    dropped = set(selection[~selection.chosen].run_id)
    return df[~(df.exp.isin(exps) & df.run_id.isin(dropped))]


def chosen_run_ids(selection: pd.DataFrame) -> set[str]:
    return set(selection[selection.chosen].run_id) if len(selection) else set()
