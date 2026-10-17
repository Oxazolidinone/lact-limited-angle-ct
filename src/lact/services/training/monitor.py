"""Monitor training runs from the files in the runs directory (works while they run on another machine / Colab)."""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from ...config import Config
from ...infra.stores.run_store import list_runs


def _age_minutes(stamp: str | None) -> float | None:
    if not stamp:
        return None
    try:
        return round((_dt.datetime.now() - _dt.datetime.fromisoformat(stamp)).total_seconds() / 60, 1)
    except ValueError:
        return None


def run_status(cfg: Config, exps: list[str] | None = None) -> pd.DataFrame:
    """One row per run: status, progress, speed, ETA, best validation PSNR, GPU, last update."""
    root = cfg.paths.runs
    names = exps or (sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else [])
    rows = []
    for exp in names:
        for run in list_runs(root, exp):
            hb, summ = run.heartbeat(), run.summary()
            step, total = hb.get("step", summ.get("steps", 0)), hb.get("total_steps")
            rows.append({
                "exp": exp, "run_id": run.run_id, "status": run.status,
                "epoch": hb.get("epoch"), "step": step, "total_steps": total,
                "progress_%": round(100 * step / total, 1) if total else None,
                "sec_per_step": hb.get("sec_per_step"), "eta_h": hb.get("eta_hours"),
                "best_val_psnr": summ.get("best_val_psnr", hb.get("best_val_psnr")),
                "gpu": ", ".join(summ.get("gpus", [])) or hb.get("gpu"),
                "oom_events": sum(1 for e in run.events() if e.get("event") == "oom"),
                "updated_min_ago": _age_minutes(hb.get("time")),
            })
    return pd.DataFrame(rows)
