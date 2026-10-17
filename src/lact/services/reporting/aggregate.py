"""Result aggregation: slices -> patient means -> mean over patients -> mean ± standard deviation over seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...config import ConfigError
from ...infra.stores.run_store import RunStore

METRICS = ["psnr", "ssim", "rmse", "rmse_hu", "dc_rel", "psnr_body", "ssim_body", "rmse_body"]
RUN_KEYS = ["exp", "arch", "regularizer", "geometry_train", "geometry_test", "split", "run_id", "seed"]
METHOD_KEYS = ["exp", "arch", "regularizer", "geometry_train", "geometry_test", "split"]


def _metrics_in(df: pd.DataFrame, metrics=None) -> list[str]:
    return [m for m in (metrics or METRICS) if m in df.columns]


def per_patient(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Per-slice mean within each patient, for each run."""
    ms = _metrics_in(df, metrics)
    keys = [k for k in RUN_KEYS if k in df.columns] + ["pid"]
    g = df.groupby(keys, dropna=False)
    out = g[ms].mean()
    out["n_slices"] = g.size()
    return out.reset_index()


def per_run(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean over patients (equal weight per patient) for each run."""
    pp = per_patient(df, metrics)
    ms = _metrics_in(pp, metrics)
    g = pp.groupby([k for k in RUN_KEYS if k in pp.columns], dropna=False)
    out = g[ms].mean()
    out["n_patients"] = g.size()
    return out.reset_index()


def _mean_std(df: pd.DataFrame, keys: list[str], ms: list[str]) -> pd.DataFrame:
    g = df.groupby(keys, dropna=False)
    out = pd.concat([g[ms].mean().add_suffix("_mean"), g[ms].std(ddof=1).add_suffix("_std")], axis=1)
    out["n_seeds"] = g.size()
    return out.reset_index()


def per_method(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Mean ± standard deviation over seeds of the patient-aggregated metrics."""
    pr = per_run(df, metrics)
    return _mean_std(pr, [k for k in METHOD_KEYS if k in pr.columns], _metrics_in(pr, metrics))


def per_method_patient(df: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """Per-patient table: mean ± standard deviation over seeds."""
    pp = per_patient(df, metrics)
    return _mean_std(pp, [k for k in METHOD_KEYS if k in pp.columns] + ["pid"], _metrics_in(pp, metrics))


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if std is None or not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def diverged_lrs(runs: list[RunStore]) -> dict[str, list[float]]:
    """{regularizer: [lr that made a run diverge]} from the runs' summary.json files (e.g. E3-E5 runs)."""
    out: dict[str, set[float]] = {}
    for run in runs:
        s = run.summary()
        if s.get("status") == "diverged" and s.get("regularizer") and s.get("lr") is not None:
            out.setdefault(s["regularizer"], set()).add(float(s["lr"]))
    return {k: sorted(v) for k, v in out.items()}


def select_best_lr(runs: list[RunStore], exclude: dict[str, list[float]] | None = None) -> dict:
    """E2: pick the learning rate with the best validation PSNR for each regularizer from the runs' summary.json.

    ``exclude``: {regularizer: [lr]} excluded despite a high E2 score - an lr that made a main-experiment run diverge
    (E2 trains only 20 epochs, so late instabilities go unnoticed). The best remaining lr is used instead."""
    rows = []
    for run in runs:
        s = run.summary()
        if not s:
            continue
        rows.append({"regularizer": s["regularizer"], "arch": s["arch"], "lr": s["lr"], "seed": s["seed"],
                     "best_val_psnr": s["best_val_psnr"] if s["status"] == "done" else -np.inf,
                     "status": s["status"]})
    if not rows:
        raise FileNotFoundError("No summary.json in the given runs")
    df = pd.DataFrame(rows)
    best, notes, excluded = {}, {}, {}
    for reg, g in df.groupby("regularizer"):
        valid = g[(g.status == "done") & np.isfinite(g.best_val_psnr)]
        bad = [lr for lr in (exclude or {}).get(reg, []) if any(np.isclose(valid.lr.astype(float), lr))]
        if bad:
            hit = np.isclose(valid.lr.to_numpy(float)[:, None], np.asarray(bad, float)[None, :]).any(axis=1)
            valid = valid[~hit]
            excluded[reg] = bad
        if valid.empty:
            raise ConfigError(f"E2: {reg} has no DONE run with a finite PSNR" + (
                f" outside the diverging learning rates {bad}" if bad else "") + "; cannot choose an lr yet.")
        lr = float(valid.sort_values("best_val_psnr", ascending=False).iloc[0]["lr"])
        best[reg] = lr
        lrs = sorted(float(v) for v in g["lr"].unique())
        if bad:
            notes[reg] = (f"lr {', '.join(f'{b:g}' for b in bad)} had the best E2 score but made a main-experiment "
                          f"run diverge; using the best remaining lr {lr:g}.")
        elif lr in (lrs[0], lrs[-1]) and len(lrs) > 1:
            notes[reg] = f"best lr {lr:g} is on the grid boundary {lrs}; consider one more step."
    return {"best_lr": best, "table": df.sort_values(["regularizer", "lr"]).to_dict(orient="records"),
            "notes": notes, "excluded_lr": excluded}
