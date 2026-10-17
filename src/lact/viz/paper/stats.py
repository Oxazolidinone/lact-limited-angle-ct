"""Statistics accompanying figures/tables: bootstrap confidence intervals, paired Wilcoxon tests + Holm correction.

The pairing unit is the slice (same patient, same slice, same geometry). With several seeds, a slice's metric is
averaged over seeds before pairing (seeds are a source of method variability, not independent samples).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as _st

KEYS = ["geometry", "pid", "idx"]


def bootstrap_ci(x: Sequence[float] | np.ndarray, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0,
                 groups: Sequence | None = None) -> tuple[float, float]:
    """(1-alpha) CI of the mean. groups != None => cluster bootstrap (e.g. by patient)."""
    arr = np.asarray(x, float)
    v = arr[np.isfinite(arr)]
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    if groups is None:
        idx = rng.integers(0, v.size, (n_boot, v.size))
        means = v[idx].mean(1)
    else:
        g = np.asarray(groups)[: v.size]
        uniq = np.unique(g)
        parts = [v[g == u] for u in uniq]
        means = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(parts), len(parts))
            means[b] = np.concatenate([parts[i] for i in pick]).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(p: Sequence[float]) -> np.ndarray:
    pv = np.asarray(p, float)
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(1.0, run)
    return adj


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."


def per_slice(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seed average for each (geometry, method, slice)."""
    return df.groupby(["method", *KEYS], as_index=False)[metric].mean()


def summarize(df: pd.DataFrame, metric: str, by: Sequence[str] = ("geometry", "method")) -> pd.DataFrame:
    """mean, std (across seeds when there are >1 seeds, otherwise across slices), 95% slice-bootstrap CI."""
    rows = []
    for key, g in df.groupby(list(by)):
        ps = g.groupby(["pid", "idx"])[metric].mean()
        lo, hi = bootstrap_ci(ps.values)
        n_seed = g["seed"].nunique() if "seed" in g else 1
        if n_seed > 1:
            sd = g.groupby("seed")[metric].mean().std(ddof=1)
        else:
            sd = ps.std(ddof=1)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "mean": ps.mean(), "std": sd,
                     "ci_lo": lo, "ci_hi": hi, "n_slices": len(ps), "n_seeds": n_seed})
    return pd.DataFrame(rows)


def paired_vs(df: pd.DataFrame, metric: str, ref: str, methods: Sequence[str], geometry: str) -> pd.DataFrame:
    """Δ = method - ref per slice; two-sided Wilcoxon, Holm-corrected p within the comparison family."""
    d = per_slice(df[df.geometry == geometry], metric).pivot_table(index=["pid", "idx"], columns="method",
                                                                    values=metric)
    rows = []
    for m in methods:
        if m == ref or m not in d or ref not in d:
            continue
        both = d[[m, ref]].dropna()
        delta = (both[m] - both[ref]).to_numpy()
        p = _st.wilcoxon(delta).pvalue if len(delta) >= 6 and np.any(delta != 0) else np.nan
        lo, hi = bootstrap_ci(delta)
        rows.append({"geometry": geometry, "method": m, "ref": ref, "delta_mean": float(delta.mean()),
                     "delta_median": float(np.median(delta)), "ci_lo": lo, "ci_hi": hi, "p": p,
                     "win_rate": float((delta > 0).mean()), "n": len(delta), "deltas": delta})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p"].fillna(1.0).to_numpy())
        out["stars"] = [stars(p) for p in out["p_holm"]]
    return out
