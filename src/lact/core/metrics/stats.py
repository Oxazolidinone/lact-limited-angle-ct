"""Statistical tests for paired comparisons between two methods.

- Per slice: average the metric over each method's seeds, then take the difference (A - B).
- Wilcoxon signed-rank per patient; with 6 or more patients, also a Wilcoxon test on the per-patient mean
  differences (the patient is the unit of analysis, robust to correlation between slices).
- Stratified bootstrap: resample seeds (with replacement, per method), then resample BLOCKS of adjacent slices within
  each patient (adjacent slices are correlated) -> confidence interval of the mean difference (macro over patients).
- Holm correction for a family of comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

MIN_PATIENTS_FOR_TEST = 6     # a two-sided Wilcoxon test with n < 6 cannot reach p < 0.05


def _aligned(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> tuple[dict, dict]:
    """{pid: array (n_seed, n_slice)} for A and B on the same slices (each slice must exist for every seed)."""
    A, B = {}, {}
    for pid in sorted(set(df_a.pid) & set(df_b.pid)):
        pa = df_a[df_a.pid == pid].pivot_table(index="seed", columns="idx", values=metric).dropna(axis=1)
        pb = df_b[df_b.pid == pid].pivot_table(index="seed", columns="idx", values=metric).dropna(axis=1)
        cols = sorted(set(pa.columns) & set(pb.columns))
        if cols:
            A[pid], B[pid] = pa[cols].to_numpy(), pb[cols].to_numpy()
    if not A:
        raise ValueError("The two methods share no slices")
    return A, B


def _nanmax(v) -> float:
    a = np.asarray(v, float)
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


def _block_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), nb)
    return np.concatenate([np.arange(s, min(n, s + block)) for s in starts])[:n]


def _bootstrap(A: dict, B: dict, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    boots = np.empty(n_boot)
    for r in range(n_boot):
        vals = []
        for pid in A:
            a, b = A[pid], B[pid]
            d = a[rng.integers(0, a.shape[0], a.shape[0])].mean(0) - b[rng.integers(0, b.shape[0], b.shape[0])].mean(0)
            vals.append(d[_block_index(d.size, block, rng)].mean())
        boots[r] = np.mean(vals)
    return boots


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "psnr", n_boot: int = 10000,
            block: int = 10, seed: int = 0, higher_is_better: bool = True, return_boot: bool = False) -> dict:
    """Compare A with B. df_* have columns pid, idx, seed, <metric> (one method per DataFrame)."""
    A, B = _aligned(df_a, df_b, metric)
    per_pat = {}
    for pid in A:
        d = A[pid].mean(0) - B[pid].mean(0)
        w = sps.wilcoxon(d) if (np.any(d != 0) and d.size > 5) else None
        better = (d > 0) if higher_is_better else (d < 0)
        per_pat[pid] = {"delta": float(d.mean()), "n": int(d.size), "pct_improved": float(better.mean() * 100),
                        "p_wilcoxon": float(w.pvalue) if w is not None else float("nan")}
    delta = float(np.mean([v["delta"] for v in per_pat.values()]))
    pat_deltas = np.array([v["delta"] for v in per_pat.values()])
    p_patients = (float(sps.wilcoxon(pat_deltas).pvalue) if pat_deltas.size >= MIN_PATIENTS_FOR_TEST
                  and np.any(pat_deltas != 0) else float("nan"))
    boots = _bootstrap(A, B, n_boot, block, np.random.default_rng(seed))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    sstd = lambda M: float(np.mean([m.mean(1).std(ddof=1) for m in M.values()])) if all(
        m.shape[0] > 1 for m in M.values()) else float("nan")
    sign = (lambda v: v > 0) if higher_is_better else (lambda v: v < 0)
    out = {"metric": metric, "delta": delta, "ci_low": float(lo), "ci_high": float(hi),
           "ci_excludes_zero": bool(lo > 0 or hi < 0),
           "all_patients_same_sign": bool(all(sign(v["delta"]) for v in per_pat.values())),
           "all_patients_opposite_sign": bool(all(not sign(v["delta"]) and v["delta"] != 0 for v in per_pat.values())),
           "seed_std_a": sstd(A), "seed_std_b": sstd(B), "per_patient": per_pat,
           "p_wilcoxon_max": _nanmax([v["p_wilcoxon"] for v in per_pat.values()]),
           "n_patients": int(pat_deltas.size), "n_patients_improved": int(sign(pat_deltas).sum()),
           "p_wilcoxon_patients": p_patients}
    if return_boot:
        out["boot"] = boots
    return out


def interaction(df_a1, df_b1, df_a2, df_b2, metric: str = "psnr", n_boot: int = 10000, block: int = 10,
                seed: int = 0) -> dict:
    """(A - B) under condition 1 minus (A - B) under condition 2, e.g. 120° versus 90° (RQ2).

    The two conditions are independent training runs, so their bootstraps are drawn independently and subtracted.
    """
    c1 = compare(df_a1, df_b1, metric, n_boot, block, seed, return_boot=True)
    c2 = compare(df_a2, df_b2, metric, n_boot, block, seed + 1, return_boot=True)
    diff = c1.pop("boot") - c2.pop("boot")
    lo, hi = np.percentile(diff, [2.5, 97.5])
    d = c1["delta"] - c2["delta"]
    return {"metric": metric, "delta_1": c1["delta"], "delta_2": c2["delta"], "interaction": d,
            "ci_low": float(lo), "ci_high": float(hi), "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni correction; NaN values are left as they are."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def decide_improvement(res: dict, direction: int = 1) -> bool:
    """Pre-registered rule: Δ > 0, the 95% CI excludes 0, the same sign for every patient, Δ larger than the seed SD.

    direction = -1 tests the opposite direction (B better than A). The returned labels are report content.
    """
    seeds = [v for v in (res.get("seed_std_a"), res.get("seed_std_b")) if v is not None and np.isfinite(v)]
    d = direction * res["delta"]
    lo = res["ci_low"] if direction > 0 else -res["ci_high"]
    same = res["all_patients_same_sign"] if direction > 0 else res.get("all_patients_opposite_sign", False)
    big = (not seeds) or d > max(seeds)
    return bool(d > 0 and lo > 0 and same and big)


def verdict(res: dict) -> str:
    if decide_improvement(res, 1):
        return "A tốt hơn"
    if decide_improvement(res, -1):
        return "B tốt hơn"
    return "chưa đủ bằng chứng"
