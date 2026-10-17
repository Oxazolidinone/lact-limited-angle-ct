"""Cost tables: parameters, FLOPs, time and memory per architecture, and the cost of the RegNet layouts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ....viz.style import REGNET_TWINS, geom_label, label_of
from ..aggregate import per_method
from .format import ReportSpec
from .results import _same_geom


def table_cost(cost: pd.DataFrame) -> pd.DataFrame | None:
    if cost.empty:
        return None
    c = cost
    return pd.DataFrame({
        "Phương pháp": c.regularizer.map(label_of) + " (" + c.arch + ")", "Hình học": c.geometry_test.map(geom_label),
        "Tham số (M)": (c.n_params / 1e6).round(3), "GFLOPs (K giai đoạn)": c.gflops_regularizer_total.round(1),
        "Thời gian trung vị (ms)": c.median_ms.round(1), "P95 (ms)": c.p95_ms.round(1),
        "Bộ nhớ suy luận (MB)": c.peak_infer_mem_mb.round(0) if "peak_infer_mem_mb" in c else np.nan,
        "Giờ huấn luyện": c.train_hours.round(2), "Bộ nhớ huấn luyện (MB)": c.peak_train_mem_mb,
        # proposal: costs must be measured on the same GPU; this column shows which runs trained on another GPU
        "GPU huấn luyện": c.train_gpus if "train_gpus" in c else np.nan,
        # > 1: the run shared its GPU with other runs, so "Giờ huấn luyện" is inflated and not directly comparable
        "Lượt chung 1 GPU": c.train_jobs_per_gpu if "train_jobs_per_gpu" in c else np.nan,
        "GPU đo suy luận": c.gpu if "gpu" in c else np.nan})


def _change(base: float, new: float, digits: int = 1) -> str:
    """"original → RegNet variant (±x%)"; "–" when a number is missing."""
    if not (np.isfinite(base) and np.isfinite(new)):
        return "–"
    pct = f" ({(new / base - 1) * 100:+.0f}%)" if base else ""
    return f"{base:.{digits}f} → {new:.{digits}f}{pct}"


def _jobs(a: float, b: float) -> str:
    return f"{int(a)}/{int(b)}" if np.isfinite(a) and np.isfinite(b) else "–"


def table_regnet_cost(df: pd.DataFrame, cost: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame | None:
    """Per family: original versus RegNet-integrated variant - quality AND resources (parameters, FLOPs, inference time
    and memory, training memory and hours). Costs are measured on the seed-0 E3 runs; independent of the seed."""
    if cost.empty:
        return None
    d = df[_same_geom(df) & (df.exp == spec.main_exp)]
    pm = per_method(d) if len(d) else pd.DataFrame(columns=["regularizer", "geometry_test", "psnr_mean"])
    c = cost[cost.exp == spec.main_exp] if "exp" in cost else cost
    rows = []
    for reg_new, reg_base in REGNET_TWINS:
        for g in spec.geoms:
            a = c[(c.regularizer == reg_base) & (c.geometry_test == g)]
            b = c[(c.regularizer == reg_new) & (c.geometry_test == g)]
            if a.empty or b.empty:
                continue
            a, b = a.iloc[0], b.iloc[0]
            get = lambda r, k: float(r[k]) if k in r and pd.notna(r[k]) else np.nan
            pa = pm[(pm.regularizer == reg_base) & (pm.geometry_test == g)].psnr_mean
            pb = pm[(pm.regularizer == reg_new) & (pm.geometry_test == g)].psnr_mean
            rows.append({
                "Họ": f"{label_of(reg_base)} → {label_of(reg_new)}", "Hình học": geom_label(g),
                "Δ PSNR test (dB)": f"{float(pb.iloc[0]) - float(pa.iloc[0]):+.2f}" if len(pa) and len(pb) else "–",
                "Tham số (M)": _change(get(a, "n_params") / 1e6, get(b, "n_params") / 1e6, 2),
                "GFLOPs (K giai đoạn)": _change(get(a, "gflops_regularizer_total"), get(b, "gflops_regularizer_total")),
                "Suy luận (ms/lát)": _change(get(a, "median_ms"), get(b, "median_ms")),
                "Bộ nhớ suy luận (MB)": _change(get(a, "peak_infer_mem_mb"), get(b, "peak_infer_mem_mb"), 0),
                "Bộ nhớ huấn luyện (MB)": _change(get(a, "peak_train_mem_mb"), get(b, "peak_train_mem_mb"), 0),
                # runs that shared a GPU with other runs have inflated training hours: for reference only
                "Giờ huấn luyện": _change(get(a, "train_hours"), get(b, "train_hours")),
                "Lượt chung 1 GPU (gốc/RegNet)": _jobs(get(a, "train_jobs_per_gpu"), get(b, "train_jobs_per_gpu"))})
    return pd.DataFrame(rows) if rows else None
