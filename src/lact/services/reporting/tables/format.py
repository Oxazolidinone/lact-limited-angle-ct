"""Report settings and the formatting shared by the result tables (mean ± std cells, geometry and
architecture order, wide tables)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ....viz.style import ARCH_ORDER, geom_label, label_of

METRIC_FMT = {"psnr": ("PSNR (dB)", 2), "ssim": ("SSIM", 4), "rmse": ("RMSE", 4), "rmse_hu": ("RMSE (HU)", 1),
              "dc_rel": ("Sai số chiếu", 4), "psnr_body": ("PSNR cơ thể", 2)}


@dataclass
class ReportSpec:
    """Experiment and geometry names used for reporting."""
    split: str = "test"
    main_exp: str = "e3_main"
    matched_exp: str = "e4_matched"
    a150_exp: str = "e5_a150"
    sweep_exp: str = "e2_sweep"
    geoms: list[str] = field(default_factory=lambda: ["fan_a120", "fan_a90"])
    n_boot: int = 10000
    # one run reported per configuration: best_val = seed with the best validation PSNR | first = smallest seed | all
    seed_policy: str = "best_val"

    @property
    def trained_exps(self) -> list[str]:
        return [self.main_exp, self.matched_exp, self.a150_exp]


def fmt_ms(row, m: str, digits: int) -> str:
    mean, std = row.get(f"{m}_mean", np.nan), row.get(f"{m}_std", np.nan)
    if not np.isfinite(mean):
        return "–"
    return f"{mean:.{digits}f}" + (f" ± {std:.{digits}f}" if np.isfinite(std) else "")


def sort_geoms(names) -> list[str]:
    """Sort geometries by decreasing scan range (150°, 120°, 90°)."""
    def ang(g):
        lab = geom_label(g).rstrip("°")
        return -float(lab) if lab.replace(".", "").isdigit() else 0.0
    return sorted(names, key=ang)


def arch_rank(reg: str) -> int:
    return ARCH_ORDER.index(reg) if reg in ARCH_ORDER else 99


def _arch_keys(pm: pd.DataFrame, by: str = "arch") -> list[str]:
    return sorted(pm[by].unique(), key=lambda a: (arch_rank(pm[pm[by] == a].regularizer.iloc[0]), a))


def wide_table(pm: pd.DataFrame, geoms: list[str], metrics=("psnr", "ssim", "rmse"), by: str = "arch") -> pd.DataFrame:
    rows = []
    n_per_reg = pm.groupby("regularizer")[by].nunique()
    for a in _arch_keys(pm, by):
        d = pm[pm[by] == a]
        reg = d.regularizer.iloc[0]
        r: dict[str, object] = {"Phương pháp": label_of(reg) if n_per_reg[reg] == 1 else f"{label_of(reg)} ({a})"}
        for g in geoms:
            dg = d[d.geometry_test == g]
            for m in metrics:
                r[f"{METRIC_FMT[m][0]} {geom_label(g)}"] = fmt_ms(dg.iloc[0], m, METRIC_FMT[m][1]) if len(dg) else "–"
        r["Seed"] = int(d.n_seeds.max())
        rows.append(r)
    return pd.DataFrame(rows)
