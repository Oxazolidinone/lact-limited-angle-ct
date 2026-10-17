"""Result tables built from per-slice evaluation results: main table, per patient, parameter-matched group,
150° scan, trend over angles, cross-geometry, operator check, hyper-parameters, seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ....config import Config
from ....infra.io.files import load_json
from ....viz.style import geom_label, label_of
from ..aggregate import fmt_mean_std, per_method, per_method_patient
from .format import ReportSpec, _arch_keys, arch_rank, fmt_ms, sort_geoms, wide_table


def _same_geom(df: pd.DataFrame) -> pd.Series:
    return df.geometry_train == df.geometry_test


def main_slices(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    return df[_same_geom(df) & df.exp.isin([spec.main_exp, "baseline"]) & df.geometry_test.isin(spec.geoms)]


def table_main(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    return wide_table(per_method(main_slices(df, spec)), spec.geoms)


def table_per_patient(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame:
    pp = per_method_patient(main_slices(df, spec))
    return pd.DataFrame([{"Hình học": geom_label(r.geometry_test), "Bệnh nhân": r.pid,
                          "Phương pháp": label_of(r.regularizer), "PSNR (dB)": fmt_ms(r, "psnr", 2),
                          "SSIM": fmt_ms(r, "ssim", 4), "RMSE (HU)": fmt_ms(r, "rmse_hu", 1)}
                         for _, r in pp.sort_values(["geometry_test", "pid"]).iterrows()])


# Architectures with a dedicated ~2M scaled-down version in E4 (their full versions stay out of the parameter-matched
# table); LEARN–CNN (0.06M) is outside the group. Every other architecture enters with its E3 run at the same geometry.
MATCHED_BY_SHRINKING = ("resnet_ed", "vit", "regnet_ed")


MATCHED_TARGET = 2.0e6


def table_matched(df: pd.DataFrame, spec: ReportSpec, geom: str = "fan_a120") -> pd.DataFrame | None:
    """~2M-parameter group: the E4 scaled-down versions + architectures taken directly from E3. The "So với 2M" column
    shows the size deviation for the reader to weigh (backbones added later have no scaled-down version)."""
    from_main = (df.exp == spec.main_exp) & ~df.regularizer.isin(["cnn3", *MATCHED_BY_SHRINKING])
    d = df[_same_geom(df) & (df.geometry_test == geom) & ((df.exp == spec.matched_exp) | from_main)]
    if d.empty:
        return None
    pm = per_method(d)
    t = wide_table(pm, [geom])
    npar = d.groupby("arch").n_params.first()
    n = [float(npar.get(a, np.nan)) for a in _arch_keys(pm)]
    t.insert(1, "Tham số (M)", [round(v / 1e6, 3) for v in n])
    t.insert(2, "So với 2M", [f"{(v / MATCHED_TARGET - 1) * 100:+.0f}%" if np.isfinite(v) else "–" for v in n])
    t.insert(3, "Nguồn", ["E4 (thu nhỏ)" if a.endswith("_pm2m") else "E3" for a in _arch_keys(pm)])
    return t


def table_a150(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame | None:
    d = df[_same_geom(df) & df.exp.isin([spec.a150_exp, "baseline"]) & (df.geometry_test == "fan_a150")]
    return wide_table(per_method(d), ["fan_a150"]) if len(d) else None


def table_trend(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame | None:
    d = df[_same_geom(df) & df.exp.isin([spec.main_exp, spec.a150_exp, "baseline"])]
    if d.empty:
        return None
    trend = per_method(d).pivot_table(index="regularizer", columns="geometry_test", values="psnr_mean")
    trend = trend.reindex(sorted(trend.index, key=arch_rank)).round(2)
    trend = trend[sort_geoms(trend.columns)]
    trend.columns = [geom_label(c) for c in trend.columns]
    trend.insert(0, "Phương pháp", [label_of(i) for i in trend.index])
    return trend.reset_index(drop=True)


def table_cross_geometry(df: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame | None:
    d = df[df.exp == spec.main_exp]
    if not (d.geometry_train != d.geometry_test).any():
        return None
    pm = per_method(d)
    rows = []
    for reg in sorted(pm.regularizer.unique(), key=arch_rank):
        for gtr in sort_geoms(pm.geometry_train.unique()):
            r = {"Phương pháp": label_of(reg), "Huấn luyện": geom_label(gtr)}
            for gte in sort_geoms(pm.geometry_test.unique()):
                v = pm[(pm.regularizer == reg) & (pm.geometry_train == gtr) & (pm.geometry_test == gte)]
                r[f"Test {geom_label(gte)}"] = fmt_ms(v.iloc[0], "psnr", 2) if len(v) else "–"
            rows.append(r)
    return pd.DataFrame(rows)


def table_e0(report: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Hình học": k, "Liên hợp": f"{v['adjoint_rel']:.1e}", "Gradient": f"{v['grad_rel_fp32']:.1e}",
                          "FBP phantom (dB)": round(v["fbp_psnr"], 2), "Tỉ lệ TB": round(v["fbp_mean_ratio"], 3),
                          "Sàn DC": round(v["dc_floor_phantom"], 4), "Đạt": "có" if v["pass"] else "không"}
                         for k, v in report["results"].items()])


def table_hyperparams(cfg: Config, spec: ReportSpec) -> pd.DataFrame | None:
    rows = []
    blr = cfg.paths.runs / spec.sweep_exp / "best_lr.json"
    if blr.exists():
        rows += [{"Mục": f"lr {label_of(reg)}", "Giá trị": f"{lr:g}"} for reg, lr in load_json(blr)["best_lr"].items()]
    for f in sorted(cfg.paths.baselines.glob("tv_best_*.json")):
        t = load_json(f)
        rows.append({"Mục": f"TV {geom_label(t['geometry'])}", "Giá trị": f"lambda={t['lambda']:g}, {t['iters']} vòng"})
    for f in sorted(cfg.paths.runs.glob("*/*/summary.json")):
        s = load_json(f)
        rows.append({"Mục": f"epoch tốt nhất {s['run_id']}", "Giá trị": f"{s['best_epoch']} ({s['status']})"})
    return pd.DataFrame(rows) if rows else None


def table_seeds(selection: pd.DataFrame) -> pd.DataFrame | None:
    """Configurations with several trained seeds: validation PSNR per seed and the seed used for reporting."""
    multi = selection[selection.n_seeds > 1] if len(selection) else selection
    if multi.empty:
        return None
    rows = []
    for (exp, arch, geom), g in multi.groupby(["exp", "arch", "geometry"], sort=False):
        g = g.sort_values("seed")
        picked = g[g.chosen]
        rows.append({"Thí nghiệm": exp, "Mô hình": label_of(g.regularizer.iloc[0]) if exp != "e4_matched" else arch,
                     "Hình học": geom_label(geom),
                     "PSNR xác thực theo seed (dB)": ", ".join(f"s{int(r.seed)}: {r.best_val_psnr:.3f}"
                                                              for r in g.itertuples()),
                     "Seed báo cáo": (", ".join(str(int(x)) for x in picked.seed) if len(picked) < len(g)
                                      else "mọi seed")})
    return pd.DataFrame(rows)


def table_seed_spread(raw: pd.DataFrame, spec: ReportSpec) -> pd.DataFrame | None:
    """Standard deviation over seeds for configurations with at least two seeds: the initialisation-noise yardstick
    for comparing single-seed configurations (smaller differences say nothing about the architecture)."""
    d = raw[_same_geom(raw) & raw.exp.isin(spec.trained_exps)]
    if d.empty:
        return None
    pm = per_method(d)
    pm = pm[pm.n_seeds > 1]
    if pm.empty:
        return None
    rows = [{"Thí nghiệm": r.exp, "Mô hình": r.arch, "Hình học": geom_label(r.geometry_test),
             "Số seed": int(r.n_seeds), "PSNR (dB)": fmt_mean_std(r.psnr_mean, r.psnr_std, 2),
             "SSIM": fmt_mean_std(r.ssim_mean, r.ssim_std, 4)} for r in pm.itertuples()]
    out = pd.DataFrame(rows)
    out.loc[len(out)] = {"Thí nghiệm": "trung vị", "Mô hình": "", "Hình học": "", "Số seed": int(pm.n_seeds.median()),
                         "PSNR (dB)": f"± {pm.psnr_std.median():.2f}", "SSIM": f"± {pm.ssim_std.median():.4f}"}
    return out
