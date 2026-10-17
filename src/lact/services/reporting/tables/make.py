"""Load the evaluation store, choose the learning rate of each architecture and write every result table
(CSV + Markdown + LaTeX)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ....config import Config
from ....infra.io.files import load_json, save_json
from ....infra.stores.run_store import RunStore, list_runs
from ....viz.tables import write_table
from ...factories import results_store
from ..aggregate import diverged_lrs, select_best_lr
from ..seeds import keep_chosen, seed_selection
from .cost import table_cost, table_regnet_cost
from .format import ReportSpec
from .results import (
    table_a150,
    table_cross_geometry,
    table_e0,
    table_hyperparams,
    table_main,
    table_matched,
    table_per_patient,
    table_seed_spread,
    table_seeds,
    table_trend,
)
from .significance import registered_tests, supplementary_tests, table_stats

log = logging.getLogger("lact")


def load_selected(cfg: Config, spec: ReportSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-slice results after applying the seed-selection convention, seed-selection table)."""
    df = results_store(cfg).load_slices(split=spec.split)
    selection = seed_selection(cfg.paths.runs, spec.trained_exps, spec.seed_policy)
    return keep_chosen(df, selection, spec.trained_exps), selection


def load_costs(cfg: Config) -> pd.DataFrame:
    files = sorted(cfg.paths.tables.glob("cost_*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def select_lr(cfg: Config, sweep_exp: str = "e2_sweep", runs: list[RunStore] | None = None) -> dict:
    """E2: choose the best lr -> <runs>/<sweep_exp>/best_lr.json and the E2_lr_sweep table.

    An lr that made an E3/E4/E5 run diverge is excluded; that architecture uses its best remaining lr for EVERY
    geometry."""
    main = [r for exp in ReportSpec().trained_exps for r in list_runs(cfg.paths.runs, exp)]
    res = select_best_lr(list_runs(cfg.paths.runs, sweep_exp) if runs is None else runs, diverged_lrs(main))
    save_json(res, cfg.paths.runs / sweep_exp / "best_lr.json")
    name = "E2_lr_sweep" if sweep_exp == "e2_sweep" else f"{sweep_exp}_lr"
    write_table(pd.DataFrame(res["table"]), cfg.paths.tables, name, "Quét tốc độ học (PSNR xác thực tốt nhất)")
    log.info("best lr: %s %s", res["best_lr"], res["notes"])
    return res


def make_tables(cfg: Config, spec: ReportSpec) -> list[Path]:
    out = cfg.paths.tables
    raw = results_store(cfg).load_slices(split=spec.split)
    if raw.empty:
        raise FileNotFoundError("No evaluation results yet: run `lact eval runs` / `lact eval baselines`")
    selection = seed_selection(cfg.paths.runs, spec.trained_exps, spec.seed_policy)
    df = keep_chosen(raw, selection, spec.trained_exps)
    how = ("trung bình ± độ lệch chuẩn qua seed" if spec.seed_policy == "all"
           else "mỗi cấu hình một lượt huấn luyện; xem B11 nếu có nhiều seed")
    tables: list[tuple[str, pd.DataFrame | None, str]] = [
        ("B1_main", table_main(df, spec), f"Chất lượng ảnh trên tập {spec.split} ({how})"),
        ("B2_per_patient", table_per_patient(df, spec), "Kết quả theo từng bệnh nhân"),
        ("B3_cost", table_cost(load_costs(cfg)), "Chi phí tính toán (batch 1, cùng GPU)"),
        ("B4_matched", table_matched(df, spec),
         "Nhóm cân bằng tham số (~2M) ở 120°: bản thu nhỏ của E4 và các kiến trúc lấy từ E3"),
        ("B5_a150", table_a150(df, spec), "Thí nghiệm bổ sung 150°"),
        ("B5_trend", table_trend(df, spec), "PSNR (dB) theo khoảng quét"),
        ("B6_cross_geometry", table_cross_geometry(df, spec), "PSNR (dB) khi suy luận trên hình học chưa gặp"),
    ]
    chk = cfg.paths.checks
    if (chk / "e0_report.json").exists():
        tables.append(("B7_e0_checks", table_e0(load_json(chk / "e0_report.json")),
                       "Kiểm tra toán tử và mô phỏng (E0)"))
    if (chk / "e1_angle_density.csv").exists():
        tables.append(("B7_e1_angle_density", pd.read_csv(chk / "e1_angle_density.csv").round(3),
                       "Kiểm tra mật độ góc 0,25° so với 0,125° (E1)"))
    tables += [("B8_stats", table_stats(registered_tests(df, spec)), "Kiểm định các so sánh đã đăng ký (PSNR)"),
               ("B9_hyperparams", table_hyperparams(cfg, spec), "Siêu tham số và epoch được chọn"),
               ("B10_supplementary", table_stats(supplementary_tests(df, spec)),
                "So sánh bổ sung: backbone mới và bản tích hợp RegNet (PSNR; ngoài họ đã đăng ký, Holm riêng)"),
               ("B11_seeds", table_seeds(selection),
                "Cấu hình có nhiều seed đã huấn luyện: seed dùng để báo cáo được chọn theo PSNR XÁC THỰC"),
               ("B13_regnet_cost", table_regnet_cost(df, load_costs(cfg), spec),
                "Tích hợp RegNet: chất lượng và tài nguyên so với bản gốc của từng họ (gốc → bản RegNet)"),
               ("B12_seed_spread", table_seed_spread(raw, spec),
                f"Độ lệch chuẩn qua seed trên tập {spec.split} (các cấu hình có từ hai seed): mức nhiễu do khởi tạo")]
    written = [write_table(t, out, name, caption) for name, t, caption in tables if t is not None and len(t)]
    log.info("Wrote %d tables to %s", len(written), out)
    return written
