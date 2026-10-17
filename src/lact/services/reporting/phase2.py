"""Phase-2 summary tables (docs/PHASE2.md): two datasets and the L-BFGS data step.

    C1_datasets      quality on EACH test dataset (Mayo, DeepLesion) by training data and update rule
    C2_second_order  L-BFGS run minus the first-order run whose checkpoint it loaded (and minus the extra-training
                     control run, when present)
    C3_runs          cost and status of each run (training hours, memory, validation PSNR, GPU model used)

A dataset's metric = per-slice mean within each patient, then the mean over that dataset's patients (the convention
of table B1). A slice's dataset follows from its patient id (DLxxxxxx = DeepLesion). Table labels and headers are
Vietnamese report content. The cg/admm rules (plan of 2026-10-08, no longer run) are still accepted when present.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Config
from ...datasets.deeplesion import is_patient_id
from ...infra.stores.run_store import list_runs
from ...viz.style import ARCH_ORDER, geom_label, label_of
from ...viz.tables import write_table
from ..factories import results_store

log = logging.getLogger("lact")

DATASETS = ("Mayo", "DeepLesion")
# split name used for training -> dataset learned from
TRAIN_DATA = {"split_7_1_2": "Mayo", "split_joint_mayo": "Mayo", "split_joint_dl": "DeepLesion",
              "split_joint": "Mayo + DeepLesion"}
TRAIN_ORDER = ["-", "Mayo", "DeepLesion", "Mayo + DeepLesion"]
UPDATE_LABEL = {("gd", ""): "LEARN (bậc một)", ("fbp", ""): "FBP của phần dư",
                ("cg", "identity"): "CG, B = I", ("cg", "sobolev"): "CG, Sobolev",
                ("admm", "identity"): "ADMM, B = I", ("admm", "sobolev"): "ADMM, Sobolev",
                ("lbfgs", ""): "L-BFGS", ("cg_fbp", ""): "CG trên FBP∘A", ("post", ""): "hậu xử lý ảnh FBP",
                ("pd", ""): "primal-dual", ("dual", ""): "hai miền + tinh chỉnh TV"}
SECOND_ORDER = ("cg", "admm", "lbfgs")
UPDATE_ORDER = list(UPDATE_LABEL.values())
KEYS = ["exp", "train_data", "update_label", "regularizer", "arch", "geometry"]


def dataset_of(pid: str) -> str:
    return "DeepLesion" if is_patient_id(pid) else "Mayo"


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the dataset, train_data, update_label and geometry columns; keep only same-geometry train/test results."""
    df = df[df.geometry_train == df.geometry_test].copy()
    for col in ("update", "metric", "train_split", "init_from"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    base = df.exp == "baseline"
    df["dataset"] = df.pid.map(dataset_of)
    df["geometry"] = df.geometry_test
    df["train_data"] = np.where(base, "-", df.train_split.map(lambda s: TRAIN_DATA.get(s, s or "?")))
    df["update_label"] = [("không học" if b else UPDATE_LABEL.get((u or "gd", m if u in SECOND_ORDER else ""),
                                                                 f"{u} {m}".strip()))
                          for b, u, m in zip(base, df["update"], df["metric"])]
    return df


def per_dataset(df: pd.DataFrame, metrics=("psnr", "ssim")) -> pd.DataFrame:
    """One row per (method, geometry, test dataset): mean over patients (and over runs when there are several seeds),
    with the patient count."""
    ms = [m for m in metrics if m in df.columns]
    pp = df.groupby([*KEYS, "run_id", "dataset", "pid"], dropna=False)[ms].mean().reset_index()
    pr = pp.groupby([*KEYS, "run_id", "dataset"], dropna=False)
    runs = pr[ms].mean()
    runs["n_patients"] = pr.size()
    runs = runs.reset_index()
    g = runs.groupby([*KEYS, "dataset"], dropna=False)
    out = g[ms].mean()
    out["n_patients"] = g["n_patients"].first()
    out["n_runs"] = g.size()
    return out.reset_index()


def _order(df: pd.DataFrame) -> pd.DataFrame:
    rank = lambda order: (lambda s: s.map(lambda v: order.index(v) if v in order else len(order)))
    df = df.assign(_t=rank(TRAIN_ORDER)(df.train_data), _u=rank(["không học", *UPDATE_ORDER])(df.update_label),
                   _a=rank(ARCH_ORDER)(df.regularizer))
    return df.sort_values(["_t", "_u", "_a", "exp"]).drop(columns=["_t", "_u", "_a"])


def table_datasets(df: pd.DataFrame, geoms: list[str]) -> pd.DataFrame | None:
    """C1: one row per method; columns = geometry x test dataset x {PSNR, SSIM}."""
    pd_ = per_dataset(df)
    if pd_.empty:
        return None
    rows = []
    for key, g in _order(pd_).groupby(["exp", "train_data", "update_label", "regularizer"], sort=False):
        exp, train, upd, reg = key
        row = {"Huấn luyện trên": train, "Cập nhật": upd, "Mô hình": label_of(reg),
               "Thí nghiệm": exp}
        for geom in geoms:
            for ds in DATASETS:
                hit = g[(g.geometry == geom) & (g.dataset == ds)]
                tag = f"{geom_label(geom)} {ds}"
                row[f"{tag} PSNR"] = f"{hit.psnr.iloc[0]:.2f}" if len(hit) else "-"
                row[f"{tag} SSIM"] = f"{hit.ssim.iloc[0]:.4f}" if len(hit) else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def _reference(df: pd.DataFrame, arch: str, geom: str, train_data: str, finetuned: bool) -> pd.DataFrame:
    """First-order reference run: same backbone, geometry and training data. A run fine-tuned from an existing run
    (``train.init_from``) is compared with a fine-tuned reference; a from-scratch run with a from-scratch reference."""
    ref = df[(df["update"].isin(["", "gd"])) & (df.exp != "baseline") & (df.arch == arch) & (df.geometry == geom)
             & (df.train_data == train_data)]
    return ref[(ref.init_from != "") == finetuned]


def _controls(df: pd.DataFrame, source: str) -> list[pd.DataFrame]:
    """Control runs of an L-BFGS run: they load the checkpoint of the same ``source`` run and train further in exactly
    the same way, but keep the old data step. The difference to them separates the L-BFGS gain from extra training."""
    c = df[(df.init_from == source) & ~df["update"].isin(SECOND_ORDER) & (df.exp != "baseline")]
    return [g for _, g in c.groupby("run_id", sort=False)]


def table_second_order(df: pd.DataFrame, geoms: list[str]) -> pd.DataFrame | None:
    """C2: PSNR/SSIM difference (L-BFGS, CG, ... run minus the reference run) on each test dataset, paired per slice
    and averaged per patient, with the number of patients that improve. One row per reference."""
    rows = []
    second = df[df["update"].isin(SECOND_ORDER) & (df.exp != "baseline")]
    for (exp, arch, geom), g in second.groupby(["exp", "arch", "geometry"], sort=False):
        if geom not in geoms:
            continue
        source = g.init_from.iloc[0]
        if g["update"].iloc[0] == "lbfgs" and source:
            # L-BFGS retrains from a first-order run's checkpoint: the reference is that source run itself
            refs = [("lượt nguồn", df[df.run_id == source])]
            refs += [("train thêm, giữ bước cũ", c) for c in _controls(df, source)]
        else:
            refs = [("train thường", _reference(df, arch, geom, g.train_data.iloc[0], bool(source)))]
        for kind, ref in refs:
            if ref.empty:
                continue
            row = {"Thí nghiệm": exp, "Cập nhật": g.update_label.iloc[0], "Mô hình": label_of(g.regularizer.iloc[0]),
                   "Hình học": geom_label(geom), "Huấn luyện trên": g.train_data.iloc[0], "Mốc": ref.exp.iloc[0],
                   "Loại mốc": kind}
            cols = ["pid", "idx"]
            a = g.groupby(cols)[["psnr", "ssim"]].mean()
            b = ref.groupby(cols)[["psnr", "ssim"]].mean()
            d = (a - b).dropna().reset_index()
            d["dataset"] = d.pid.map(dataset_of)
            for ds in DATASETS:
                pat = d[d.dataset == ds].groupby("pid")[["psnr", "ssim"]].mean()
                if pat.empty:
                    continue
                row[f"{ds} ΔPSNR (dB)"] = f"{pat.psnr.mean():+.2f}"
                row[f"{ds} ΔSSIM"] = f"{pat.ssim.mean():+.4f}"
                row[f"{ds} bệnh nhân tốt lên"] = f"{int((pat.psnr > 0).sum())}/{len(pat)}"
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def table_runs(cfg: Config, exps: list[str]) -> pd.DataFrame | None:
    """C3: one row per finished run of ``exps`` (from summary.json and metrics.csv)."""
    rows = []
    for exp in exps:
        for run in list_runs(cfg.paths.runs, exp):
            s = run.summary()
            if not s:
                continue
            cg = np.nan
            if run.metrics_csv.exists():
                m = pd.read_csv(run.metrics_csv)
                if "val_cg_iters" in m and m.val_cg_iters.notna().any():
                    cg = float(m.val_cg_iters.dropna().iloc[-1])
            upd = s.get("update") or "gd"
            mem = float(s.get("peak_train_mem_mb") or np.nan) / 1024.0
            rows.append({"Thí nghiệm": exp, "Mô hình": label_of(s.get("regularizer", "")),
                         "Hình học": geom_label(str(s.get("geometry"))),
                         "Cập nhật": UPDATE_LABEL.get((upd, (s.get("metric") or "") if upd in SECOND_ORDER else ""),
                                                      upd),
                         "Trạng thái": s.get("status"), "Khởi tạo từ": s.get("init_from") or "-",
                         "PSNR xác thực": round(float(s.get("best_val_psnr", np.nan)), 3),
                         "Epoch tốt nhất": s.get("best_epoch"),
                         "Giờ train": round(float(s.get("train_seconds", np.nan)) / 3600.0, 2),
                         # phase-2 runs used several GPU models: record it so training hours are never compared across
                         # machines
                         "GPU": ", ".join(sorted({str(g).replace("NVIDIA GeForce ", "") for g in s.get("gpus") or []}))
                         or "-",
                         "Lượt/GPU": s.get("jobs_per_gpu", 1),
                         "Bộ nhớ đỉnh (GB)": "-" if np.isnan(mem) else round(mem, 1),
                         "Vòng CG/giai đoạn": "-" if np.isnan(cg) else round(cg, 1)})
    if not rows:
        return None
    out = pd.DataFrame(rows)
    return out.drop(columns="Vòng CG/giai đoạn") if (out["Vòng CG/giai đoạn"] == "-").all() else out


def make_phase2_tables(cfg: Config, split: str = "test", geoms: list[str] | None = None,
                       exps: list[str] | None = None) -> list[Path]:
    """Write C1-C3 to ``paths.tables``. ``exps``: only these experiments (default: every result in the store)."""
    geoms = geoms or ["fan_a90", "fan_a120"]
    raw = results_store(cfg).load_slices(split=split)
    if raw.empty:
        raise FileNotFoundError("No evaluation results yet: run `lact eval runs` / `lact eval baselines`")
    df = prepare(raw)
    if exps:
        df = df[df.exp.isin([*exps, "baseline"])]
    run_exps = list(exps or [])
    if not run_exps and cfg.paths.runs.exists():
        run_exps = sorted(p.name for p in cfg.paths.runs.iterdir() if p.is_dir())
    tables = [("C1_datasets", table_datasets(df, geoms),
               f"Chất lượng trên từng bộ dữ liệu của tập {split} theo dữ liệu huấn luyện và quy tắc cập nhật"),
              ("C2_second_order", table_second_order(df, geoms),
               "Bước dữ liệu L-BFGS trừ mốc cùng mô hình, cùng hình học, cùng dữ liệu huấn luyện (ghép cặp theo lát)"),
              ("C3_runs", table_runs(cfg, run_exps), "Các lượt huấn luyện của giai đoạn 2: trạng thái và chi phí")]
    written = [write_table(t, cfg.paths.tables, name, caption) for name, t, caption in tables
               if t is not None and len(t)]
    log.info("Wrote %d tables to %s", len(written), cfg.paths.tables)
    return written
