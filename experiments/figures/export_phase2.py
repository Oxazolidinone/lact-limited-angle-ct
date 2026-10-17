"""Collect the phase-2 EVALUATION results (test split) into one compact folder for plotting and output comparison.

Run on the training machine after evaluation has finished (read-only), from the project root; scripts/vast/phase2.sh
calls it at the end:

    python experiments/figures/export_phase2.py --outputs outputs_joint --out outputs_joint/figdata

Outputs (the inputs of make_figures_phase2.py):
  slices.csv   exp, run_id, method, model, update, train, dataset, geometry, pid, idx, psnr, ssim, rmse_hu, ...
  images.npz   <geometry>/<train>/<method>/<pid>/<idx> and <geometry>/gt/<pid>/<idx> (float16) for illustrative slices
  params.json  per-stage parameters of each run (alphas, betas; RegFormer: weights or alphas)
  meta.json    mu_ref, mu_water, image size, run table (source, validation PSNR, training hours, GPU)
  runs/<exp>/<run_id>/{metrics.csv, summary.json}   training curves
  tables/      copies of tables C1-C3
  figures/     PNG figure set (make_figures_phase2.py)
  BAO_CAO_KET_QUA.md   automatic summary (Vietnamese): setup, tables C1-C3, figure list (numbers only, no commentary)

Method key: the regularizer name (resnet_ed, unet_ed, densenet_ed, regformer) for first-order runs; "+lbfgs" is added
for runs with the L-BFGS data step (trained from scratch like first-order runs), "+ctrl" for first-order runs that load
a checkpoint (train.init_from; the current plan has no such run); baselines are fbp, tv.
Training-data key: mayo | dl | joint; baselines: none. Finally the PNG figure set is drawn into <out>/figures (skip with
--no-figures); redraw the PDF/PNG thesis figures on a personal machine with make_figures_phase2.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lact.config import load_config_file
from lact.infra.io.hdf5_store import ImageStore
from lact.infra.stores.results_store import ResultsStore
from lact.infra.stores.run_store import RunStore
from lact.services.factories import load_trained
from lact.services.reporting.phase2 import prepare

KEEP = ["psnr", "ssim", "rmse", "rmse_hu", "psnr_body", "ssim_body", "dc_rel"]
TRAIN_KEY = {"Mayo": "mayo", "DeepLesion": "dl", "Mayo + DeepLesion": "joint", "-": "none"}
SECOND = ("lbfgs", "cg", "admm")


def method_of(row) -> str:
    """Method key of one result row (see the module docstring)."""
    if row.exp == "baseline":
        return str(row.regularizer)
    if row.update in SECOND:
        return f"{row.regularizer}+{row.update}"
    return f"{row.regularizer}+ctrl" if row.init_from else str(row.regularizer)


def load_slices(rs: ResultsStore, split: str) -> pd.DataFrame:
    raw = rs.load_slices(split=split)
    if raw.empty:
        return raw
    df = prepare(raw)
    df["model"] = df.regularizer
    df["method"] = [method_of(r) for r in df.itertuples()]
    df["train"] = df.train_data.map(lambda s: TRAIN_KEY.get(s, "other"))
    return df


def pick_slices(df: pd.DataFrame, keep: dict[str, int], max_patients: dict[str, int] | None = None
                ) -> list[tuple[str, int]]:
    """Illustrative slices, INDEPENDENT of the model: for each patient, the slices around the median FBP PSNR in the
    first geometry; the same slices for every geometry so scan ranges can be compared. ``max_patients``: datasets with
    many test patients (DeepLesion) keep only that many patients, evenly spaced by id."""
    fbp = df[df.method == "fbp"]
    if fbp.empty:
        fbp = df
    g0 = sorted(fbp.geometry.unique())[0]
    fbp = fbp[fbp.geometry == g0]
    allowed: set[str] = set()
    for ds, d in fbp.groupby("dataset"):
        pids = sorted(d.pid.unique())
        cap = (max_patients or {}).get(ds)
        if cap and len(pids) > cap:
            pids = [pids[i] for i in sorted(set(np.linspace(0, len(pids) - 1, cap).round().astype(int).tolist()))]
        allowed |= set(pids)
    out = []
    for (ds, pid), fp in fbp[fbp.pid.isin(allowed)].groupby(["dataset", "pid"]):
        order = fp.groupby("idx").psnr.mean().sort_values().index.to_numpy()
        n = min(int(keep.get(ds, 1)), len(order))
        qs = [0.5] if n == 1 else np.linspace(0.35, 0.65, n)
        out += [(pid, int(order[int(q * (len(order) - 1))])) for q in qs]
    return sorted(set(out))


def _tags(rs: ResultsStore, split: str) -> dict[tuple[str, str], str]:
    return {(str(i.get("run_id")), str(i.get("geometry_test"))): t for t, i in rs.index().items()
            if i.get("split") == split and i.get("geometry_train") == i.get("geometry_test")}


FIG_CAPTIONS = [
    ("fig_qual_cross_", "Ảnh tái tạo CHÉO bộ dữ liệu (mô hình train trên bộ kia): ảnh đầy đủ, vùng phóng to, |sai số|"),
    ("fig_qual_", "Ảnh tái tạo (mô hình train trên chính bộ này): ảnh đầy đủ, vùng phóng to, bản đồ |sai số|"),
    ("fig_psnr_", "PSNR theo lát của từng phương pháp"),
    ("fig_ssim_", "SSIM theo lát của từng phương pháp"),
    ("fig_lbfgs_delta_", "ΔPSNR theo lát: lượt L-BFGS trừ lượt train thường cùng mô hình (cả hai train từ đầu)"),
    ("fig_lbfgs_vs_ctrl_", "ΔPSNR theo lát: lượt L-BFGS trừ lượt đối chứng (train thêm, giữ bước dữ liệu cũ)"),
    ("fig_curves_", "PSNR xác thực theo epoch: lượt train thường và lượt L-BFGS"),
    ("fig_stage_", "Tham số theo giai đoạn (α_k, β_k)"),
]
DATASET_VI = {"mayo": "Mayo", "dl": "DeepLesion"}


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows += ["| " + " | ".join("-" if pd.isna(v) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows) + "\n"


def write_summary(out: Path, df: pd.DataFrame, meta: dict) -> Path:
    """Markdown summary of the results from tables C1-C3 and the figures already in ``out``. Records only numbers and
    how to read them; commentary on the results is written later, after reviewing the numbers."""
    lines = ["# Kết quả giai đoạn 2: hai bộ dữ liệu, bước dữ liệu L-BFGS", "",
             f"Tổng hợp tự động từ kết quả đánh giá trên tập {meta['split']} "
             "(`experiments/figures/export_phase2.py`). Mọi "
             "mô hình được đánh giá trên cùng tập kiểm thử của cả hai bộ dữ liệu; chỉ số của một bộ là trung bình theo "
             "lát trong từng bệnh nhân rồi trung bình giữa các bệnh nhân.", "",
             "## 1. Dữ liệu kiểm thử và các lượt", ""]
    test = df[df.exp == "baseline"] if (df.exp == "baseline").any() else df
    one = test[test.geometry == sorted(test.geometry.unique())[0]].drop_duplicates(["pid", "idx"])
    for ds, d in one.groupby("dataset"):
        lines.append(f"- {ds}: {len(d)} lát của {d.pid.nunique()} bệnh nhân.")
    n_by = pd.Series([r["exp"] for r in meta["runs"].values()]).value_counts() if meta["runs"] else pd.Series(dtype=int)
    lines.append("- Lượt đã đánh giá: " + ", ".join(f"`{e}` {n}" for e, n in n_by.sort_index().items()) + ".")
    lines.append("- Khoảng quét: " + ", ".join(sorted(df.geometry.unique())) + ".")
    titles = {"C1_datasets": "## 2. Chất lượng trên từng bộ dữ liệu kiểm thử (bảng C1)",
              "C2_second_order": "## 3. Lượt L-BFGS so với mốc, ghép cặp theo lát (bảng C2)",
              "C3_runs": "## 4. Các lượt huấn luyện (bảng C3)"}
    notes = {"C1_datasets": "Mỗi dòng một phương pháp. Dòng train trên bộ này có cả cột của bộ kia: đó là kết quả chéo "
                            "bộ dữ liệu.",
             "C2_second_order": "Hiệu = lượt L-BFGS trừ mốc trên cùng lát. Mốc \"train thường\" là lượt cùng mô hình, "
                                "cùng dữ liệu, cùng lr và số epoch; cả hai train từ đầu, chỉ khác bước dữ liệu.",
             "C3_runs": "Giờ train là thời gian thực của từng lượt khi nhiều lượt chạy chung một GPU."}
    for name, title in titles.items():
        f = out / "tables" / f"{name}.csv"
        lines += ["", title, ""]
        if f.exists():
            lines += [notes[name], "", _md_table(pd.read_csv(f))]
        else:
            lines.append("(chưa có bảng này)")
    lines += ["", "## 5. Hình", ""]
    figs = sorted((out / "figures").glob("*.png")) if (out / "figures").is_dir() else []
    if not figs:
        lines.append("(chưa có hình: chạy experiments/figures/make_figures_phase2.py)")
    for prefix, cap in FIG_CAPTIONS:
        for f in figs:
            if not f.name.startswith(prefix) or (prefix == "fig_qual_" and f.name.startswith("fig_qual_cross_")):
                continue
            rest = f.stem[len(prefix):].split("_")
            where = DATASET_VI.get(rest[0], rest[0]) + (f", {rest[1].lstrip('a')}°" if len(rest) > 1 else "")
            lines += [f"**{cap}** ({where})", "", f"![{f.stem}](figures/{f.name})", ""]
    path = out / "BAO_CAO_KET_QUA.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs_joint", help="phase-2 outputs directory (eval/, runs/, ...)")
    ap.add_argument("--images", default=None, help="images.h5 (default: taken from the config.yaml of a run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--keep-mayo", type=int, default=2, help="illustrative slices per Mayo patient")
    ap.add_argument("--keep-dl", type=int, default=1, help="illustrative slices per DeepLesion patient")
    ap.add_argument("--dl-patients", type=int, default=16, help="DeepLesion patients with illustrative slices")
    ap.add_argument("--no-figures", action="store_true", help="do not draw the figure set into <out>/figures")
    a = ap.parse_args()
    root, out = Path(a.outputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rs = ResultsStore(root / "eval")
    df = load_slices(rs, a.split)
    if df.empty:
        raise SystemExit("No evaluation results yet")
    cols = ["exp", "run_id", "method", "model", "update", "train", "dataset", "geometry", "seed", "pid", "idx",
            "init_from", *[c for c in KEEP if c in df]]
    df[cols].to_csv(out / "slices.csv", index=False)

    cfg_file = next((f for f in sorted((root / "runs").glob("*/*/config.yaml")) if f.parent.parent.name != "e3_main"),
                    None) or next(iter(sorted((root / "runs").glob("*/*/config.yaml"))))
    base_cfg = load_config_file(cfg_file)
    images_h5 = Path(a.images) if a.images else root / "data" / "images.h5"
    img = ImageStore(images_h5 if images_h5.exists() else base_cfg.data.images_h5)

    # ---- illustrative images
    tags = _tags(rs, a.split)
    picks = pick_slices(df, {"Mayo": a.keep_mayo, "DeepLesion": a.keep_dl}, {"DeepLesion": a.dl_patients})
    images: dict[str, np.ndarray] = {}
    for g, dg in df.groupby("geometry"):
        for pid, idx in picks:
            images[f"{g}/gt/{pid}/{idx}"] = img.read(pid, "img256", idx).astype(np.float16)
        for (run_id, method, train), _ in dg.groupby(["run_id", "method", "train"]):
            tag = tags.get((str(run_id), str(g)))
            if tag is None:
                continue
            for pid, idx in picks:
                p = rs.load_pred(tag, pid, idx)
                if p is not None:
                    images[f"{g}/{train}/{method}/{pid}/{idx}"] = p.astype(np.float16)
    np.savez_compressed(out / "images.npz", **images)

    # ---- per-stage parameters, run table, training curves
    params: dict = {}
    meta: dict = {"mu_ref": img.mu_ref, "mu_water": img.mu_water, "split": a.split,
                  "img_size": int(base_cfg.data.img_size_rec), "picks": [list(p) for p in picks], "runs": {}}
    learned = df[df.exp != "baseline"].drop_duplicates("run_id")
    for r in learned.itertuples():
        run = RunStore(root / "runs" / r.exp / r.run_id)
        if not run.config_file.exists():
            continue
        s = run.summary()
        meta["runs"][r.run_id] = {
            "exp": r.exp, "method": r.method, "model": r.model, "update": r.update, "train": r.train,
            "geometry": r.geometry, "init_from": r.init_from or None,
            **{k: s.get(k) for k in ("best_val_psnr", "best_epoch", "train_seconds", "n_params", "lr", "gpus",
                                     "jobs_per_gpu", "peak_train_mem_mb", "status")}}
        try:
            model, _, _ = load_trained(run, "best" if run.checkpoint("best").exists() else "last", "cpu")
            params[r.run_id] = {k: [float(x) for x in v.detach().flatten()] for k, v in model.stage_params().items()}
        except Exception as e:
            print(f"(skipping parameters of {r.run_id}: {e})")
        dst = out / "runs" / r.exp / r.run_id
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.csv", "summary.json"):
            if (run.dir / name).exists():
                shutil.copy2(run.dir / name, dst / name)
    (out / "params.json").write_text(json.dumps(params, indent=1))
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "tables").mkdir(exist_ok=True)
    for f in sorted((root / "tables").glob("C*.*")):
        shutil.copy2(f, out / "tables" / f.name)
    print(f"DONE {out}: {len(df)} rows, {len(images)} images of {len(picks)} slices, {len(meta['runs'])} runs")

    if not a.no_figures:
        cmd = [sys.executable, str(Path(__file__).with_name("make_figures_phase2.py")), "--data", str(out),
               "--out", str(out / "figures"), "--lang", "vi", "--preset", "thesis", "--formats", "png"]
        print("drawing figures:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    print("summary:", write_summary(out, df, meta))


if __name__ == "__main__":
    main()
