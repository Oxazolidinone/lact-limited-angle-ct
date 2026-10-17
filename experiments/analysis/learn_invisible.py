"""Trial of "learning the invisible part" merged with a trained model (the idea of Bubba et al., "Learning the
Invisible", arXiv:1811.04602, applied to unrolled networks).

Images are split by the project's visibility operator (lact.core.analysis.visibility): x = P_vis x + P_inv x, where
P_vis keeps the edge directions that the limited-angle geometry MEASURES at each pixel. Trained models (LEARN,
RegFormer, ...) reconstruct the visible part very well; almost all remaining error lies in the invisible part. Here a
second network G handles only that part:

    v = P_vis(x_hat)                      visible part of the trained model's image
    x_merged = v + P_inv(G(v))            G predicts the invisible part FROM the visible part; it cannot alter measured
    parts

G trains only on true training images: input P_vis(x), target P_inv(x), loss ||P_inv(G(P_vis x) - x)||^2. No
projections are needed, nor the trained model's images on the training split (those look better than at test time
because the model has seen the training split, and would teach G the wrong thing).

    python experiments/analysis/learn_invisible.py -c configs/env/server.yaml configs/variant/joint.yaml \\
        configs/data/split_joint_mayo.yaml --exps e3_main --split val --out outputs_joint/invisible_pilot

Images, sinograms, matrices and the split come from the ``-c`` configuration (like ``lact eval runs
--on-config-data``); model and geometry come from each run. Only runs trained on the training dataset of that
configuration are accepted.

Outputs: slices.csv (per slice), summary.csv / summary.md (per run, Vietnamese report): PSNR of the original model,
of the merged image, and two references: "tran" = invisible part taken from the true image (the ceiling of any merge of
this kind), "G_tu_anh_that" = G applied to the visible part of the true image (what G achieves with a perfect visible
part).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lact.config import load_config, load_config_file
from lact.core.analysis.visibility import VisibilityOperator
from lact.core.metrics import psnr, ssim
from lact.core.models import build_regularizer, count_params
from lact.infra.data.dataset import collate
from lact.infra.io.hdf5_store import ImageStore
from lact.infra.stores.run_store import RunStore, list_runs
from lact.infra.system.runtime import apply_precision
from lact.services.evaluation.evaluate import check_test_access
from lact.services.factories import dataset_for, image_store, load_trained, projector_for
from lact.services.reporting.phase2 import TRAIN_DATA


def find_runs(dcfg, exps: list[str]) -> list[tuple[RunStore, object]]:
    """DONE runs of ``exps`` trained on the same dataset as the training split of ``dcfg``."""
    want = TRAIN_DATA.get(dcfg.split.name)
    out = []
    for run in (r for exp in exps for r in list_runs(dcfg.paths.runs, exp)):
        if run.status != "done" or not run.checkpoint("best").is_file():
            continue
        cfg = load_config_file(run.config_file)
        if want is not None and TRAIN_DATA.get(cfg.split.name) == want:
            out.append((run, cfg))
    return out


def read_images(store: ImageStore, pids: list[str], limit: int | None = None) -> torch.Tensor:
    """True 256 x 256 images of the patients: (N, 1, n, n) float32."""
    xs = []
    for pid in pids:
        xs.append(np.asarray(store.read(pid, "img256"), dtype=np.float32))
        if limit and sum(len(a) for a in xs) >= limit:
            break
    x = torch.from_numpy(np.concatenate(xs))[:, None]
    return x[:limit] if limit else x


def fuse(V: VisibilityOperator, net: torch.nn.Module, x_hat: torch.Tensor) -> torch.Tensor:
    v = V.vis(x_hat)
    return v + V.inv(net(v))


def train_g(V: VisibilityOperator, x_tr: torch.Tensor, x_va: torch.Tensor, a, device, log) -> torch.nn.Module:
    """Train G on true images; keep the weights with the best validation PSNR (merged image, visible part from the
    truth)."""
    torch.manual_seed(a.seed)
    net = build_regularizer("unet_ed", channels=a.channels, depths=[2] * len(a.channels), dec_depth=1,
                            gn_groups=a.gn_groups).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    steps = a.epochs * max(1, len(x_tr) // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=a.lr * 0.01)
    best, best_state = -1.0, None
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(x_tr), generator=g)
        tot = 0.0
        for i in range(0, len(perm) - a.batch + 1, a.batch):
            x = x_tr[perm[i:i + a.batch]].to(device)
            loss = torch.mean(V.inv(net(V.vis(x)) - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
        net.eval()
        with torch.no_grad():
            vals = [psnr(fuse(V, net, x_va[i:i + 16].to(device)), x_va[i:i + 16].to(device)).cpu()
                    for i in range(0, len(x_va), 16)]
        val = float(torch.cat(vals).mean())
        if val > best:
            best, best_state = val, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, a.epochs // 8) == 0 or ep == a.epochs - 1:
            log(f"    epoch {ep + 1:3d}/{a.epochs}: loss {tot / max(1, len(perm) // a.batch):.3e} | validation PSNR "
                f"(visible part from the truth) {val:.2f} | best {best:.2f}")
    assert best_state is not None
    net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def evaluate(run: RunStore, cfg, dcfg, V: VisibilityOperator, net: torch.nn.Module, split: str, device,
             limit) -> list[dict]:
    """Run the trained model on the ``split`` of ``dcfg`` and merge; returns one row per slice."""
    geom = cfg.require_geometry()
    model, _, _ = load_trained(run, "best", device)
    model.set_projector(projector_for(dcfg, geom, device=device)).eval()
    ds = dataset_for(dcfg, split, geom.name, preload=False, limit=limit, augment=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)
    m = cfg.require_model()
    rows = []
    for b in dl:
        y, x0, x = b["y"].to(device), b["x0"].to(device), b["x"].to(device)
        x_hat = model(y, x0)
        fused = fuse(V, net, x_hat)
        oracle = V.vis(x_hat) + V.inv(x)
        from_gt = fuse(V, net, x)
        e = x_hat.clamp(0, 1) - x
        ev, ei = (e * V.vis(e)).flatten(1).mean(1), (e * V.inv(e)).flatten(1).mean(1)
        ef = fused.clamp(0, 1) - x
        fv, fi = (ef * V.vis(ef)).flatten(1).mean(1), (ef * V.inv(ef)).flatten(1).mean(1)
        vals = {"psnr_goc": psnr(x_hat, x), "psnr_ghep": psnr(fused, x), "psnr_tran": psnr(oracle, x),
                "psnr_G_tu_anh_that": psnr(from_gt, x), "ssim_goc": ssim(x_hat, x), "ssim_ghep": ssim(fused, x),
                "mse_thay_goc": ev, "mse_khuat_goc": ei, "mse_thay_ghep": fv, "mse_khuat_ghep": fi}
        for i in range(x.shape[0]):
            rows.append({"exp": cfg.exp, "run_id": run.run_id, "model": m.regularizer.name, "update": m.update_tag,
                         "geometry": geom.name, "split": split, "pid": b["pid"][i], "idx": int(b["idx"][i]),
                         **{k: float(v[i]) for k, v in vals.items()}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", nargs="+", required=True, help="data configuration (images, sinograms, split)")
    ap.add_argument("-s", "--set", nargs="*", default=[], help="configuration overrides, key=value")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"], help="split to score (trying ideas: val)")
    ap.add_argument("--exps", nargs="+", required=True, help="experiments holding the trained runs")
    ap.add_argument("--geoms", nargs="*", default=[], help="only these geometries")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--gn-groups", type=int, default=8)
    ap.add_argument("--n-dirs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N slices (debugging)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dcfg = load_config(a.config, a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    check_test_access(dcfg, a.split)
    apply_precision(dcfg.precision.tf32)
    train_name = TRAIN_DATA.get(dcfg.split.name, dcfg.split.name)
    device = torch.device(a.device)
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    runs = find_runs(dcfg, a.exps)
    if a.geoms:
        runs = [(r, c) for r, c in runs if c.require_geometry().name in a.geoms]
    if not runs:
        raise SystemExit(f"No DONE run of {a.exps} trained on {train_name} in {dcfg.paths.runs}")
    store = image_store(dcfg)
    rows: list[dict] = []
    for geom_name in sorted({c.require_geometry().name for _, c in runs}):
        group = [(r, c) for r, c in runs if c.require_geometry().name == geom_name]
        V = VisibilityOperator(group[0][1].require_geometry(), dcfg.data.img_size_rec, a.n_dirs, device=device)
        x_tr = read_images(store, dcfg.split.patients("train"), a.limit)
        x_va = read_images(store, dcfg.split.patients("val"), a.limit)
        t0 = time.time()
        log(f"== {train_name} {geom_name}: training G on {len(x_tr)} true images, validating on {len(x_va)}")
        net = train_g(V, x_tr, x_va, a, device, log)
        log(f"   G: {count_params(net) / 1e6:.2f}M parameters, trained in {time.time() - t0:.0f} s")
        torch.save({"model": net.state_dict(), "args": vars(a)}, out / f"G_{geom_name}.pt")
        for run, c in group:
            r = evaluate(run, c, dcfg, V, net, a.split, device, a.limit)
            rows += r
            d = pd.DataFrame(r)
            log(f"   {run.run_id}: original PSNR {d.psnr_goc.mean():.2f} -> merged {d.psnr_ghep.mean():.2f} "
                f"({d.psnr_ghep.mean() - d.psnr_goc.mean():+.2f} dB) | ceiling {d.psnr_tran.mean():.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "slices.csv", index=False)
    num = [c for c in df.columns if c.startswith(("psnr_", "ssim_", "mse_"))]
    summ = df.groupby(["exp", "model", "update", "geometry", "split"], sort=False)[num].mean().reset_index()
    summ.insert(5, "n_lat", df.groupby(["exp", "model", "update", "geometry", "split"], sort=False).size().to_numpy())
    summ["delta_psnr"] = summ.psnr_ghep - summ.psnr_goc
    wins = (df.psnr_ghep > df.psnr_goc).groupby([df.exp, df.model, df["update"], df.geometry, df.split], sort=False)
    summ["ti_le_lat_tot_len"] = wins.mean().to_numpy()
    summ.to_csv(out / "summary.csv", index=False)
    show = summ[["exp", "model", "update", "geometry", "n_lat", "psnr_goc", "psnr_ghep", "delta_psnr",
                 "ti_le_lat_tot_len", "psnr_tran", "psnr_G_tu_anh_that", "mse_thay_goc", "mse_khuat_goc",
                 "mse_thay_ghep", "mse_khuat_ghep"]].copy()
    for c in ("mse_thay_goc", "mse_khuat_goc", "mse_thay_ghep", "mse_khuat_ghep"):
        show[c] = (show[c] * 1e3).round(3)
    md = ["# Thử nghiệm học phần không nhìn thấy", "",
          f"Bộ train: {train_name}; chấm trên tập {a.split}; G: U-Net {a.channels}, {a.epochs} epoch, lr {a.lr}.",
          "MSE ghi theo đơn vị 10^-3. `psnr_tran`: phần không nhìn thấy lấy từ ảnh thật (mức tối đa).", "",
          "| " + " | ".join(show.columns) + " |", "| " + " | ".join("---" for _ in show.columns) + " |"]
    md += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
           for r in show.itertuples(index=False)]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    print(f"DONE {out}: {len(df)} slices, {len(summ)} runs")


if __name__ == "__main__":
    main()
