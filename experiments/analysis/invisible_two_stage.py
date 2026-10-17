r"""Two-stage reconstruction after Bubba et al. ("Learning the Invisible", arXiv:1811.04602): one model handles only
the VISIBLE part, a second model only the INVISIBLE part, and the two are merged. Both train from scratch.

Images are split by the project's visibility operator (lact.core.analysis.visibility): x = P_vis x + P_inv x, where
P_vis keeps the edge directions the limited-angle geometry measures at each pixel.

    Stage 1  the reconstruction model of the ``--spec`` grid (default RegFormer on Mayo, the base_mayo setting),
             trained from scratch with a loss ONLY on the visible part:      || P_vis(x1 - x) ||^2
             (the best checkpoint is chosen by the validation PSNR of the visible part)
    Stage 2  network G (U-Net) takes the REAL x1 images of the frozen stage 1, with a loss only on the invisible part:
                                                                         || P_inv(G(x1) - x) ||^2
    Merge    x = P_vis(x1) + P_inv(G(x1))

Difference from the paper: its stage 1 is an l1-shearlet solution (not learned), so stage-1 images are equally good on
the training and test splits; here stage 1 is learned, so training images look better than at test time and G has to
absorb that gap.

Stage-1 runs live in <outputs>/invisible/runs (NOT in <outputs>/runs: their validation PSNR is the visible-part PSNR
and must never mix with the tables and figures of first-order runs). Re-running the same command resumes stage 1 from
its checkpoint.

    python experiments/analysis/invisible_two_stage.py --geom fan_a120 \
        -c configs/env/server.yaml configs/variant/joint.yaml

Outputs in <outputs>/invisible/two_stage/<geometry>/: G.pt, slices.csv (per slice), summary.csv / summary.md
(Vietnamese report). Columns: "ghep" = merged image; "tang1" = unmerged stage-1 image (its hidden part was not trained,
for reference only); "tran" = hidden part from the true image (the ceiling of this merge); "thuong" = the first-order
run of the same model (--ref-exp).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lact.config import ExperimentSpec, load_config
from lact.core.analysis.visibility import VisibilityOperator
from lact.core.metrics import psnr, ssim
from lact.core.models import build_regularizer, count_params
from lact.infra.data.dataset import collate
from lact.infra.stores.run_store import list_runs
from lact.infra.system.runtime import apply_precision, seed_everything, select_device
from lact.services import scheduling
from lact.services.evaluation.evaluate import check_test_access
from lact.services.factories import dataset_for, load_trained, projector_for
from lact.services.training.trainer import Trainer, run_store_for


def vis_psnr(V: VisibilityOperator, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Visible-part PSNR per image (not clipped to [0, 1]: P_vis x has negative values)."""
    return 10.0 * torch.log10(1.0 / (V.vis(out - x) ** 2).flatten(1).mean(1).clamp_min(1e-20))


class VisibleTrainer(Trainer):
    """The project's Trainer with the loss and the checkpoint criterion switched to the visible part."""

    n_dirs = 16

    def setup(self) -> bool:
        ok = super().setup()
        if ok:
            self.V = VisibilityOperator(self.geom, self.cfg.data.img_size_rec, self.n_dirs, device=self.device)
            self.log.info("Loss and validation PSNR of this run are computed on the VISIBLE PART (%d directions)",
                          self.n_dirs)
        return ok

    def _step_once(self, b: dict, clip: float) -> tuple[str, float, float, float]:
        dev = self.device
        y, x0, x = b["y"].to(dev), b["x0"].to(dev), b["x"].to(dev)
        loss = torch.mean(self.V.vis(self.model(y, x0) - x) ** 2)
        lv = float(loss.detach())
        diverged, ref = self.guard.check(lv)
        if diverged:
            return "diverged", lv, float("nan"), ref
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
        self.opt.step()
        self.sched.step()
        return "ok", lv, gn, ref

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        dl = DataLoader(self.ds_val, batch_size=int(self.t.val_batch_size), shuffle=False, collate_fn=collate,
                        num_workers=0)
        acc: dict[str, list[torch.Tensor]] = {"val_psnr": [], "val_ssim": [], "val_rmse": [], "val_dc": [],
                                              "val_psnr_full": []}
        for b in dl:
            y, x0, x = b["y"].to(self.device), b["x0"].to(self.device), b["x"].to(self.device)
            out = self.model(y, x0)
            acc["val_psnr"].append(vis_psnr(self.V, out, x).cpu())          # checkpoint selection criterion
            acc["val_ssim"].append(ssim(out, x).cpu())                      # whole image, monitoring only
            acc["val_rmse"].append((self.V.vis(out - x) ** 2).flatten(1).mean(1).sqrt().cpu())
            acc["val_dc"].append(self.P.data_residual(out, y).cpu())
            acc["val_psnr_full"].append(psnr(out, x).cpu())                 # whole image, monitoring only
        return {k: float(torch.cat(v).mean()) for k, v in acc.items()}


@torch.no_grad()
def predict(model, cfg, split: str, device, limit: int | None = None) -> dict:
    """Images of the trained model on ``split``: {"xh", "x": (N, 1, n, n) on the CPU, "pid", "idx"}."""
    ds = dataset_for(cfg, split, cfg.require_geometry().name, preload=False, limit=limit, augment=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate, num_workers=0)
    xh, x, pid, idx = [], [], [], []
    for b in dl:
        xh.append(model(b["y"].to(device), b["x0"].to(device)).float().cpu())
        x.append(b["x"])
        pid += list(b["pid"])
        idx += [int(i) for i in b["idx"]]
    return {"xh": torch.cat(xh), "x": torch.cat(x), "pid": pid, "idx": idx}


def fuse(V: VisibilityOperator, net: torch.nn.Module, xh: torch.Tensor) -> torch.Tensor:
    return V.vis(xh) + V.inv(net(xh))


def train_g(V: VisibilityOperator, tr: dict, va: dict, a, device, log) -> torch.nn.Module:
    """Train G on stage-1 images of the training split; keep the weights with the best merged-image validation PSNR."""
    torch.manual_seed(a.seed)
    net = build_regularizer("unet_ed", channels=a.channels, depths=[2] * len(a.channels), dec_depth=1,
                            gn_groups=a.gn_groups).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    n = len(tr["x"])
    steps = a.epochs * max(1, n // a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=a.lr * 0.01)
    best, best_ep, best_state = -1.0, -1, None
    g = torch.Generator().manual_seed(a.seed)
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for i in range(0, max(1, n - a.batch + 1), a.batch):
            sel = perm[i:i + a.batch]
            xh, x = tr["xh"][sel].to(device), tr["x"][sel].to(device)
            loss = torch.mean(V.inv(net(xh) - x) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, nb = tot + float(loss.detach()), nb + 1
        net.eval()
        with torch.no_grad():
            vals = [psnr(fuse(V, net, va["xh"][i:i + 16].to(device)), va["x"][i:i + 16].to(device)).cpu()
                    for i in range(0, len(va["x"]), 16)]
        val = float(torch.cat(vals).mean())
        if val > best:
            best, best_ep = val, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep == 0 or (ep + 1) % max(1, a.epochs // 10) == 0 or ep == a.epochs - 1:
            log(f"    epoch {ep + 1:3d}/{a.epochs}: loss {tot / max(1, nb):.3e} | merged-image PSNR (validation) "
                f"{val:.2f} | "
                f"best {best:.2f} (epoch {best_ep + 1})")
    assert best_state is not None
    net.load_state_dict(best_state)
    return net.eval()


@torch.no_grad()
def split_mse(V: VisibilityOperator, out: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE of the image (clipped to [0, 1]) split into visible and invisible parts; both parts add up to the MSE."""
    e = out.clamp(0, 1) - x
    return (e * V.vis(e)).flatten(1).mean(1), (e * V.inv(e)).flatten(1).mean(1)


@torch.no_grad()
def score(V: VisibilityOperator, net, data: dict, ref: dict | None, split: str, device) -> list[dict]:
    rows = []
    for i in range(0, len(data["x"]), 16):
        xh, x = data["xh"][i:i + 16].to(device), data["x"][i:i + 16].to(device)
        outs = {"ghep": fuse(V, net, xh), "tang1": xh, "tran": V.vis(xh) + V.inv(x)}
        if ref is not None:
            outs["thuong"] = ref["xh"][i:i + 16].to(device)
        vals = {}
        for name, out in outs.items():
            vals[f"psnr_{name}"], vals[f"ssim_{name}"] = psnr(out, x), ssim(out, x)
            vals[f"mse_thay_{name}"], vals[f"mse_khuat_{name}"] = split_mse(V, out, x)
        for j in range(x.shape[0]):
            rows.append({"split": split, "pid": data["pid"][i + j], "idx": data["idx"][i + j],
                         **{k: float(v[j]) for k, v in vals.items()}})
    return rows


def find_ref(runs_root: Path, exp: str, arch: str, geom: str):
    """DONE first-order run of the same architecture and geometry (the reference); None when missing."""
    for run in list_runs(runs_root, exp):
        s = run.summary() if run.status == "done" else {}
        if (s.get("arch"), s.get("geometry")) == (arch, geom) and run.checkpoint("best").is_file():
            return run
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", nargs="+", required=True, help="environment / data-variant configuration")
    ap.add_argument("-s", "--set", nargs="*", default=[], help="configuration overrides, key=value")
    ap.add_argument("--spec", default="configs/experiment/base_mayo.yaml", help="grid of the first-order run")
    ap.add_argument("--exp", default="inv_mayo", help="experiment name of the stage-1 run")
    ap.add_argument("--arch", default="regformer", help="model.arch of stage 1")
    ap.add_argument("--geom", required=True, help="geometry, e.g. fan_a120")
    ap.add_argument("--ref-exp", default="base_mayo", help="experiment holding the first-order run to compare with")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--epochs", type=int, default=100, help="epochs of G")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--gn-groups", type=int, default=8)
    ap.add_argument("--n-dirs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N slices of each split in stage 2 "
                                                            "(debugging)")
    a = ap.parse_args()

    spec_exp = ExperimentSpec.load(a.spec).exp
    jobs = scheduling.expand(a.spec, a.config, a.set, models=[a.arch], exp_rename={spec_exp: a.exp})
    picked = [(j, load_config(j.files, j.overrides, use_defaults=False)) for j in jobs]
    picked = [(j, c) for j, c in picked if c.require_geometry().name == a.geom and int(c.seed) == a.seed]
    if len(picked) != 1:
        raise SystemExit(f"{a.spec}: exactly one run ({a.arch}, {a.geom}, seed {a.seed}) expected, found {len(picked)}")
    job, main_cfg = picked[0]
    root = main_cfg.paths.outputs / "invisible"
    cfg = load_config(job.files, [*job.overrides, f"paths.runs={root / 'runs'}"], use_defaults=False)
    out = root / "two_stage" / a.geom
    out.mkdir(parents=True, exist_ok=True)
    for s in a.splits:
        check_test_access(cfg, s)

    # ---------------------------------------------------------------- stage 1: visible part only
    VisibleTrainer.n_dirs = a.n_dirs
    status = VisibleTrainer(cfg).fit().get("status")
    run = run_store_for(cfg)
    if run.status != "done":
        raise SystemExit(f"Stage 1 ({run.run_id}) is not finished: {status}. Run the same command again to resume.")

    # ---------------------------------------------------------------- stage 2: invisible part
    lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    seed_everything(a.seed, cfg.precision.deterministic)
    apply_precision(cfg.precision.tf32)
    device = select_device(cfg.device)
    geom = cfg.require_geometry()
    V = VisibilityOperator(geom, cfg.data.img_size_rec, a.n_dirs, device=device)
    model, _, state = load_trained(run, "best", device)
    model.set_projector(projector_for(cfg, geom, device=device)).eval()
    log(f"== {a.exp} {a.arch} {a.geom}: stage 1 = {run.run_id}, best epoch {int(state.get('epoch', -1))}, "
        f"visible-part PSNR (validation) {float(run.summary().get('best_val_psnr', float('nan'))):.2f}")
    data = {s: predict(model, cfg, s, device, a.limit) for s in ["train", "val", *[s for s in a.splits if s != "val"]]}
    del model
    for s, d in data.items():
        v, i = (torch.cat(p) for p in zip(*(split_mse(V, d["xh"][k:k + 16].to(device), d["x"][k:k + 16].to(device))
                                             for k in range(0, len(d["x"]), 16)), strict=True))
        log(f"   stage-1 images, {s} split: {len(d['x'])} slices | visible-part MSE {float(v.mean()) * 1e3:.4f}e-3, "
            f"hidden part {float(i.mean()) * 1e3:.4f}e-3")
    t0 = time.time()
    log(f"   training G on {len(data['train']['x'])} stage-1 images, validating on {len(data['val']['x'])}")
    net = train_g(V, data["train"], data["val"], a, device, log)
    log(f"   G: {count_params(net) / 1e6:.2f}M parameters, trained in {time.time() - t0:.0f} s")
    torch.save({"model": net.state_dict(), "args": vars(a), "stage1": run.run_id}, out / "G.pt")

    ref_run = find_ref(main_cfg.paths.runs, a.ref_exp, a.arch, a.geom)
    if ref_run is None:
        log(f"   (no DONE first-order run of {a.arch} {a.geom} in {a.ref_exp}: dropping the 'thuong' column)")
        ref_model = None
    else:
        ref_model, _, _ = load_trained(ref_run, "best", device)
        ref_model.set_projector(projector_for(cfg, geom, device=device)).eval()
    rows: list[dict] = []
    for s in a.splits:
        ref = predict(ref_model, cfg, s, device, a.limit) if ref_model is not None else None
        rows += score(V, net, data[s], ref, s, device)
    df = pd.DataFrame(rows)
    df.to_csv(out / "slices.csv", index=False)
    num = [c for c in df.columns if c.startswith(("psnr_", "ssim_", "mse_"))]
    summ = df.groupby("split", sort=False)[num].mean().reset_index()
    summ.insert(1, "n_lat", df.groupby("split", sort=False).size().to_numpy())
    summ.insert(0, "geometry", a.geom)
    summ.to_csv(out / "summary.csv", index=False)
    names = [n for n in ("thuong", "ghep", "tran", "tang1") if f"psnr_{n}" in df.columns]
    label = {"thuong": f"Train thường ({a.ref_exp})", "ghep": "Hai tầng, ảnh ghép",
             "tran": "Trần (phần khuất = ảnh thật)", "tang1": "Tầng 1 chưa ghép"}
    md = [f"# Tái tạo hai tầng: {a.arch}, {a.geom}", "",
          f"Tầng 1: `{run.run_id}` (mất mát phần nhìn thấy). Tầng 2: U-Net {a.channels}, {a.epochs} epoch, lr {a.lr}, "
          f"batch {a.batch}. Mốc: {ref_run.run_id if ref_run else 'chưa có'}. MSE theo đơn vị 10^-3.", ""]
    for r in summ.itertuples(index=False):
        md += [f"## Tập {r.split} ({r.n_lat} lát)", "",
               "| Phương pháp | PSNR | SSIM | MSE phần thấy | MSE phần khuất |", "| --- | --- | --- | --- | --- |"]
        md += [f"| {label[n]} | {getattr(r, f'psnr_{n}'):.2f} | {getattr(r, f'ssim_{n}'):.4f} | "
               f"{getattr(r, f'mse_thay_{n}') * 1e3:.4f} | {getattr(r, f'mse_khuat_{n}') * 1e3:.4f} |" for n in names]
        md.append("")
        log(f"   {r.split} split: " + " | ".join(f"{n} {getattr(r, f'psnr_{n}'):.2f}" for n in names))
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    shutil.copy2(__file__, out / Path(__file__).name)          # the code that produced these results
    print(f"XONG {out}")


if __name__ == "__main__":
    main()
