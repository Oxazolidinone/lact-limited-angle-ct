"""Phase-2 configuration: the joint store, the splits, the base_* / lbfgs_* grids, update rules and new commands."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lact.cli.main import build_parser
from lact.config import ConfigError, ExperimentSpec, load_config, load_yaml
from lact.config.loader import save_yaml
from lact.datasets.deeplesion import is_patient_id, read_key_manifest
from lact.services import scheduling as experiments
from lact.services.data.prep import compute_mu_ref
from lact.services.scheduling import expand, filter_jobs, renamed

JOINT = ["configs/variant/joint.yaml"]
PLAN = ["base_mayo", "base_dl", "lbfgs_mayo", "lbfgs_dl"]
CMP = ["cmp_mayo", "cmp_dl"]          # comparison table with the CG-GLORE paper
DPMA_GRIDS = ["dpma1_mayo", "dpma1_dl", "dpma_mayo", "dpma_dl"]      # DPMA: stage 1 then stage 2


def _split(name):
    return load_yaml(f"configs/data/{name}.yaml")["split"]


def test_joint_splits_are_consistent():
    """DeepLesion as in QN-Mixer and CG-GLORE: 2,000 / 200 / 300 key slices drawn from the three official splits; the
    joint-store files derive from that slice list."""
    rows = read_key_manifest("configs/data/deeplesion_keyslices.csv")
    assert len(rows) == len({r["file_name"] for r in rows}) == 2500
    assert {k: sum(r["split"] == k for r in rows) for k in ("train", "val", "test")} == {"train": 2000, "val": 200,
                                                                                        "test": 300}
    assert all(r["patient"] == "DL" + r["file_name"][:6] and r["file_name"].endswith(".png") for r in rows)
    dl = {k: sorted({r["patient"] for r in rows if r["split"] == k}) for k in ("train", "val", "test")}
    assert not set(dl["train"]) & set(dl["val"]) and not set(dl["train"]) & set(dl["test"])      # patient-level split
    assert not set(dl["val"]) & set(dl["test"])
    mayo = _split("split_7_1_2")
    joint, joint_dl, joint_mayo = _split("split_joint"), _split("split_joint_dl"), _split("split_joint_mayo")
    for k in ("train", "val", "test"):
        assert joint[k] == mayo[k] + dl[k]
    assert joint_dl["train"] == dl["train"] and joint_dl["val"] == dl["val"]
    assert joint_mayo["train"] == mayo["train"] and joint_mayo["val"] == mayo["val"]
    assert joint["test"] == joint_dl["test"] == joint_mayo["test"]              # every model tests on the same patients
    assert sum(not is_patient_id(p) for p in joint["test"]) == 2
    patients = load_yaml("configs/data/joint.yaml")["data"]["patients"]         # qua _base_: joint_patients.yaml
    assert len(patients) == len(set(patients)) == 10 + sum(len(v) for v in dl.values())
    for sp in (joint, joint_dl, joint_mayo):
        assert set(sp["train"] + sp["val"] + sp["test"]) <= set(patients)
    assert not is_patient_id("L067") and is_patient_id("DL000003") and not is_patient_id("DL3")


def test_joint_variant_config():
    c = load_config([*JOINT, "configs/data/split_joint.yaml"])
    assert c.data.format == "mixed" and c.data.deeplesion_root.name == "deeplesion_key"
    assert c.data.norm.mode == "fixed" and c.data.norm.mu_ref == pytest.approx(0.03790080174803734)
    assert c.paths.runs.parts[-2:] == ("outputs_joint", "runs")
    assert c.paths.operators.parts[-2:] == ("outputs", "operators")
    assert c.split.name == "split_joint" and c.split.train[:7] == _split("split_7_1_2")["train"]
    assert len(c.split.train) > 1000 and c.split.val[0] == "L333"
    stats = compute_mu_ref(c)                                    # reads no data: mu_ref is given
    assert stats["mode"] == "fixed" and stats["mu_ref"] == c.data.norm.mu_ref
    assert stats["hu_at_mu_ref"] == pytest.approx(974.0, abs=0.01)
    with pytest.raises(ConfigError, match="mu_ref"):
        load_config([], ["data.norm.mode=fixed"])
    with pytest.raises(ConfigError, match=r"data\.format"):
        load_config([], ["data.format=nifti"])


def test_key_slice_selection_and_download(tmp_path, monkeypatch):
    """Key-slice draw: within the official splits, skipping noisy series and non-512 slices, reproducible for a seed;
    downloading unpacks exactly the drawn slices into the Images_png/<series>/<slice>.png layout."""
    import zipfile

    from lact.datasets import deeplesion as D
    from lact.datasets.deeplesion import download as D_download
    from lact.datasets.deeplesion import read_png_hu, write_png_hu

    lines = ["File_name,Possibly_noisy,Image_size,Train_Val_Test"]
    for p in range(1, 61):
        split = 1 if p <= 40 else 2 if p <= 50 else 3
        for k in range(3):
            # one noisy slice excludes the whole 000005_01_01 series
            noisy = int(p == 5 and k == 0)
            size = "768, 768" if p == 6 and k == 1 else "512, 512"
            lines.append(f"{p:06d}_01_01_{10 * k + 7:03d}.png,{noisy},\"{size}\",{split}")
    lines.append(lines[1])                                       # two lesions on one slice: counted once
    info = tmp_path / "DL_info.csv"
    info.write_text("\n".join(lines) + "\n")
    rows = D.select_key_slices(info, 30, 6, 9, seed=0)
    assert rows == D.select_key_slices(info, 30, 6, 9, seed=0) != D.select_key_slices(info, 30, 6, 9, seed=1)
    assert [sum(r["split"] == k for r in rows) for k in ("train", "val", "test")] == [30, 6, 9]
    assert len({r["file_name"] for r in rows}) == 45
    assert all((int(r["patient"][2:]) <= 40) == (r["split"] == "train") for r in rows)
    names = {r["file_name"] for r in rows}
    assert not any(n.startswith("000005_01_01") for n in names) and "000006_01_01_017.png" not in names
    with pytest.raises(RuntimeError, match="has only"):
        D.select_key_slices(info, 30, 6, 31)
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    res = D.write_key_slice_configs(rows, cfg_dir)
    assert res["n_slices"] == {"train": 30, "val": 6, "test": 9}
    assert D.read_key_manifest(cfg_dir / "deeplesion_keyslices.csv") == rows
    sp = load_yaml(cfg_dir / "split_joint_dl.yaml")["split"]
    assert sp["train"] == res["patients"]["train"] and sp["test"][:2] == ["L310", "L506"]
    pats = load_yaml(cfg_dir / "joint_patients.yaml")["data"]["patients"]
    assert pats[:10] == load_yaml("configs/data/mayo_3mm.yaml")["data"]["patients"] and len(pats) == len(set(pats))
    # two fake NIH zip files in the Images_png/<series>/<slice>.png layout; each series has unselected neighbours
    src = tmp_path / "nih"
    hu = np.arange(16, dtype=np.float32).reshape(4, 4) - 100
    zips = {1: src / "Images_png_01.zip", 2: src / "Images_png_02.zip"}
    src.mkdir()
    for k, zp in zips.items():
        with zipfile.ZipFile(zp, "w") as z:
            for line in lines[1:-1]:
                name = line.split(",")[0]
                if (int(name[:6]) <= 30) != (k == 1):
                    continue
                series, sl = name[:-4].rsplit("_", 1)
                for s_ in (sl, f"{int(sl) + 1:03d}"):
                    write_png_hu(src / "x" / series / f"{s_}.png", hu + int(name[:6]) + int(s_))
                    z.write(src / "x" / series / f"{s_}.png", f"Images_png/{series}/{s_}.png")

    class FakeBox:
        def __init__(self, *_, **__):
            pass

        def find(self, path):
            return type("Item", (), {"name": path, "size": 1, "id": 7})()

        def list_folder(self, _id):
            return [type("Item", (), {"name": zp.name, "size": 1, "id": k})() for k, zp in zips.items()]

        def download(self, item, dst):
            dst = Path(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            data = {D.INFO_CSV: info.read_bytes(), D.MD5_FILE: b""}.get(item.name)
            dst.write_bytes(data if data is not None else (src / item.name).read_bytes())
            return dst

    monkeypatch.setattr(D_download, "BoxShare", FakeBox)
    manifest = cfg_dir / "deeplesion_keyslices.csv"
    # only one zip file scanned so far: not enough slices
    with pytest.raises(RuntimeError, match="Still missing"):
        D.download_key_slices(tmp_path / "raw", manifest, zips=[1], workers=1)
    dest = D.download_key_slices(tmp_path / "raw", manifest, zips=[1, 2], workers=2)
    got = sorted(f"{f.parent.name}_{f.name}" for f in (dest / "Images_png").glob("*/*.png"))
    assert got == sorted(names) and (dest / "DL_info.csv").is_file()            # neighbouring slices are not unpacked
    assert not list((dest / "zips").glob("*.zip"))                              # zips are deleted once used
    one = rows[0]["file_name"]
    f = dest / "Images_png" / one[:12] / f"{one[13:16]}.png"
    assert np.array_equal(read_png_hu(f), hu + int(one[:6]) + int(one[13:16]))
    assert D.download_key_slices(tmp_path / "raw", manifest, zips=[1, 2]) == dest          # complete: no new download


def test_plan_grids():
    """Plan of 2026-10-10 (revised 2026-10-11): first-order runs and L-BFGS data-step runs, BOTH trained from scratch on
    each dataset; L-BFGS runs use the exact recipe of the first-order run of the same model (same lr), no checkpoint."""
    grids = {n: expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True) for n in PLAN}
    assert {n: len(j) for n, j in grids.items()} == {"base_mayo": 2, "base_dl": 8, "lbfgs_mayo": 8, "lbfgs_dl": 8}
    ids = [j.run_id for jobs in grids.values() for j in jobs]
    assert len(ids) == len(set(ids)) == 26
    four = {"learn_unet", "learn_resnet", "learn_densenet", "regformer"}
    assert {j.arch for j in grids["base_mayo"]} == {"regformer"}                # the three LEARN backbones have E3 runs
    assert all({j.arch for j in grids[n]} == four for n in ("base_dl", "lbfgs_mayo", "lbfgs_dl"))
    # no lr sweep: LEARN keeps the Mayo lr (3e-4), RegFormer uses 1e-4 as in the papers; L-BFGS runs use the lr of the
    # first-order run of the same model
    assert sorted(j.run_id for j in grids["base_dl"] if "lr0.0001" in j.run_id) == [
        "base_dl_regformer_fan_a120_lr0.0001_s0", "base_dl_regformer_fan_a90_lr0.0001_s0"]
    assert all("lr0.0001" in j.run_id for j in grids["base_mayo"])
    for n in ("base_dl", "lbfgs_mayo", "lbfgs_dl"):
        assert sum("lr0.0003" in j.run_id for j in grids[n]) == 6
        assert all(("lr0.0001" in j.run_id) == (j.arch == "regformer") for j in grids[n])
    split = {"base_mayo": "split_joint_mayo", "base_dl": "split_joint_dl", "lbfgs_mayo": "split_joint_mayo",
             "lbfgs_dl": "split_joint_dl"}
    for name, jobs in grids.items():
        for j in jobs:
            c = load_config(j.files, j.overrides, use_defaults=False)
            m = c.require_model()
            assert c.split.name == split[name] and c.data.format == "mixed"
            assert (c.train.epochs, c.train.batch_size, c.train.loss, c.train.optimizer) == (50, 1, "mse", "adam")
            assert c.train.init_from is None and int(c.seed) == 0                # every run trains from scratch
            if name.startswith("lbfgs_"):
                assert m.update == m.update_tag == "lbfgs" and m.lbfgs_memory == 13 and not m.uses_cg
                assert (m.lbfgs_damping, m.lbfgs_curv_min) == (0.5, 0.0)        # stability when training from scratch
            else:
                assert m.update == "gd" and m.update_tag == ("fbp" if m.framework == "regformer" else "gd")
    rf = load_config(grids["base_mayo"][0].files, grids["base_mayo"][0].overrides, use_defaults=False).require_model()
    assert (rf.framework, rf.arch, rf.K) == ("regformer", "regformer", 10)
    for bad in ("model.update=cg", "model.framework=mamba"):
        with pytest.raises(ConfigError):
            load_config(["configs/model/regformer.yaml"], [bad])


def test_compare_grids():
    """Comparison table with the CG-GLORE paper: four methods trained FROM SCRATCH on each dataset with ONE shared
    training recipe, as that paper trained its compared models (configs/train/bmvc.yaml); data, split, checkpoint
    selection and numerical mode as in the existing runs."""
    grids = {n: expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True) for n in CMP}
    ids = [j.run_id for jobs in grids.values() for j in jobs]
    assert len(ids) == len(set(ids)) == 16 and all(len(j) == 8 for j in grids.values())
    assert sorted(j.run_id for j in grids["cmp_mayo"] if "fan_a90" in j.run_id) == [
        "cmp_mayo_cg_glore_fan_a90_lr0.0001_s0", "cmp_mayo_fbpconvnet_fan_a90_lr0.0001_s0",
        "cmp_mayo_learn_orig_fan_a90_lr0.0001_s0", "cmp_mayo_lpd_fan_a90_lr0.0001_s0"]
    old = {j.run_id for n in PLAN for j in expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True)}
    assert not old & set(ids)
    model = {"cg_glore": ("cg_glore", "cg_glore", 15, "cg_fbp"), "fbpconvnet": ("fbpconvnet", "fbpconvnet", 1, "post"),
             "lpd": ("lpd", "lpd", 10, "pd"), "learn_orig": ("learn", "learn_orig", 30, "gd")}
    # the CG-GLORE paper's recipe (authors' code): (optimizer, lr, lr_min, schedule, epochs, max_steps, warmup, gradient
    # clipping, clipping mode, weight decay, momentum, beta2), identical for every method
    recipe = ("adamw", 1e-4, 1e-5, "cosine", 50, None, 0.0, None, "norm", 1e-2, 0.0, 0.999)
    ref = load_config(*next((j.files, j.overrides) for j in expand("configs/experiment/base_dl.yaml", JOINT,
                                                                    models=["learn_densenet"])), use_defaults=False)
    for name, jobs in grids.items():
        assert {j.arch for j in jobs} == set(model)
        for j in jobs:
            c = load_config(j.files, j.overrides, use_defaults=False)
            m, t = c.require_model(), c.train
            assert (m.framework, m.regularizer.name, m.K, m.update_tag) == model[j.arch]
            assert (t.optimizer, t.lr, t.lr_min, t.schedule, t.epochs, t.max_steps, t.warmup_epochs, t.grad_clip,
                    t.grad_clip_mode, t.weight_decay, t.momentum, t.betas[1]) == recipe, j.arch
            assert m.update == "gd" and not m.uses_cg and t.init_from is None and int(c.seed) == 0
            assert c.split.name == ("split_joint_mayo" if name == "cmp_mayo" else "split_joint_dl")
            # thesis-side settings, as in the existing runs: batch, loss, checkpoint selection, numerical mode, data,
            # FBP
            for key in ("batch_size", "loss", "select_by", "val_every"):
                assert getattr(t, key) == getattr(ref.train, key), key
            assert c.data == ref.data and c.baseline.fbp == ref.baseline.fbp
            # same TF32 as the existing runs; "deterministic" mode off (each setup trains once; it is 2.6x slower)
            assert (c.precision.tf32, c.precision.amp, c.precision.deterministic) == (ref.precision.tf32, False, False)
            assert ref.precision.deterministic is True
            assert c.data.augment == ref.data.augment                           # no data augmentation
    by_arch = {j.arch: load_config(j.files, j.overrides, use_defaults=False).require_model() for j in grids["cmp_dl"]}
    learn = by_arch["learn_orig"]
    assert learn.share_theta is False                                           # one weight set per stage
    assert (learn.regularizer.params["init_std"], learn.regularizer.params["zero_last"]) == (0.01, False)
    assert by_arch["lpd"].regularizer.params["init"] == "zero"                  # as in the original paper
    with pytest.raises(ConfigError):                                            # their update step is fixed
        load_config(["configs/model/cg_glore.yaml"], ["model.update=lbfgs"])


def test_author_training_recipes_are_supported():
    """Trainer pieces for each method's own author recipe (currently unused by any grid: the comparison table follows
    the CG-GLORE paper's shared recipe): SGD with momentum, a uniform log-scale schedule, value gradient clipping
    (FBPConvNet); cosine to 0 (Learned Primal-Dual); Gaussian initialisation (LEARN, in use)."""
    from lact.core.models import build_regularizer
    from lact.services.training.optim import build_optimizer, build_scheduler, lr_lambda_factory

    f = lr_lambda_factory(total_steps=100, warmup_steps=0, lr=1e-2, lr_min=1e-3, schedule="exp")
    assert f(0) == 1.0 and f(50) == pytest.approx(10 ** -0.5) and f(100) == pytest.approx(0.1) == f(500)
    cos = lr_lambda_factory(total_steps=100, warmup_steps=0, lr=1e-3, lr_min=0.0)
    assert cos(0) == 1.0 and cos(50) == pytest.approx(0.5) and cos(100) == pytest.approx(0.0, abs=1e-12)
    t = load_config([], ["train.optimizer=sgd", "train.momentum=0.99", "train.schedule=exp", "train.lr=1e-2",
                         "train.lr_min=1e-3", "train.grad_clip=1e-2", "train.grad_clip_mode=value"]).train
    net = torch.nn.Linear(3, 2)
    opt = build_optimizer(net, t)
    assert isinstance(opt, torch.optim.SGD) and opt.defaults["momentum"] == 0.99 and opt.defaults["lr"] == 1e-2
    sched = build_scheduler(opt, t, steps_per_epoch=10, total_steps=20)
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    for bad in ("train.optimizer=lion", "train.schedule=step", "train.grad_clip_mode=both",
                "train.schedule=exp train.lr_min=0"):
        with pytest.raises(ConfigError):
            load_config([], bad.split())
    torch.manual_seed(0)
    reg = build_regularizer("learn_orig", channels=48, kernel=5, init_std=0.01, zero_last=False)
    w = torch.cat([m.weight.flatten() for m in reg.net if isinstance(m, torch.nn.Conv2d)])
    assert float(w.std()) == pytest.approx(0.01, rel=0.05) and float(reg.net[-1].weight.abs().max()) > 0
    assert all(float(m.bias.abs().max()) == 0 for m in reg.net if isinstance(m, torch.nn.Conv2d))
    assert float(build_regularizer("cnn3").net[-1].weight.abs().max()) == 0     # the project default is unchanged


def test_dpma_grids():
    """DPMA trains in two stages as in the original code: dpma1_* grids train M1 from scratch; dpma_* grids load M1 of
    the run with the same dataset and geometry, then train M2. Shared CG-GLORE recipe; the 240° range holds the
    views."""
    from lact.config.defaults import load_geometry

    grids = {n: expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True) for n in DPMA_GRIDS}
    assert all(len(j) == 2 for j in grids.values())
    assert len({j.run_id for jobs in grids.values() for j in jobs}) == 8
    for name, jobs in grids.items():
        data = name.rsplit("_", 1)[1]
        for j in jobs:
            c = load_config(j.files, j.overrides, use_defaults=False)
            m, t = c.require_model(), c.train
            assert (m.framework, m.arch, m.regularizer.name, m.update_tag, m.K) == ("dpma", "dpma", "dpma", "dual", 1)
            assert (t.optimizer, t.lr, t.weight_decay, t.lr_min, t.epochs, t.batch_size) == ("adamw", 1e-4, 1e-2,
                                                                                         1e-5, 50, 1)
            assert (t.schedule, t.warmup_epochs, t.grad_clip) == ("cosine", 0.0, None)      # CG-GLORE paper recipe
            assert (c.precision.tf32, c.precision.deterministic) == (True, False)
            assert c.split.name == f"split_joint_{data}"
            first = name.startswith("dpma1_")
            assert m.regularizer.params["stage"] == (1 if first else 2)
            assert t.init_from == (None if first else f"dpma1_{data}")          # source is stage 1 on the SAME dataset
            p = m.regularizer.params
            assert (p["width"], p["refine_iters"], p["refine_step"], p["refine_lambda"], p["refine_delta"],
                    p["refine_tv_iters"], p["target_name"], p["target_span_deg"]) == (16, 300, 0.15, 0.1, 0.005, 10,
                                                                                      "fan_a240", 240.0)
    full = load_geometry("fan_a240")
    for g in ("fan_a120", "fan_a90"):
        geom = load_geometry(g)
        assert geom.replace(name=full.name, span_deg=full.span_deg) == full     # same geometry, only the range differs
        off = (geom.angles_deg()[0] - full.angles_deg()[0]) / full.step_deg
        assert abs(off - round(off)) < 1e-9 and round(off) == {"fan_a120": 240, "fan_a90": 300}[g]
    assert full.span_deg >= full.short_scan_deg                                 # enough angles for FBP
    with pytest.raises(ConfigError):
        load_config(["configs/model/dpma.yaml"], ["model.update=lbfgs"])


def test_grid_can_be_rerun_under_another_name():
    """--exp-rename: same grid, renamed experiment, overrides through -s (e.g. re-running L-BFGS with other epochs)."""
    rename = {"lbfgs_": "lbfgs25_"}
    assert renamed("lbfgs_dl", rename) == "lbfgs25_dl" and renamed("base_dl", rename) == "base_dl"
    jobs = expand("configs/experiment/lbfgs_dl.yaml", JOINT, ["train.epochs=25"], models=["learn_resnet"],
                  exp_rename=rename)
    assert [j.run_id for j in jobs] == ["lbfgs25_dl_learn_resnet_fan_a120_lr0.0003_s0",
                                        "lbfgs25_dl_learn_resnet_fan_a90_lr0.0003_s0"]
    c = load_config(jobs[0].files, jobs[0].overrides, use_defaults=False)
    assert c.exp == "lbfgs25_dl" and c.train.epochs == 25 and c.train.init_from is None
    # same grid with another data step and starting point: everything goes through -s, the grid is unchanged
    other = expand("configs/experiment/lbfgs_mayo.yaml", JOINT,
                   ["model.update=gd", "train.init_from=base_mayo,e3_main"], exp_rename={"lbfgs_": "ctrl_"},
                   require_lr=True)
    assert len(other) == 8 and all(j.run_id.startswith("ctrl_mayo_") for j in other)
    tags = set()
    for j in other:
        c = load_config(j.files, j.overrides, use_defaults=False)
        assert c.exp == "ctrl_mayo" and c.train.init_from == "base_mayo,e3_main" and c.train.epochs == 50
        tags.add(c.require_model().update_tag)
    assert tags == {"gd", "fbp"}


def test_phase2_keeps_the_precision_mode_of_the_mayo_runs():
    """Phase 2 runs on the same GPU type as the Mayo campaign: every new run keeps the numerical mode of the E3 runs
    (the reference), and every model command sets that mode from the configuration rather than PyTorch defaults."""
    from lact.infra.system.runtime import apply_precision

    mayo = load_config([]).precision                                            # E0-E6 on Mayo: as they ran
    joint = load_config([*JOINT, "configs/data/split_joint.yaml"]).precision
    mode = lambda p: (p.tf32, p.deterministic, p.amp)
    assert mode(joint) == mode(mayo) == (True, True, False)
    for n in PLAN:
        for j in expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True):
            assert load_config(j.files, j.overrides, use_defaults=False).precision.tf32 is True
    one = expand("configs/experiment/base_dl.yaml", JOINT, ["precision.tf32=false"], models=["learn_unet"])[0]
    assert load_config(one.files, one.overrides, use_defaults=False).precision.tf32 is False     # overridable with -s
    keep = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    try:
        apply_precision(False)
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        apply_precision(True)
        assert torch.backends.cudnn.allow_tf32 and torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = keep
    # every model command sets the numerical mode (PyTorch enables TF32 for cuDNN by default)
    for f in ("training/trainer/loop.py", "evaluation/evaluate.py", "evaluation/recheck.py", "evaluation/benchmark.py"):
        assert "apply_precision(" in Path("src/lact/services", f).read_text(encoding="utf-8"), f


def test_exclude_drops_runs_by_substring():
    """--exclude drops runs by run_id substring; the dropped and remaining parts make up exactly the 26 planned runs."""
    every = [j for n in PLAN for j in expand(f"configs/experiment/{n}.yaml", JOINT, require_lr=True)]
    rest = filter_jobs(every, exclude=["regformer"])
    late = filter_jobs(every, ["regformer"])
    a, b = {j.run_id for j in rest}, {j.run_id for j in late}
    assert (len(a), len(b)) == (18, 8) and not a & b and a | b == {j.run_id for j in every}
    assert filter_jobs(every, ["_fan_a90_"], ["regformer"]) == [j for j in every if "_fan_a90_" in j.run_id
                                                                 and "regformer" not in j.run_id]


def test_parallel_grid_waits_for_sources_and_starts_long_runs_first(tmp_path, monkeypatch):
    """One command for a source grid and its dependent grid: runs with train.init_from start only once their source
    run is DONE; pilot measurements queue long runs first and tell which runs need a GPU of their own. (The current plan
    trains every run from scratch; here the L-BFGS grid gets init_from through -s to test the waiting mechanism.)"""
    sets = [f"paths.runs={tmp_path / 'runs'}"]
    jobs = filter_jobs([j for n, extra in (("base_dl", []), ("lbfgs_dl", ["train.init_from=base_dl"]))
                        for j in expand(f"configs/experiment/{n}.yaml", JOINT, sets + extra, require_lr=True,
                                        models=["learn_unet", "regformer"])], ["_fan_a90_"])
    seen = {}

    class FakeRunner:
        interrupted = stopped_starting = False

        def __init__(self, *_, **kw):
            seen["init"] = kw

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def run(self, commands, weights=None, ready=None):
            seen.update(order=[k for k, _ in commands], weights=weights, ready=ready)
            return {k: 0 for k, _ in commands}

    monkeypatch.setattr(experiments.runner, "ProcessRunner", FakeRunner)
    codes = experiments.run_jobs(jobs, gpus=["GPU-a", "GPU-b"], per_gpu=2, gpu_mem_mb=11264.0,
                                 job_mem_mb={"regformer": 9000.0, "learn_unet": 2900.0},
                                 sec_per_step={"regformer": 0.9, "learn_unet": 0.2})
    assert codes == [0, 0, 0, 0] and seen["init"]["per_gpu"] == 2
    assert ["regformer" in k for k in seen["order"]] == [True, True, False, False]          # long runs queued first
    w = seen["weights"]
    assert all((w[k] > 0.5) == ("regformer" in k) for k in seen["order"])                 # RegFormer alone on a GPU
    base, lb = "base_dl_learn_unet_fan_a90_lr0.0003_s0", "lbfgs_dl_learn_unet_fan_a90_lr0.0003_s0"
    ready = seen["ready"]
    assert ready(base) and not ready(lb)
    d = tmp_path / "runs" / "base_dl" / base
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"arch": "learn_unet", "geometry": "fan_a90", "seed": 0}))
    (d / "best.pt").write_bytes(b"")
    # no DONE marker: the source is unfinished
    assert not ready(lb)
    (d / "DONE").write_text("{}")
    assert ready(lb) and not ready("lbfgs_dl_regformer_fan_a90_lr0.0001_s0")


def test_lr_table_rules(tmp_path):
    base = {"exp": "x", "grid": {"model": ["configs/model/learn_cnn.yaml", "configs/model/learn_vit.yaml"],
                                 "geometry": ["configs/geometry/fan_a120.yaml"], "seed": [0]}}
    ok = tmp_path / "ok.yaml"
    save_yaml({**base, "lr_table": {"cnn3": 2e-4, "vit": 5e-5}}, ok)
    jobs = expand(str(ok))
    assert [j.run_id for j in jobs] == ["x_learn_cnn_fan_a120_lr0.0002_s0", "x_learn_vit_fan_a120_lr5e-05_s0"]
    miss = tmp_path / "miss.yaml"
    save_yaml({**base, "lr_table": {"cnn3": 2e-4}}, miss)
    with pytest.raises(ConfigError, match="lr_table has no lr"):
        expand(str(miss))
    assert len(expand(str(miss), models=["learn_cnn"])) == 1            # unselected models need no lr
    for bad in ({"lr_table": {"cnn3": -1.0}}, {"lr_table": {"cnn3": 1e-4}, "grid": {**base["grid"], "lr": [1e-4]}}):
        f = tmp_path / "bad.yaml"
        save_yaml({**base, **bad}, f)
        with pytest.raises(ConfigError):
            ExperimentSpec.load(str(f))
    both = tmp_path / "both.yaml"                # table for models with an lr + lr_from for the rest
    save_yaml({**base, "lr_table": {"cnn3": 2e-4}, "lr_from": "chua_quet"}, both)
    runs = [f"paths.runs={tmp_path}"]
    assert [j.run_id for j in expand(str(both), overrides=runs)] == [
        "x_learn_cnn_fan_a120_lr0.0002_s0", "x_learn_vit_fan_a120_lr0.0001_s0"]          # ViT: default lr, not swept
    with pytest.raises(ConfigError, match="chua_quet"):
        expand(str(both), overrides=runs, require_lr=True)


def test_model_update_config():
    m = load_config(["configs/model/learn_cnn.yaml"]).require_model()
    assert m.update == "gd" and m.update_kwargs()["update"] == "gd"             # default: LEARN as before
    m = load_config(["configs/model/learn_cnn.yaml"], ["model.update=admm", "model.metric=sobolev",
                                                       "model.cg_iters=7", "model.cg_tol=1e-3"]).require_model()
    assert m.update_kwargs() == {"update": "admm", "metric": "sobolev", "lambda0": 100.0, "rho0": 1.0, "gamma0": 1.0,
                                 "cg_iters": 7, "cg_tol": 1e-3}
    m = load_config(["configs/model/learn_resnet.yaml"], ["model.update=lbfgs", "model.lbfgs_memory=4"]).require_model()
    assert m.update_kwargs()["lbfgs_memory"] == 4 and m.update_tag == "lbfgs" and not m.uses_cg
    m = load_config(["configs/model/regformer.yaml"], ["model.update=lbfgs"]).require_model()
    assert m.update_tag == "lbfgs" and m.lbfgs_memory == 13                     # default: remember every stage
    for bad in ("model.update=newton", "model.metric=l1", "model.cg_iters=0", "model.cg_tol=-1", "model.lambda0=0",
                "model.update=lbfgs model.metric=sobolev", "model.lbfgs_memory=0"):
        with pytest.raises(ConfigError):
            load_config(["configs/model/learn_cnn.yaml"], bad.split())


@pytest.mark.parametrize("argv", [
    ["eval", "cg-check", "--exp", "x_cg", "--factor", "4"], ["eval", "runs", "--exp", "e3_main", "--on-config-data"],
    ["eval", "recheck", "--exp", "e3_main", "--on-config-data"], ["report", "phase2", "--exp", "base_dl"],
    ["train", "grid", "configs/experiment/lbfgs_dl.yaml", "--models", "learn_resnet", "--exp-rename", "lbfgs_=x_"],
    ["train", "pilot", "--spec", "configs/experiment/lbfgs_dl.yaml", "--tag", "_x", "--geometry", "fan_a90"],
    ["train", "grid", "configs/experiment/base_dl.yaml", "configs/experiment/lbfgs_dl.yaml", "--exclude",
     "regformer", "--mem-table", "a.csv", "b.csv", "--gpus", "all", "--per-gpu", "2"],
    ["train", "pilot", "--spec", "configs/experiment/base_dl.yaml", "--exclude", "regformer"],
])
def test_cli_parses_phase2_commands(argv):
    assert callable(build_parser().parse_args(argv).func)


def test_phase2_scripts_exist_and_reference_real_files():
    text = Path("scripts/vast/phase2.sh").read_text(encoding="utf-8")
    for name in PLAN:
        assert Path(f"configs/experiment/{name}.yaml").is_file() and name in text
    for gone in ("p0_sweep", "p1_dl", "p2_joint", "p3_cg", "p3_admm", "p4_dl"):          # old plan archived
        assert not Path(f"configs/experiment/{gone}.yaml").exists() and gone not in text
    for f in ("configs/data/split_joint.yaml", "configs/data/split_joint_mayo.yaml"):
        assert f in text and Path(f).is_file()
    assert "--exp-rename lbfgs_=ctrl_" in text and "model.update=gd" in text            # control runs (--control)
    for name in CMP:                                                                    # CG-GLORE comparison group
        assert Path(f"configs/experiment/{name}.yaml").is_file() and name in text
    assert "--compare-only" in text and Path("scripts/vast/mac_push_compare.sh").is_file()
    for name in DPMA_GRIDS:
        assert Path(f"configs/experiment/{name}.yaml").is_file() and name in text
    assert "CMP_EVAL=(cmp_mayo cmp_dl dpma_mayo dpma_dl)" in text                    # dpma1_* are not reported
    assert "-g fan_a240 --sizes" in text and Path("configs/geometry/fan_a240.yaml").is_file()
    # thousands of DeepLesion patients => thousands of cache files open at once: the script must raise the open-file
    # limit
    assert "ulimit -n 65536" in Path("scripts/vast/env.sh").read_text(encoding="utf-8")
    assert Path("scripts/vast/mac_push_phase2.sh").is_file() and Path("configs/variant/joint.yaml").is_file()
    push = Path("scripts/vast/mac_push_phase2.sh").read_text(encoding="utf-8")
    # the three backbones with E3 source runs
    assert "learn_unet learn_resnet learn_densenet}" in push
