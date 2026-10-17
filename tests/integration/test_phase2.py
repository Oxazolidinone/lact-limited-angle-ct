"""Phase 2 on synthetic data through the same CLI services (no ASTRA needed):
joint Mayo + DeepLesion store (given mu_ref) -> Mayo-trained runs evaluated across the joint store -> L-BFGS attached
to trained runs and retrained (plan of 2026-10-10) -> tables C1-C3. The cg/admm rules of the 2026-10-08 plan are still
in the code and therefore tested here (training, initialisation from first-order runs, CG convergence); experiment
names p0_-p3_ in this file are only labels of synthetic runs. Expected table headers/labels are report content."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
import torch

from lact.config import ConfigError, load_config
from lact.config.loader import save_yaml
from lact.infra.io.files import load_json
from lact.infra.io.hdf5_store import ImageStore
from lact.services import operators
from lact.services.data import fake as fake_data
from lact.services.data import prep as data_prep
from lact.services.data import simulation
from lact.services.evaluation import benchmark, cgcheck, recheck
from lact.services.evaluation import evaluate as evaluation
from lact.services.factories import dataset_for, results_store
from lact.services.reporting import make_phase2_tables
from lact.services.reporting.phase2 import dataset_of, per_dataset, prepare
from lact.services.training.trainer import Trainer, find_init_run, run_store_for

MAYO = fake_data.FAKE_PATIENTS
MAYO_SPLIT = {"train": MAYO[:7], "val": ["L333"], "test": ["L310", "L506"]}
DL = [f"DL{p:06d}" for p in range(1, 9)]
DL_SPLIT = {"train": DL[:4], "val": DL[4:6], "test": DL[6:]}
COMMON = ["data.img_size_sim=64", "data.img_size_rec=32", "model.K=2",
          "train.epochs=1", "train.max_steps=4", "train.limit_val=4", "train.val_batch_size=2",
          "train.tensorboard=false", "train.log_every=1", "precision.deterministic=false"]
SMALL = {"learn_cnn": ["model.regularizer.channels=8"],                         # scaled-down version of each model
         "regformer": ["model.regularizer.dim=8", "model.regularizer.heads=2"],
         # comparison table with the CG-GLORE paper
         "cg_glore": ["model.regularizer.channels=4", "model.regularizer.landmarks=16"],
         "fbpconvnet": ["model.regularizer.base=4"], "lpd": ["model.regularizer.channels=8"],
         "learn_orig": ["model.regularizer.channels=8"],
         "dpma": ["model.regularizer.width=4", "model.regularizer.chunk_size=4", "model.regularizer.target_name=t240",
                  "model.regularizer.refine_iters=4"]}


def _paths(out, tmp):
    return [f"paths.{k}={out / k}" for k in ("data", "runs", "eval", "tables", "figures", "baselines")] + [
        f"paths.outputs={out}", f"paths.operators={tmp / 'operators'}", f"data.images_h5={out}/data/images.h5",
        f"data.stats_json={out}/data/stats.json", f"data.slices_json={out}/data/slices.json"]


@pytest.fixture(scope="module")
def world(tmp_path_factory, small_geom):
    pytest.importorskip("pydicom")
    tmp = tmp_path_factory.mktemp("phase2")
    fake_data.make_fake_dicom(tmp / "dicom", n_slices=3, size=64)
    dl_root = fake_data.make_fake_deeplesion(tmp / "deeplesion", n_patients=8, n_slices=6, size=64)
    # 1) Mayo store as in the E0-E6 campaign: mu_ref from the Mayo training split
    mayo = load_config(["configs/model/learn_cnn.yaml"], _paths(tmp / "mayo", tmp) + [
        f"data.dicom_root={tmp / 'dicom'}", "exp=e3_main", *COMMON, *SMALL["learn_cnn"]]).replace(geometry=small_geom)
    data_prep.preprocess(mayo)
    mu_ref = load_json(mayo.data.stats_json)["mu_ref"]
    # 2) joint store: same mu_ref plus DeepLesion patients; each split is a file like configs/data/split_joint*
    splits = {"joint": {k: MAYO_SPLIT[k] + DL_SPLIT[k] for k in MAYO_SPLIT},
              "joint_dl": {"train": DL_SPLIT["train"], "val": DL_SPLIT["val"],
                           "test": MAYO_SPLIT["test"] + DL_SPLIT["test"]},
              "joint_mayo": {**MAYO_SPLIT, "test": MAYO_SPLIT["test"] + DL_SPLIT["test"]}}
    for name, sp in splits.items():
        save_yaml({"split": {"name": f"split_{name}", **sp}}, tmp / f"split_{name}.yaml")

    def joint(split, extra=(), model="learn_cnn"):
        files = ["configs/data/joint.yaml", str(tmp / f"split_{split}.yaml"), f"configs/model/{model}.yaml"]
        return load_config(files, _paths(tmp / "joint", tmp) + [
            f"data.dicom_root={tmp / 'dicom'}", f"data.deeplesion_root={dl_root}",
            f"data.info_csv={dl_root}/DL_info.csv", "data.max_slices_per_patient=4",
            f"data.norm.mu_ref={mu_ref!r}", f"data.patients=[{', '.join(MAYO + DL)}]",
            *COMMON, *SMALL[model], *extra]).replace(geometry=small_geom)

    data_prep.preprocess(joint("joint"))
    for c in (mayo, joint("joint")):
        operators.build_operators(c, [small_geom], sizes=[32, 64])
        simulation.simulate_all(c, [small_geom], batch=2)
    evaluation.freeze_configs(mayo, "pytest")
    evaluation.freeze_configs(joint("joint"), "pytest")
    return {"mayo": mayo, "joint": joint, "tmp": tmp}


def _fit(cfg):
    t = Trainer(cfg)
    s = t.fit()
    assert s["status"] == "done" and np.isfinite(s["best_val_psnr"])
    return t, s


@pytest.fixture(scope="module")
def runs(world):
    """Runs shared by the tests below (each run trains once)."""
    j = world["joint"]
    cg = ["model.update=cg", "model.cg_iters=6", "model.cg_tol=0.01"]
    out = {"mayo_gd": _fit(world["mayo"]),
           "dl_gd": _fit(j("joint_dl", ["exp=p1_dl"])),
           "joint_gd": _fit(j("joint", ["exp=p2_joint"])),
           "joint_cg": _fit(j("joint", ["exp=p3_cg", *cg])),
           "joint_admms": _fit(j("joint", ["exp=p3_admms", "model.update=admm", "model.metric=sobolev",
                                           "model.cg_iters=6", "model.lambda0=300"])),
           "mayo_regformer": _fit(j("joint_mayo", ["exp=p0_mayo"], model="regformer")),
           "joint_regformer": _fit(j("joint", ["exp=p2_joint"], model="regformer"))}
    for key, (t, _) in out.items():
        if key != "mayo_gd":
            assert evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2) is not None
    return out


def test_joint_store_holds_both_datasets_on_the_mayo_scale(world):
    mayo, j = world["mayo"], world["joint"]("joint")
    a, b = ImageStore(mayo.data.images_h5), ImageStore(j.data.images_h5)
    assert b.mu_ref == a.mu_ref and load_json(j.data.stats_json)["mode"] == "fixed"
    assert set(b.patients()) == set(MAYO + DL) and b.n_slices("DL000001") == 4 and b.n_slices("L067") == 3
    # Mayo images in the joint store match the Mayo store exactly
    for pid in ("L067", "L333", "L506"):
        assert np.array_equal(a.read(pid, "img256"), b.read(pid, "img256"))
    assert len(dataset_for(j, "train")) == 7 * 3 + 4 * 4 and len(dataset_for(j, "test")) == 2 * 3 + 2 * 4
    assert len(dataset_for(world["joint"]("joint_dl"), "train")) == 4 * 4
    assert data_prep.patient_source(j, "DL000003")[0] == "deeplesion"
    assert data_prep.patient_source(j, "L067")[0] == "dicom"
    with pytest.raises(ConfigError):
        load_config([], ["data.norm.mode=fixed"])                                # mu_ref missing
    with pytest.raises(ValueError, match="deeplesion_root"):
        data_prep.patient_source(load_config(["configs/data/joint.yaml"], ["data.deeplesion_root=null"]), "DL000003")


def test_mayo_run_evaluated_on_joint_store(world, runs):
    """A run trained on the Mayo store evaluated on the joint store: Mayo patients reproduce the old results, plus
    DeepLesion."""
    run = runs["mayo_gd"][0].run
    own = evaluation.evaluate_run(run, "test", device="cpu", batch_size=2)
    dcfg = world["joint"]("joint_mayo")
    cross = evaluation.evaluate_run(run, "test", device="cpu", batch_size=2, data_cfg=dcfg)
    assert set(cross.pid) == {"L310", "L506", "DL000007", "DL000008"} and (cross.train_split == "split_7_1_2").all()
    both = own.merge(cross, on=["pid", "idx"], suffixes=("_own", "_joint"))
    assert len(both) == 6 and np.allclose(both.psnr_own, both.psnr_joint, atol=1e-4)
    assert results_store(dcfg).find(exp="e3_main", split="test")                 # written to the joint store's results
    assert evaluation.evaluate_run(run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is None    # already present
    # check validation PSNR on the joint store (same validation patient L333)
    assert recheck.val_psnr(run, "cpu", dcfg) == pytest.approx(run.summary()["best_val_psnr"], abs=1e-4)
    # find runs by experiment in the joint store's runs: copy the Mayo run there as scripts/vast/phase2.sh does
    import shutil

    shutil.copytree(run.dir, dcfg.paths.runs / "e3_main" / run.run_id)
    df = recheck.recheck(dcfg, ["e3_main"], tol_db=1e-3, on_config_data=True)
    assert len(df) == 1 and df.ok.all()


def test_second_order_runs_log_their_parameters(runs):
    t, s = runs["joint_cg"]
    assert s["update"] == "cg" and s["metric"] == "identity" and s["init_from"] is None
    m = pd.read_csv(t.run.metrics_csv)
    assert {"betas", "lambdas", "val_cg_iters"} <= set(m.columns) and "alphas" not in m.columns
    assert 0 < m.val_cg_iters.iloc[-1] <= 6 and "cg_iters" in pd.read_csv(t.run.steps_csv).columns
    t, s = runs["joint_admms"]
    assert s["update"] == "admm" and s["metric"] == "sobolev"
    assert {"betas", "lambdas", "rhos", "gammas"} <= set(pd.read_csv(t.run.metrics_csv).columns)
    t, s = runs["joint_gd"]
    assert s["update"] == "gd" and s["metric"] is None
    assert {"alphas", "betas"} <= set(pd.read_csv(t.run.metrics_csv).columns)


def test_regformer_runs_through_the_same_pipeline(world, runs):
    """RegFormer (a model outside LEARN) trains, saves, reloads and evaluates through the LEARN services."""
    t, s = runs["mayo_regformer"]
    assert s["arch"] == "regformer" and s["update"] == "fbp" and s["metric"] is None
    assert t.run.run_id.startswith("p0_mayo_regformer_") and s["n_params_regularizer"] == s["n_params"] - 2
    m = pd.read_csv(t.run.metrics_csv)
    assert "weights" in m.columns and "alphas" not in m.columns and "val_cg_iters" not in m.columns
    df = results_store(world["joint"]("joint")).load_slices(split="test", exp="p0_mayo")
    assert len(df) == 14 and set(df["update"]) == {"fbp"} and set(df.train_split) == {"split_joint_mayo"}
    assert np.isfinite(df.psnr).all() and (df.regularizer == "regformer").all()
    # the reloaded best checkpoint reproduces the recorded validation PSNR (the internal FBP is rebuilt from the
    # projector)
    assert recheck.val_psnr(t.run, "cpu") == pytest.approx(s["best_val_psnr"], abs=1e-4)


def test_init_from_first_order_run(world, runs):
    """Fine-tuning: a CG run takes the regularizer and beta of the first-order run with the same backbone/geometry;
    lam keeps its initialisation."""
    import torch

    cfg = world["joint"]("joint", ["exp=p3ft_cg", "model.update=cg", "model.cg_iters=4", "model.lambda0=50",
                                   "train.init_from=p2_joint"])
    src = find_init_run(cfg)
    assert src.run_id == runs["joint_gd"][0].run_id
    t = Trainer(cfg)
    assert t.setup()
    ref = torch.load(src.checkpoint("best"), weights_only=False)["model"]
    own = t.model.state_dict()
    assert all(torch.equal(own[k], ref[k]) for k in own if k.startswith("reg.") or k == "b")
    assert torch.allclose(t.model.lambdas, torch.full((2,), 50.0)) and "a" not in own
    s = t.fit()
    assert s["status"] == "done" and s["init_from"] == src.run_id
    assert [e["run_id"] for e in t.run.events() if e.get("event") == "init_from"] == [src.run_id]
    with pytest.raises(ConfigError, match="init_from"):
        find_init_run(world["joint"]("joint", ["train.init_from=khong_co"]))
    evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2)


def test_lbfgs_retrain_from_trained_runs(world, runs):
    """Step 2 of the plan: attach L-BFGS to a trained run and retrain, for both the LEARN framework and RegFormer."""
    import torch

    j = world["joint"]
    # source: the first listed experiment with a matching run; a source with another seed is accepted when it is the
    # only one
    cfg = j("joint", ["exp=lbfgs_x", "model.update=lbfgs", "train.init_from=khong_co,p2_joint"])
    src = find_init_run(cfg)
    assert src.run_id == runs["joint_gd"][0].run_id
    assert find_init_run(j("joint", ["seed=3", "model.update=lbfgs", "train.init_from=p2_joint"])).run_id == src.run_id
    t = Trainer(cfg)
    assert t.setup()
    ref, own = torch.load(src.checkpoint("best"), weights_only=False)["model"], t.model.state_dict()
    assert all(torch.equal(own[k], ref[k]) for k in own if k.startswith("reg.") or k == "b")
    assert "a" in ref and "a" not in own and torch.allclose(t.model.alphas, torch.ones(2))    # alpha reset to 1
    s = t.fit()
    assert s["status"] == "done" and s["update"] == "lbfgs" and s["metric"] is None and s["init_from"] == src.run_id
    assert {"alphas", "betas"} <= set(pd.read_csv(t.run.metrics_csv).columns)
    # RegFormer: regularization blocks from the FBP run, the FBP weight dropped, a new alpha step
    cfg_r = j("joint_mayo", ["exp=lbfgs_x", "model.update=lbfgs", "train.init_from=p0_mayo,e3_main"], model="regformer")
    t2 = Trainer(cfg_r)
    assert t2.setup()
    src_r = runs["mayo_regformer"][0].run
    ref, own = torch.load(src_r.checkpoint("best"), weights_only=False)["model"], t2.model.state_dict()
    assert all(torch.equal(own[k], ref[k]) for k in own if k.startswith("stages."))
    assert "w" in ref and "w" not in own and "a_qn" in own
    s2 = t2.fit()
    assert s2["status"] == "done" and s2["update"] == "lbfgs" and s2["init_from"] == src_r.run_id
    for tr in (t, t2):
        df = evaluation.evaluate_run(tr.run, "test", device="cpu", batch_size=2)
        assert np.isfinite(df.psnr).all() and set(df["update"]) == {"lbfgs"}
    evaluation.evaluate_run(src_r, "test", device="cpu", batch_size=2)
    files = make_phase2_tables(j("joint"), "test", [cfg.require_geometry().name], ["lbfgs_x", "p2_joint", "p0_mayo"])
    c2 = pd.read_csv(files[1]).set_index("Mô hình")
    assert set(c2["Cập nhật"]) == {"L-BFGS"} and set(c2["Thí nghiệm"]) == {"lbfgs_x"}
    # reference = source run
    assert c2.loc["LEARN–CNN", "Mốc"] == "p2_joint" and c2.loc["RegFormer", "Mốc"] == "p0_mayo"
    assert c2.loc["RegFormer", "Huấn luyện trên"] == "Mayo" and set(c2["Loại mốc"]) == {"lượt nguồn"}
    c1 = pd.read_csv(files[0])
    assert {"L-BFGS", "LEARN (bậc một)", "FBP của phần dư"} <= set(c1["Cập nhật"])


def test_one_grid_command_trains_sources_then_dependents(world):
    """As on the server: ONE queue for both the first-order and the L-BFGS grids, run as real child processes. L-BFGS
    runs queued FIRST still wait for their source run to be DONE and load exactly its checkpoint."""
    from lact.services import scheduling as experiments

    tmp = world["tmp"]
    c = world["joint"]("joint_dl")
    save_yaml({"geometry": {"name": "t120", "span_deg": 120.0, "step_deg": 2.0, "n_det": 96, "det_spacing_mm": 10.0}},
              tmp / "t120.yaml")
    common = {"configs": ["configs/paths.yaml", "configs/data/mayo_3mm.yaml", str(tmp / "split_joint_dl.yaml"),
                          "configs/train/default.yaml"],
              "lr_table": {"cnn3": 1.0e-4},
              "grid": {"model": ["configs/model/learn_cnn.yaml"], "geometry": [str(tmp / "t120.yaml")], "seed": [0]}}
    save_yaml({"exp": "grid_base", **common}, tmp / "grid_base.yaml")
    save_yaml({"exp": "grid_lbfgs", **common, "overrides": ["model.update=lbfgs", "train.init_from=grid_base"]},
              tmp / "grid_lbfgs.yaml")
    sets = _paths(tmp / "joint", tmp) + [
        f"data.dicom_root={c.data.dicom_root}", f"data.deeplesion_root={c.data.deeplesion_root}",
        f"data.info_csv={c.data.info_csv}", "data.max_slices_per_patient=4", f"data.norm.mu_ref={c.data.norm.mu_ref!r}",
        f"data.patients=[{', '.join(MAYO + DL)}]", *COMMON, *SMALL["learn_cnn"], "device=cpu"]
    jobs = [j for spec in ("grid_lbfgs", "grid_base")                               # dependent runs queued first
            for j in experiments.expand(str(tmp / f"{spec}.yaml"), ["configs/data/joint.yaml"], sets, require_lr=True)]
    assert [j.run_id.split("_learn_")[0] for j in jobs] == ["grid_lbfgs", "grid_base"] and all(j.pending for j in jobs)
    assert experiments.run_jobs(jobs, gpus=["GPU-pytest"], per_gpu=2) == [0, 0]
    base, lb = (run_store_for(load_config(j.files, j.overrides, use_defaults=False)) for j in jobs[::-1])
    assert base.status == lb.status == "done"
    assert lb.summary()["init_from"] == base.run_id and lb.summary()["update"] == "lbfgs"
    assert (lb.dir / "config.yaml").stat().st_mtime >= (base.dir / "DONE").stat().st_mtime       # ran AFTER its source
    # missing source: the dependent run never starts and the command returns an error code instead of hanging
    save_yaml({"exp": "grid_orphan", **common, "overrides": ["model.update=lbfgs", "train.init_from=khong_co"]},
              tmp / "grid_orphan.yaml")
    orphan = experiments.expand(str(tmp / "grid_orphan.yaml"), ["configs/data/joint.yaml"], sets, require_lr=True)
    codes = experiments.run_jobs(orphan, gpus=["GPU-pytest"], per_gpu=2)
    assert codes and all(code != 0 for code in codes)
    cfg = load_config(orphan[0].files, orphan[0].overrides, use_defaults=False)
    assert run_store_for(cfg).status == "todo"


def test_cg_check_and_pilot(world, runs):
    cfg = world["joint"]("joint")
    df = cgcheck.cg_check(cfg, ["p0_mayo", "p2_joint", "p3_cg", "p3_admms"], n_slices=4, factor=20, batch_size=2)
    assert sorted(df.exp) == ["p3_admms", "p3_cg"]                 # first-order LEARN and RegFormer runs have no CG
    assert np.isfinite(df.psnr).all() and np.isfinite(df.psnr_ref).all() and (df.cg_used <= df.cg_iters).all()
    assert (df.cg_used_ref >= df.cg_used).all() and (df.max_abs_slice_diff >= 0).all()
    cg = world["joint"]("joint", ["model.update=cg", "model.cg_iters=5"])
    p = benchmark.pilot({"learn_cnn": cfg, "cg": cg}, steps=2)
    assert list(p["update"]) == ["gd", "cg"] and list(p.cg_iters_cap) == [0, 5] and p.cg_iters_used.iloc[1] == 5.0
    assert (p.n_train_slices == 7 * 3 + 4 * 4).all()


def test_phase2_tables(world, runs):
    cfg = world["joint"]("joint")
    evaluation.evaluate_baselines(cfg, ["fbp"], [cfg.require_geometry()], "test", "cpu", batch_size=2)
    raw = results_store(cfg).load_slices(split="test")
    df = prepare(raw)
    assert dataset_of("DL000007") == "DeepLesion" and dataset_of("L310") == "Mayo"
    assert set(df.train_data) >= {"-", "Mayo", "DeepLesion", "Mayo + DeepLesion"}
    pdn = per_dataset(df)
    one = pdn[(pdn.exp == "p2_joint") & (pdn.regularizer == "cnn3")]
    assert sorted(one.dataset) == ["DeepLesion", "Mayo"] and (one.n_patients == 2).all()
    files = make_phase2_tables(cfg, "test", [cfg.require_geometry().name])
    assert [f.name for f in files] == ["C1_datasets.csv", "C2_second_order.csv", "C3_runs.csv"]
    c1 = pd.read_csv(files[0])
    assert {"FBP", "LEARN–CNN", "RegFormer"} <= set(c1["Mô hình"])
    assert set(c1["Huấn luyện trên"]) >= {"Mayo", "DeepLesion"}
    reg = c1[(c1["Mô hình"] == "RegFormer") & (c1["Thí nghiệm"] != "lbfgs_x")]      # lbfgs_x: runs of the L-BFGS test
    assert set(reg["Cập nhật"]) == {"FBP của phần dư"} and set(reg["Huấn luyện trên"]) == {"Mayo", "Mayo + DeepLesion"}
    assert list(c1["Mô hình"]).index("RegFormer") > list(c1["Mô hình"]).index("LEARN–CNN")     # after the LEARN models
    assert any("Mayo PSNR" in c for c in c1.columns) and any("DeepLesion PSNR" in c for c in c1.columns)
    c2 = pd.read_csv(files[1])
    # each second-order run is compared with its own first-order reference: from scratch -> p2_joint; the fine-tuned run
    # (p3ft_cg) needs a fine-tuned reference, which does not exist here, so it has no row
    second = c2[c2["Thí nghiệm"].isin(["p3_cg", "p3_admms"])]
    assert set(second["Thí nghiệm"]) == {"p3_cg", "p3_admms"} and set(second["Mốc"]) == {"p2_joint"}
    assert not c2["Thí nghiệm"].eq("p3ft_cg").any()
    # the FBP variant of RegFormer is not a second-order run
    assert set(second["Mô hình"]) == {"LEARN–CNN"}
    ft = world["joint"]("joint", ["exp=p3ft_gd", "train.init_from=p2_joint"])
    t, _ = _fit(ft)
    evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2)
    c2 = pd.read_csv(make_phase2_tables(cfg, "test", [cfg.require_geometry().name])[1]).set_index("Thí nghiệm")
    assert c2.loc["p3ft_cg", "Mốc"] == "p3ft_gd" and c2.loc["p3_cg", "Mốc"] == "p2_joint"
    # runs of the L-BFGS test: p3ft_gd also trained further from the same checkpoint => control
    if "lbfgs_x" in c2.index:
        lb = c2.loc["lbfgs_x"]
        assert sorted(zip(lb["Mô hình"], lb["Mốc"], lb["Loại mốc"])) == [
            ("LEARN–CNN", "p2_joint", "lượt nguồn"), ("LEARN–CNN", "p3ft_gd", "train thêm, giữ bước cũ"),
            ("RegFormer", "p0_mayo", "lượt nguồn")]
    assert c2["Mayo bệnh nhân tốt lên"].str.match(r"^\d/2$").all()
    c3 = pd.read_csv(files[2])
    assert {"p1_dl", "p2_joint", "p3_cg", "p3_admms"} <= set(c3["Thí nghiệm"]) and "GPU" in c3.columns
    assert (c3[c3["Thí nghiệm"] == "p3_cg"]["Vòng CG/giai đoạn"] != "-").all()


def test_phase2_figdata_and_figures(world, runs, tmp_path, monkeypatch):
    """End of a training campaign: collect plotting data (illustrative reconstructions, per-slice metrics, parameters,
    curves) and draw the comparison figures. Runs the two scripts scripts/vast/phase2.sh calls, on a synthetic joint
    store."""
    import runpy

    cfg = world["joint"]("joint")
    evaluation.evaluate_baselines(cfg, ["fbp"], [cfg.require_geometry()], "test", "cpu", batch_size=2)
    out = tmp_path / "figdata"
    monkeypatch.setattr("sys.argv", ["export_phase2.py", "--outputs", str(cfg.paths.outputs), "--out", str(out),
                                     "--no-figures"])
    runpy.run_path("experiments/figures/export_phase2.py", run_name="__main__")
    sl = pd.read_csv(out / "slices.csv")
    assert {"fbp", "regformer", "cnn3"} <= set(sl.method) and {"Mayo", "DeepLesion"} == set(sl.dataset)
    assert {"mayo", "dl", "joint", "none"} >= set(sl.train) and {"mayo", "none"} <= set(sl.train)
    npz = np.load(out / "images.npz")
    geom = cfg.require_geometry().name
    gts = [k for k in npz.files if k.startswith(f"{geom}/gt/")]
    assert gts and npz[gts[0]].dtype == np.float16 and npz[gts[0]].shape == (32, 32)
    assert any(k.startswith(f"{geom}/mayo/regformer/") for k in npz.files)
    assert any(k.startswith(f"{geom}/none/fbp/") for k in npz.files)
    meta = load_json(out / "meta.json")
    assert meta["runs"] and all({"method", "train", "geometry", "init_from"} <= set(v) for v in meta["runs"].values())
    assert load_json(out / "params.json") and any((out / "runs").glob("*/*/metrics.csv"))
    report = (out / "BAO_CAO_KET_QUA.md").read_text(encoding="utf-8")       # summary: numbers, no figures yet
    assert "## 2. Chất lượng trên từng bộ dữ liệu" in report and "| Huấn luyện trên |" in report
    assert "Mayo: " in report and "DeepLesion: " in report and "(chưa có hình" in report
    figs = tmp_path / "figs"
    monkeypatch.setattr("sys.argv", ["make_figures_phase2.py", "--data", str(out), "--out", str(figs),
                                     "--formats", "png"])
    runpy.run_path("experiments/figures/make_figures_phase2.py", run_name="__main__")
    names = {f.name for f in figs.glob("*.png")}
    assert any(n.startswith("fig_psnr_") for n in names) and any(n.startswith("fig_curves_") for n in names)
    if any(m.endswith("+lbfgs") for m in sl.method):          # runs of the L-BFGS test: figures against the source run
        assert any(n.startswith("fig_lbfgs_delta_") for n in names) and any(n.startswith("fig_qual_") for n in names)


def test_learn_invisible_pilot(runs, tmp_path, monkeypatch):
    """A second network predicts the invisible part and merges it with the visible part of a trained run
    (experiments/analysis/learn_invisible.py): the merged image keeps the visible part, and the ceiling is not below the
    original."""
    import runpy

    run = runs["mayo_regformer"][0].run
    out = tmp_path / "invisible"
    monkeypatch.setattr("sys.argv", ["learn_invisible.py", "-c", str(run.config_file), "--exps", "p0_mayo",
                                     "--split", "val", "--out", str(out), "--epochs", "2", "--batch", "2",
                                     "--channels", "8", "16", "--gn-groups", "4", "--device", "cpu"])
    runpy.run_path("experiments/analysis/learn_invisible.py", run_name="__main__")
    sl, summ = pd.read_csv(out / "slices.csv"), pd.read_csv(out / "summary.csv")
    assert set(sl.pid) == {"L333"} and len(summ) == 1 and summ.model[0] == "regformer"
    assert np.isfinite(sl[["psnr_goc", "psnr_ghep", "psnr_tran", "psnr_G_tu_anh_that"]].to_numpy()).all()
    assert (sl.psnr_tran >= sl.psnr_goc - 1e-3).all()            # replacing the hidden part by the truth can only help
    # merging leaves the visible part alone
    assert np.allclose(sl.mse_thay_ghep, sl.mse_thay_goc, rtol=0.2, atol=1e-6)
    assert (out / f"G_{summ.geometry[0]}.pt").is_file() and "psnr_tran" in (out / "summary.md").read_text("utf-8")


def test_two_stage_visible_then_invisible(world, runs, tmp_path, monkeypatch):
    """Two stages after "Learning the Invisible" (experiments/analysis/invisible_two_stage.py): stage 1 trains from
    scratch with a visible-part loss, G learns the hidden part on stage-1 images, the merged image is compared with the
    first-order run."""
    import runpy

    tmp = world["tmp"]
    c = world["joint"]("joint_mayo", model="regformer")
    save_yaml({"geometry": {"name": "t120", "span_deg": 120.0, "step_deg": 2.0, "n_det": 96, "det_spacing_mm": 10.0}},
              tmp / "inv_t120.yaml")
    save_yaml({"exp": "p0_mayo", "lr_table": {"regformer": 1.0e-4},
               "configs": ["configs/paths.yaml", "configs/data/mayo_3mm.yaml", str(tmp / "split_joint_mayo.yaml"),
                           "configs/train/default.yaml"],
               "grid": {"model": ["configs/model/regformer.yaml"], "geometry": [str(tmp / "inv_t120.yaml")],
                        "seed": [0]}}, tmp / "inv_spec.yaml")
    sets = _paths(tmp / "joint", tmp) + [
        f"data.dicom_root={c.data.dicom_root}", f"data.deeplesion_root={c.data.deeplesion_root}",
        f"data.info_csv={c.data.info_csv}", "data.max_slices_per_patient=4", f"data.norm.mu_ref={c.data.norm.mu_ref!r}",
        f"data.patients=[{', '.join(MAYO + DL)}]", *COMMON, *SMALL["regformer"], "device=cpu"]
    argv = ["invisible_two_stage.py", "-c", "configs/data/joint.yaml", "-s", *sets,
            "--spec", str(tmp / "inv_spec.yaml"), "--exp", "inv_mayo", "--geom", "t120", "--ref-exp", "p0_mayo",
            "--epochs", "2", "--batch", "2", "--channels", "8", "16", "--gn-groups", "4"]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path("experiments/analysis/invisible_two_stage.py", run_name="__main__")
    root = c.paths.outputs / "invisible"
    # the stage-1 run lives outside the main runs
    stage1 = next((root / "runs" / "inv_mayo").glob("*/DONE")).parent
    assert not (c.paths.runs / "inv_mayo").exists()
    m = pd.read_csv(stage1 / "metrics.csv")
    assert "val_psnr_full" in m.columns and np.isfinite(m.val_psnr).all()        # val_psnr = PSNR of the visible part
    assert "VISIBLE PART" in (stage1 / "train.log").read_text(encoding="utf-8")
    out = root / "two_stage" / "t120"
    sl, summ = pd.read_csv(out / "slices.csv"), pd.read_csv(out / "summary.csv")
    assert set(sl.split) == {"val", "test"} and set(sl[sl.split == "val"].pid) == {"L333"}
    cols = [f"psnr_{n}" for n in ("ghep", "tang1", "tran", "thuong")]
    assert np.isfinite(sl[cols].to_numpy()).all() and list(summ.split) == ["val", "test"]
    # replacing the hidden part by the truth can only help
    assert (sl.psnr_tran >= sl.psnr_tang1 - 1e-3).all()
    # both parts add up exactly to the MSE of the merged image
    assert np.allclose(sl.mse_thay_ghep + sl.mse_khuat_ghep,
                       10 ** (-sl.psnr_ghep / 10), rtol=1e-3, atol=1e-7)
    report = (out / "summary.md").read_text(encoding="utf-8")
    assert "Train thường (p0_mayo)" in report and "Hai tầng, ảnh ghép" in report and (out / "G.pt").is_file()
    # same command again: stage 1 is DONE and not retrained, stage 2 runs again
    before = (stage1 / "DONE").stat().st_mtime
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path("experiments/analysis/invisible_two_stage.py", run_name="__main__")
    assert (stage1 / "DONE").stat().st_mtime == before


@pytest.mark.parametrize(("model", "tag", "stage_cols"), [
    pytest.param("cg_glore", "cg_fbp", {"alphas"}, marks=pytest.mark.skipif(
        importlib.util.find_spec("lact.core.models.baselines.cg_glore") is None, reason="CG-GLORE is not distributed")),
    ("fbpconvnet", "post", set()), ("lpd", "pd", set()),
    ("learn_orig", "gd", {"alphas", "betas"})])
def test_compared_methods_run_through_the_same_pipeline(world, model, tag, stage_cols):
    """Methods of the CG-GLORE comparison table train, save, reload, evaluate and enter table C1 through the services of
    the existing runs (this file runs after the table tests, so the cmp_x experiment does not affect them)."""
    j = world["joint"]
    t, s = _fit(j("joint_mayo", ["exp=cmp_x"], model=model))
    assert s["arch"] == model and s["update"] == tag and s["metric"] is None
    assert t.run.run_id.startswith(f"cmp_x_{model}_") and 0 < s["n_params_regularizer"] <= s["n_params"]
    cols = set(pd.read_csv(t.run.metrics_csv).columns)
    assert stage_cols <= cols and "val_cg_iters" not in cols
    assert not ({"alphas", "betas", "weights"} - stage_cols) & cols
    dcfg = j("joint")
    assert evaluation.evaluate_run(t.run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is not None
    df = results_store(dcfg).load_slices(split="test", exp="cmp_x")
    df = df[df.arch == model]
    assert len(df) == 14 and set(df["update"]) == {tag} and np.isfinite(df.psnr).all()
    assert recheck.val_psnr(t.run, "cpu") == pytest.approx(s["best_val_psnr"], abs=1e-4)
    files = make_phase2_tables(dcfg, "test", [dcfg.require_geometry().name])
    c1 = pd.read_csv(files[0])
    label = {"cg_glore": "CG-GLORE", "fbpconvnet": "FBPConvNet", "lpd": "Learned Primal-Dual",
             "learn_orig": "LEARN (bản gốc)"}[model]
    row = c1[(c1["Thí nghiệm"] == "cmp_x") & (c1["Mô hình"] == label)]
    assert len(row) == 1 and row["Huấn luyện trên"].iloc[0] == "Mayo"


def test_dpma_trains_in_two_runs_like_the_original(world, small_geom):
    """DPMA through the services: a stage-1 run trains M1; a stage-2 run loads that M1 (frozen) and trains M2;
    evaluation adds the refinement iterations, while the validation PSNR recorded during training is that of the
    learned part."""
    j = world["joint"]
    base = j("joint_mayo", model="dpma")
    operators.build_operators(base, [small_geom.replace(name="t240", span_deg=240.0)], sizes=[32])
    t1, s1 = _fit(j("joint_mayo", ["exp=dpma1_x", "model.regularizer.stage=1"], model="dpma"))
    assert s1["arch"] == "dpma" and s1["update"] == "dual" and t1.model.m2 is None
    cfg2 = j("joint_mayo", ["exp=dpma_x", "train.init_from=dpma1_x"], model="dpma")
    assert find_init_run(cfg2).run_id == t1.run.run_id
    t2, s2 = _fit(cfg2)
    # only M2 learns, same size as M1
    assert s2["init_from"] == t1.run.run_id and s2["n_params"] == s1["n_params"]
    best1 = torch.load(t1.run.checkpoint("best"), map_location="cpu", weights_only=False)["model"]
    own = t2.model.state_dict()
    assert all(torch.equal(own[k], v) for k, v in best1.items()) and any(k.startswith("m2.") for k in own)
    with pytest.raises(ConfigError):            # without a stage-1 run, stage 2 cannot start
        Trainer(j("joint_mayo", ["exp=dpma_y", "train.init_from=khong_co"], model="dpma")).fit()
    assert recheck.val_psnr(t2.run, "cpu") == pytest.approx(s2["best_val_psnr"], abs=1e-4)
    dcfg = j("joint")
    assert evaluation.evaluate_run(t2.run, "test", device="cpu", batch_size=2, data_cfg=dcfg) is not None
    df = results_store(dcfg).load_slices(split="test", exp="dpma_x")
    assert len(df) == 14 and set(df["update"]) == {"dual"} and np.isfinite(df.psnr).all()
