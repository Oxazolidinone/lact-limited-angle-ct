import pytest

from lact.cli.main import build_parser
from lact.config import load_config
from lact.services.scheduling import expand, remaining_work


def test_expand_experiments():
    jobs = expand("configs/experiment/e3_main.yaml")
    assert len(jobs) == 30 and len({j.run_id for j in jobs}) == 30          # 15 models x 2 geometries x 1 seed
    assert {j.run_id.split("_s")[-1] for j in jobs} == {"0"}                # seeds 1, 2 kept only for finished runs
    assert jobs[0].argv[1:5] == ["-m", "lact", "train", "run"]
    sweep = expand("configs/experiment/e2_sweep.yaml")
    assert len(sweep) == 45 and all("fan_a120" in j.run_id for j in sweep)
    assert len(expand("configs/experiment/e4_matched.yaml")) == 3
    assert len(expand("configs/experiment/e5_a150.yaml")) == 15
    # model.* overrides (as in `make smoke`) must work with grids taking their lr from E2
    small = expand("configs/experiment/e3_main.yaml", overrides=["model.K=3", "train.max_steps=6"])
    assert len(small) == 30 and "model.K=3" in small[0].overrides


def test_remaining_work_uses_measured_speed_per_architecture():
    """Remaining GPU hours: steps of each run x the architecture's seconds/step; unmeasured ones take the median."""
    jobs = expand("configs/experiment/e4_matched.yaml")
    rows = remaining_work(jobs, {"learn_resnet_pm2m": 0.1}, n_train=1000)
    assert len(rows) == 3 and {r["exp"] for r in rows} == {"e4_matched"}
    epochs = load_config(jobs[0].files, jobs[0].overrides, use_defaults=False).train.epochs
    measured = [r for r in rows if r["arch"] == "learn_resnet_pm2m"]
    assert len(measured) == 1 and measured[0]["measured"] and measured[0]["steps_left"] == epochs * 1000
    assert measured[0]["gpu_hours"] == pytest.approx(0.1 * epochs * 1000 / 3600)
    others = [r for r in rows if r["arch"] != "learn_resnet_pm2m"]
    assert not any(r["measured"] for r in others) and {r["gpu_hours"] for r in others} == {measured[0]["gpu_hours"]}
    jobs[0].status = "done"                                   # finished runs are no longer counted
    assert len(remaining_work(jobs, {}, n_train=1000)) == 2


@pytest.mark.parametrize("argv", [
    ["data", "preprocess", "--overwrite"], ["ops", "check", "-g", "fan_full"], ["train", "run", "-m", "learn_cnn"],
    ["train", "grid", "configs/experiment/e3_main.yaml", "--dry-run"], ["eval", "runs", "--exp", "e3_main"],
    ["eval", "baselines", "--names", "fbp"], ["report", "tables", "--n-boot", "100"], ["demo", "--port", "1"],
    ["train", "plan", "--pilot", "pilot.csv", "--n-gpus", "4"], ["eval", "recheck", "--tol", "0.05"],
])
def test_cli_parses(argv):
    a = build_parser().parse_args(argv)
    assert callable(a.func)
