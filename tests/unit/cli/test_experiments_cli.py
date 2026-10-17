import pytest

from lact.cli.main import build_parser
from lact.services.scheduling import expand

GRIDS = {"base_mayo": 4, "base_dl": 4, "base_dt": 2, "lbfgs_mayo": 2, "lbfgs_dl": 2, "lbfgs_dt": 2, "cmp_mayo": 6,
         "cmp_dl": 6, "cmp_dt": 6, "dpma1_mayo": 2, "dpma1_dl": 2, "dpma1_dt": 2, "dpma_mayo": 2, "dpma_dl": 2,
         "dpma_dt": 2}


@pytest.mark.parametrize(("name", "n"), list(GRIDS.items()))
def test_expand_experiments(name, n):
    jobs = expand(f"configs/experiment/{name}.yaml")
    assert len(jobs) == n and len({j.run_id for j in jobs}) == n
    assert {j.run_id.split("_s")[-1] for j in jobs} == {"0"} and all(j.run_id.startswith(f"{name}_") for j in jobs)
    assert jobs[0].argv[1:5] == ["-m", "lact", "train", "run"]


def test_grid_overrides_reach_every_run():
    small = expand("configs/experiment/lbfgs_mayo.yaml", overrides=["model.K=3", "train.max_steps=6"])
    assert len(small) == 2 and all("model.K=3" in j.overrides for j in small)
    assert {j.arch for j in small} == {"learn_densenet"}


@pytest.mark.parametrize("argv", [
    ["data", "preprocess", "--overwrite"], ["ops", "build", "-g", "fan_a90"], ["train", "run", "-m", "learn_densenet"],
    ["train", "grid", "configs/experiment/lbfgs_mayo.yaml", "--dry-run"], ["eval", "runs", "--exp", "lbfgs_mayo"],
    ["eval", "baselines", "--names", "fbp"], ["eval", "recheck", "--tol", "0.05"], ["train", "status"],
])
def test_cli_parses(argv):
    a = build_parser().parse_args(argv)
    assert callable(a.func)


@pytest.mark.parametrize("group", ["pipeline", "report", "demo"])
def test_removed_groups_are_gone(group):
    with pytest.raises(SystemExit):
        build_parser().parse_args([group])
