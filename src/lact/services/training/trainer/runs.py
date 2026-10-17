"""Identity of a training run: its id, its run directory, and the finished run a new run may initialise from."""
from __future__ import annotations

from ....config import Config, ConfigError
from ....infra.stores.run_store import RunStore, list_runs

EXIT_INTERRUPTED = 3


def make_run_id(cfg: Config) -> str:
    m = cfg.require_model()
    return f"{cfg.exp}_{m.arch}_{cfg.require_geometry().name}_lr{float(cfg.train.lr):g}_s{int(cfg.seed)}"


def run_store_for(cfg: Config) -> RunStore:
    return RunStore(cfg.paths.runs / cfg.exp / (cfg.run_id or make_run_id(cfg)))


def find_init_run(cfg: Config) -> RunStore:
    """Run used for initialisation. ``train.init_from`` = one or more experiment names separated by commas; the FIRST
    experiment with a DONE run of the same architecture and geometry is used. If that experiment has several such runs
    (several seeds), the run with the same seed is taken; with no same-seed run and more than one candidate, it
    fails."""
    arch, geom, seed = cfg.require_model().arch, cfg.require_geometry().name, int(cfg.seed)
    exps = [e.strip() for e in str(cfg.train.init_from).split(",") if e.strip()]
    for exp in exps:
        hits = [(run, run.summary()) for run in list_runs(cfg.paths.runs, exp)
                if run.status == "done" and run.checkpoint("best").is_file()]
        hits = [(run, s) for run, s in hits if (s.get("arch"), s.get("geometry")) == (arch, geom)]
        same_seed = [run for run, s in hits if int(s.get("seed", -1)) == seed]
        if len(same_seed) == 1 or len(hits) == 1:
            return same_seed[0] if len(same_seed) == 1 else hits[0][0]
        if hits:
            raise ConfigError(f"train.init_from={exp}: {len(hits)} DONE runs of ({arch}, {geom}) but none (or more "
                              f"than one) with seed {seed}; keep a single run in {cfg.paths.runs / exp}")
    raise ConfigError(f"train.init_from={cfg.train.init_from}: no DONE run of ({arch}, {geom}) in "
                      f"{[str(cfg.paths.runs / e) for e in exps]}")
