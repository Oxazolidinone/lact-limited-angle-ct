"""Run the main workflow with one command, with explicit dependency barriers.

prepare (data, A, E0/E1, TV) -> E2 -> choose lr -> E3/E4/E5 -> E6 and reporting.
Only training runs within one queue run in parallel. Evaluation/index writing runs sequentially.
State lives in outputs/pipeline/main.json; each run's checkpoints are still managed by training.py.
"""
from __future__ import annotations

import logging
import math
import sys
import tempfile
from pathlib import Path

from ...config import Config, ConfigError, load_config, load_config_file, load_geometry, load_yaml
from ...config.loader import REPO_ROOT
from ...infra.data.cache import file_signature
from ...infra.io.files import load_json, save_json
from ...infra.io.hdf5_store import ImageStore
from ...infra.operators.operator_store import operator_path
from ...infra.stores.manifest import load_manifest, manifest_key, update_manifest
from ...infra.system.locks import file_lock
from ...infra.system.processes import INTERRUPTED, ProcessRunner
from ..reporting.tables import select_lr
from ..scheduling import Job, expand, order_jobs, sweep_sec_per_step
from ..training.trainer import run_store_for
from .stages import (
    CHECK_GEOMS,
    HALF_GEOMS,
    MAIN_EXPS,
    MAIN_GEOMS,
    SWEEP,
    UNTIL,
    Stage,
    _digest,
    _scientific_config,
    state_file,
)

log = logging.getLogger("lact")
class Pipeline:
    def __init__(self, cfg: Config, files: list[str], overrides: list[str], workers: int = 1,
                 gpus: list[str] | None = None, per_gpu: int = 1, extend: bool = False,
                 models: list[str] | None = None, job_mem_mb: dict[str, float] | None = None,
                 gpu_mem_mb: float | None = None, solo_gpus: list[str] | None = None):
        self.cfg, self.files, self.overrides = cfg, files, overrides
        self.workers, self.extend = workers, extend
        # job_mem_mb: {model.arch: peak GPU memory during training (MB), measured by `lact train pilot`}; together with
        # gpu_mem_mb (one GPU's memory) it packs runs: heavy runs alone, light runs share up to per_gpu runs per GPU.
        self.job_mem_mb, self.gpu_mem_mb = dict(job_mem_mb or {}), gpu_mem_mb
        self.solo_gpus = list(solo_gpus or [])        # UUIDs of GPUs that run one job at a time
        # models: run only the models whose model.arch is listed (empty = every model of the grid)
        self.models = sorted(set(models or []))
        # gpus: UUIDs of the GPUs to use; each job sees exactly one GPU, at most per_gpu jobs per GPU (workers = slots)
        self.gpus, self.per_gpu = list(gpus or []), per_gpu
        if self.gpus:
            self.workers = len(self.gpus) * per_gpu
        if workers < 1 or per_gpu < 1:
            raise ConfigError("workers, per_gpu must be >= 1.")
        if cfg.run_id or cfg.model or cfg.geometry or cfg.exp != "dev":
            raise ConfigError("The pipeline takes model/geometry/exp/run_id from each grid; "
                              "do not override them in the shared configuration.")
        self.path = state_file(cfg)
        self.state: dict = {}
        self.runner: ProcessRunner

    def _load_state(self) -> None:
        protocol = {str(p.relative_to(REPO_ROOT)): load_yaml(p)
                    for folder in ("experiment", "model", "geometry", "train")
                    for p in sorted((REPO_ROOT / "configs" / folder).glob("*.yaml"))}
        fingerprint = _digest([_scientific_config(self.cfg), protocol] + ([self.models] if self.models else []))
        self.state = load_json(self.path) if self.path.exists() else {}
        if self.state and self.state.get("fingerprint") != fingerprint:
            if not self.extend:
                raise ConfigError(
                    f"The configuration differs from the pipeline saved at {self.path}. Restore the old configuration "
                    "to resume, or use separate outputs/data/runs paths for a new experiment. To only EXTEND the "
                    "workflow on the same data (new architectures or runs in a grid), run again with --extend.")
            # Extension: the shared fingerprint is no longer trusted, but each old run is still compared with its
            # saved config.yaml (train) and data/matrices with their metadata (prepare).
            log.warning("--extend: accepting existing outputs for the extended workflow; old runs are still checked "
                        "against their saved configuration.")
            self.state.setdefault("previous_fingerprints", []).append(self.state["fingerprint"])
        self.state.update(fingerprint=fingerprint, status="running")
        self.state.setdefault("stages", {})
        self.state.setdefault("jobs", {})
        self._save()

    def _save(self) -> None:
        save_json(self.state, self.path)

    def _argv(self, args: list[str]) -> list[str]:
        return [sys.executable, "-m", "lact", *args, "-c", *self.files, "-s", *self.overrides]

    def _fresh(self, stage: Stage, inputs: dict) -> bool:
        """The stage succeeded AND its inputs are unchanged AND its outputs still exist unchanged."""
        previous = self.state["stages"].get(stage.key, {})
        artifacts = previous.get("artifacts", {})
        return (previous.get("status") == "done" and bool(artifacts) and previous.get("inputs") == inputs
                and all(Path(p).is_file() and file_signature(p) == sig for p, sig in artifacts.items()))

    def _record(self, stage: Stage, code: int, inputs: dict) -> int:
        outputs = stage.outputs + (sorted(p for p in stage.scan_dir.rglob("*") if p.is_file())
                                   if stage.scan_dir and stage.scan_dir.exists() else [])
        if code == 0 and any(not p.is_file() for p in outputs):
            log.error("%s exited with 0 but outputs are missing", stage.key)
            code = 1
        self.state["stages"][stage.key] = {
            "status": "done" if code == 0 else "interrupted" if code == INTERRUPTED else "failed",
            "code": code, "inputs": inputs,
            "artifacts": {str(p): file_signature(p) for p in outputs if p.is_file()},
        }
        return code

    def stages(self, group: list[Stage]) -> int:
        """Run INDEPENDENT stages in parallel (at most ``workers`` processes; one GPU per stage when GPUs are given).
        Finished stages are skipped. Returns 0 when every stage succeeds, 3 when stopped, else the first error code."""
        if self.runner.expired():
            return INTERRUPTED
        todo: list[tuple[Stage, dict]] = []
        for st in group:
            inputs = {str(p): file_signature(p) if p.is_file() else None for p in st.inputs}
            if self._fresh(st, inputs) and not st.always:
                log.info("Skipping finished stage: %s", st.key)
                continue
            self.state["stages"][st.key] = {"status": "running", "command": st.args}
            todo.append((st, inputs))
        if not todo:
            return 0
        self._save()
        codes = self.runner.run([(st.key, self._argv(st.args)) for st, _ in todo], self.workers)
        results = [self._record(st, codes.get(st.key, INTERRUPTED), inputs) for st, inputs in todo]
        self._save()
        if self.runner.interrupted:
            return INTERRUPTED
        return next((c for c in results if c), 0)

    def stage(self, stage: Stage) -> int:
        return self.stages([stage])

    def prepare(self) -> int:
        c = self.cfg
        # Reuse data created by earlier commands. Missing metadata means the preprocessed set is rebuilt for
        # consistency.
        image_files = [c.data.images_h5, c.data.stats_json, c.data.slices_json]
        if all(p.is_file() for p in image_files):
            stats = load_json(c.data.stats_json)
            expected = {"train_patients": c.split.train, "img_size_rec": c.data.img_size_rec,
                        "img_size_sim": c.data.img_size_sim, "mu_water_mm": c.data.mu_water_mm,
                        "hu_clip": list(c.data.hu_clip), "mode": c.data.norm.mode}
            norm_key = {"hu_window": "hu_max", "fixed": "mu_ref"}.get(c.data.norm.mode, "percentile")
            expected[norm_key] = getattr(c.data.norm, norm_key)
            if any(stats.get(k) != v for k, v in expected.items()):
                raise ConfigError("images.h5/stats.json do not match the data/split configuration; use separate "
                                  "outputs.")
            slices = load_json(c.data.slices_json)
            images = ImageStore(c.data.images_h5)
            if (slices.get("split") != {k: c.split.patients(k) for k in ("train", "val", "test")}
                    or images.mu_ref != stats["mu_ref"]
                    or set(images.patients()) != set(slices["patients"])):
                raise ConfigError("images.h5/stats.json/slices.json are out of sync.")
            # A session can die after writing the H5/JSON files but before updating the manifest.
            man = load_manifest(c.paths.manifest)
            missing = [p for p in image_files if manifest_key(p) not in man]
            if missing:
                update_manifest(c.paths.manifest, missing)
            log.info("Reusing preprocessed images: %s", c.data.images_h5)
            preprocess = None
        else:
            preprocess = Stage("preprocess", ["data", "preprocess", "--overwrite"], image_files)
        op_files = [operator_path(load_geometry(g), n, c.paths.operators)
                    for g in MAIN_GEOMS + CHECK_GEOMS + HALF_GEOMS
                    for n in (c.data.img_size_rec, c.data.img_size_sim)]
        man = load_manifest(c.paths.manifest)
        repair_operators_manifest = any(manifest_key(p) not in man for p in op_files)

        def ops_of(g: str) -> list[Path]:
            return [operator_path(load_geometry(g), n, c.paths.operators)
                    for n in (c.data.img_size_rec, c.data.img_size_sim)]

        # Dependency tiers: stages within a tier run in parallel; a tier waits for the previous one.
        # Building A needs only the geometry (not the images), so it runs alongside preprocessing when that is needed.
        levels = [
            [Stage("operators", ["ops", "build", "-g", *MAIN_GEOMS, *CHECK_GEOMS, *HALF_GEOMS],
                   op_files + [p.with_suffix(".json") for p in op_files], always=repair_operators_manifest)],
            # E0 needs only A; simulation and E1 need images + A; the three geometries are independent
            [Stage("e0", ["ops", "check"], [c.paths.checks / "e0_report.json"], inputs=op_files)]
            + [Stage(f"simulate_{g}", ["data", "simulate", "-g", g], [c.paths.sino_file(g)],
                     always=manifest_key(c.paths.sino_file(g)) not in man, inputs=image_files + ops_of(g))
               for g in MAIN_GEOMS]
            + [Stage("e1", ["ops", "density"], [c.paths.checks / "e1_summary.json",
                                              c.paths.checks / "e1_angle_density.csv"],
                     inputs=image_files + op_files)],
            # TV tuning of a geometry needs only that geometry's sinograms
            [Stage(f"tv_{g}", ["eval", "tune-tv", "-g", g], [c.paths.baselines / f"tv_best_{g}.json"],
                   inputs=[c.data.images_h5, c.paths.sino_file(g)]) for g in MAIN_GEOMS],
            [Stage("stage_cache", ["data", "stage", "-g", *MAIN_GEOMS], always=True)],
        ]
        if preprocess is not None:
            levels[0].insert(0, preprocess)
        for group in levels:
            code = self.stages(group)
            if code:
                return code
        return 0

    def jobs(self, names: list[str]) -> list[Job]:
        jobs = [job for name in names for job in expand(f"configs/experiment/{name}.yaml", self.files, self.overrides,
                                                        require_lr=name != SWEEP, models=self.models or None)]
        if self.models and not jobs:
            raise ConfigError(f"--models {self.models}: no model of the grid {names} matches")
        return jobs

    def train(self, jobs: list[Job], sweep: bool = False) -> int:
        """The barrier opens only when EVERY expected job has ended with the right status."""
        stores = {}
        for job in jobs:
            cfg = load_config(job.files, job.overrides, use_defaults=False)
            store = run_store_for(cfg)
            if store.config_file.exists():
                saved = load_config_file(store.config_file)
                if _scientific_config(saved) != _scientific_config(cfg):
                    raise ConfigError(f"{store.dir}: configuration changed; refusing to resume a mismatched "
                                      "checkpoint.")
            stores[job.run_id] = store
        if len(stores) != len(jobs):
            raise ConfigError("The grid has duplicate run_ids (check seed/train.lr overrides); cannot run in parallel.")

        def record(key: str, code: int) -> None:
            self.state["jobs"][key] = {"status": stores[key].status, "code": code}
            self._save()

        for key, store in stores.items():
            self.state["jobs"][key] = {"status": store.status}
        self._save()
        todo = [j for j in jobs if stores[j.run_id].status in ("todo", "partial")]
        pending = [(j.run_id, j.argv) for j in order_jobs(todo, stores, sweep_sec_per_step(self.cfg.paths.runs, SWEEP))]
        weights = self.gpu_shares(todo)
        if weights:
            # Runs needing a whole GPU go first (stable sort): they take free GPUs at once and light runs share the
            # remaining GPUs, instead of every light run holding a GPU while heavy runs wait for a free one.
            pending.sort(key=lambda item: weights.get(item[0], 1.0) <= 0.5)
            alone = sum(w > 0.5 for w in weights.values())
            log.info("Packing runs by measured memory: %d runs need a GPU of their own, %d share up to %d runs per GPU",
                     alone, len(weights) - alone, self.per_gpu)
            self.runner.run(pending, self.workers, record, weights=weights)
        else:
            self.runner.run(pending, self.workers, record)
        if self.runner.interrupted:
            return INTERRUPTED
        allowed = {"done", "diverged"} if sweep else {"done"}
        bad = [f"{key}: {r.status}" for key, r in stores.items()
               if r.status not in allowed or not r.summary()
               or (r.status == "done" and not r.checkpoint("best").is_file())]
        if bad:
            log.error("Results are incomplete for the dependent stage:\n%s", "\n".join(bad))
            return 1
        return 0

    def gpu_shares(self, jobs: list[Job], budget: float = 0.85) -> dict[str, float]:
        """{run_id: GPU share} = peak memory of the architecture / (``budget`` x GPU memory); unmeasured
        architectures => 1 (run alone). Empty when not running several runs per GPU or without measurements."""
        if not (self.gpus and self.per_gpu > 1 and self.job_mem_mb and self.gpu_mem_mb):
            return {}
        cap = budget * float(self.gpu_mem_mb)
        return {j.run_id: min(1.0, self.job_mem_mb[j.arch] / cap) if j.arch in self.job_mem_mb else 1.0
                for j in jobs}

    def finish(self, jobs: list[Job]) -> int:
        c = self.cfg
        run_dirs = [str(run_store_for(load_config(j.files, j.overrides, use_defaults=False)).dir) for j in jobs]
        # The three evaluation commands are independent (index.json is locked), so they run in parallel on the GPUs.
        # Cost is measured ALONE so inference timings are not disturbed by other jobs; reporting runs last.
        levels = [
            [Stage("eval_runs", ["eval", "runs", "--runs", *run_dirs, "--split", "test"], always=True),
             Stage("eval_baselines", ["eval", "baselines", "--split", "test"], always=True),
             Stage("e6", ["eval", "cross", "configs/experiment/e6_crossgeo.yaml", "--split", "test"],
                   always=True)],
            # inputs = the runs' summaries: adding runs (extending a grid) triggers a new cost measurement
            [Stage("cost", ["eval", "cost", "--exp", "e3_main", "e4_matched"],
                   [c.paths.tables / "cost_e3_main_e4_matched.csv"],
                   inputs=[Path(d) / "summary.json" for d in run_dirs])],
        ]
        # reporting depends on the evaluation store and the cost table: new results regenerate it
        report_inputs = [c.paths.eval / "index.json", c.paths.tables / "cost_e3_main_e4_matched.csv"]
        levels += [[Stage(f"report_{name}", ["report", name, "--split", "test"], scan_dir=folder,
                          inputs=report_inputs)]
                   for name, folder in (("tables", c.paths.tables), ("figures", c.paths.figures),
                                        ("analysis", c.paths.figures))]
        for group in levels:
            code = self.stages(group)
            if code:
                return code
        return 0

    def _execute(self, until: str) -> int:
        code = self.prepare()
        if code or until == "prepare":
            return code
        sweep = self.jobs([SWEEP])
        code = self.train(sweep, sweep=True)
        if code:
            return code
        # Ignore summaries of runs outside the grid; never choose an lr from an unfinished E2.
        runs = [run_store_for(load_config(j.files, j.overrides, use_defaults=False)) for j in sweep]
        select_lr(self.cfg, SWEEP, runs=runs)
        if until == "sweep":
            return 0
        code = self.stage(Stage("freeze", ["eval", "freeze", "--note", "pipeline: E2 and TV are complete"],
                                [self.cfg.paths.outputs / "configs_frozen.lock"]))
        if code:
            return code
        jobs = self.jobs(MAIN_EXPS)
        code = self.train(jobs)
        if code or until == "train":
            return code
        return self.finish(jobs)

    def run(self, until: str = "all", max_hours: float | None = None) -> int:
        if until not in UNTIL or (max_hours is not None and (max_hours <= 0 or not math.isfinite(max_hours))):
            raise ConfigError("Invalid --until, or --max-hours is not a finite positive number.")
        lock = Path(tempfile.gettempdir()) / "lact-locks" / (_digest(str(self.path.resolve())) + ".lock")
        with file_lock(lock, blocking=False), \
                ProcessRunner(REPO_ROOT, max_hours, gpus=self.gpus or None, per_gpu=self.per_gpu,
                              solo_gpus=self.solo_gpus) as runner:
            self.runner = runner
            self._load_state()
            try:
                code = self._execute(until)
                self.state.update(status="done" if code == 0 and until == "all" else
                                  "ready" if code == 0 else "interrupted" if code == INTERRUPTED else "failed",
                                  until=until, code=code)
                self._save()
                return code
            except Exception as exc:
                self.state.update(status="failed", error=str(exc))
                self._save()
                raise
