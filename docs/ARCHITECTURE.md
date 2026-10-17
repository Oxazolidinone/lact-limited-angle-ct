# Source architecture (v0.3)

## 1. Principles

- **Layers with one-way dependencies.** An upper layer may call a lower one, never the reverse:

  ```
  cli ──► services ──► datasets ──► infra | viz ──► config ──► core
  ```

  `infra` and `viz` are independent of each other. The rule lives in `pyproject.toml` (`[tool.importlinter]`) and is
  checked by `lint-imports` (in `make check-all` and the pre-commit hooks).
- **A pure `core`.** Only numpy/scipy/torch/pandas; no file I/O, no configuration dictionaries, no plotting. Functions
  take primitive values or domain objects (`FanBeamGeometry`, `SparseFanBeam`, tensors).
- **Typed configuration, checked at the boundary.** YAML is read once (`config.load_config`) and converted into a
  `Config` dataclass. Unknown keys, wrong types, a patient in two splits... fail at load time. Code inside uses
  `cfg.train.lr`, never `cfg["train"]["lr"]`.
- **One composition root.** `services/factories.py` is the only place that turns a `Config` into concrete objects
  (projector, FBP, model, dataset, results store). Other services call these functions.
- **Stores instead of file-name conventions.** `RunStore` (one training run) and `ResultsStore` (evaluation results +
  `index.json`) own the directory layout; nothing else builds paths or parses file names.
- **One entry point.** `lact <group> <command>`; the `Makefile` is only a set of shortcuts. Experiment grids run each
  run in a child process `python -m lact train run ...` to isolate GPU memory and to move easily to SLURM.
- **Thesis code stays outside the library.** `experiments/` (analyses, figures, paper numbers) imports `lact`; `lact`
  never imports `experiments` (import-linter contract).

## 2. Layers

| Layer | Directory | Responsibility | May import |
| --- | --- | --- | --- |
| Domain | `core/` | `imaging/` (fan-beam geometry, phantoms, noise, intensity conversion, augmentation), `operators/` (sparse operator with autograd, `fbp/` filters and operator, ‖A‖), `models/` (`unrolled/` LEARN framework and model interface, `regularizers/`, `baselines/` RegFormer, CG-GLORE, DPMA, LPD, FBPConvNet, `registry`), `solvers/` (CG, L-BFGS, TV-PDHG, Chambolle TV), `metrics/` (image metrics, statistics), `analysis/` (frequency, visibility, ERF, per stage, cost) | numpy, scipy, torch, pandas |
| Configuration | `config/` | Read/merge YAML (`_base_`, `-s` overrides), the `schema/` dataclasses (one module per group of sections), `ExperimentSpec` | `core`, yaml |
| Infrastructure | `infra/` | `io/` (JSON/SHA, HDF5 stores), `operators/` (projection matrices: ASTRA only in `astra_matrix.py`, Siddon fallback, save/load), `stores/` (`RunStore`: checkpoints, heartbeat, sessions, events; `ResultsStore`; SHA-256 manifest), `data/` (`LACTDataset`, local cache), `system/` (seed/log/device/`GracefulInterrupt`, process queue, file locks, GPU selection) | `config`, `core` |
| Presentation | `viz/` | Figures (matplotlib) and tables (CSV/MD/LaTeX) from numpy/pandas; `paper/` holds the paper-quality figures (`figures/` package), style and LaTeX tables | (nothing) |
| Datasets | `datasets/` | Raw data of each dataset: `dicom.py` (Mayo DICOM), `mayo.py` (download), `deeplesion/` (PNG reader, download, splits), `detect/` (2DeteCT raw sinograms, LION split, Zenodo partial downloads), `box.py` and `http_range.py` (shared download code) | `infra`, `viz`, `config`, `core` |
| Application | `services/` | `data/` (preprocessing by `data.format`, 2DeteCT use cases, simulation + noise, synthetic data, input staging), `operators/` (build A, E0 and E1 checks), `training/` (`trainer/` loop, optimizer, monitoring), `scheduling/` (experiment grids, child-process runner, queue order and estimates), `evaluation/` (evaluation, baselines, E6, cost), `reporting/` (aggregation, `tables/`, figures, analyses, phase-2 tables), `orchestration/` (`lact pipeline`); `factories.py` (composition root), `demo.py` | every lower layer |
| Interface | `cli/` | Command-line parsing, calling services, the Gradio interface | every lower layer |

## 3. Main flows

```
lact data download-mayo        -> datasets.mayo.download_mayo                 -> datasets.box.BoxShare (resumable)
lact data download-deeplesion  -> datasets.deeplesion.download_deeplesion     -> BoxShare, MD5 check
lact data split-deeplesion     -> datasets.deeplesion.make_split              -> writes the split YAML
lact data download-2detect     -> datasets.detect.download_slices             -> HTTP Range reads of the Zenodo zips
lact data preprocess           -> services.data.prep.preprocess               -> datasets.dicom | datasets.deeplesion |
                                                                                 services.data.detect (by data.format)
lact ops build                 -> services.operators.build_operators          -> infra.operators.operator_store
lact data simulate             -> services.data.simulation.simulate_all       -> factories.projector_for/fbp_for
lact train run                 -> services.training.trainer.Trainer           -> factories.model_for/dataset_for, RunStore
lact train grid                -> services.scheduling.expand/run_jobs         (child processes `lact train run`)
lact data stage                -> services.data.staging.stage_all             -> infra.data.cache.stage_file
lact train status              -> services.training.monitor.run_status        -> RunStore.heartbeat/summary
lact eval runs                 -> services.evaluation.evaluate.evaluate_run   -> factories.load_trained, ResultsStore
lact report tables             -> services.reporting.make_tables              -> ResultsStore.load_slices, viz tables
lact pipeline run              -> services.orchestration.pipeline.Pipeline    (stages in orchestration/stages.py)
```

### Cache and protection against lost work

- `cfg` always keeps the paths of the original data. `services.data.staging.local_input` returns the copy in
  `cache.dir` when `cache.enabled`; the factories (`dataset_for`, `projector_for`, `image_store`, `sino_store`) all
  go through it, so a run's `config.yaml` never contains temporary paths.
- `write_via_cache` builds large files in the local cache, then `publish_file` moves them to the outputs
  (`.partial` + `os.replace`).
- `Trainer` saves `last.pt` by time/steps together with `EpochProgress` (epoch, batch, loss sum) to resume at the exact
  position; `GracefulInterrupt` turns SIGTERM/SIGINT into a flag checked after every step.

## 4. Where does new code go?

| To add... | Put it in | Notes |
| --- | --- | --- |
| A regularizer | `core/models/regularizers/<name>.py` + `@register("<name>")` or an entry in `REGISTRY` | Add `configs/model/learn_<name>.yaml`; the unrolled framework needs no change |
| A compared method | `core/models/baselines/<name>.py`, built in `core/models/registry.build_compared` | Add its tag to `COMPARED_TAGS` (`config/schema/model.py`) and a `configs/model/<name>.yaml` |
| An image metric | `core/metrics/image.py` (add it to `all_metrics`) | Appears in the evaluation CSVs automatically |
| A classical baseline | The algorithm in `core/solvers/`, registered in `services/evaluation/baselines.baseline_method` | Add the name to `BASELINES` |
| A configuration field | The matching dataclass in `config/schema/` + its default in YAML | Unknown keys are rejected, so declare it |
| A storage format | `infra/` (one adapter) | Services only go through the adapter |
| A dataset | Reader/downloader in `datasets/<name>/`, a branch in `services/data/prep` by `data.format`, `configs/data/<name>.yaml` + `configs/variant/<name>.yaml` | Return (HU, meta) like `datasets.dicom.read_series`; own outputs directory |
| A measurement condition (noise...) | `core/imaging/noise.py` + the `noise` section | Add `configs/variant/<name>.yaml` with its own outputs and cache |
| A table or figure | A `table_*` function in `services/reporting/tables/` or a figure in `services/reporting/figures.py`; drawing code in `viz/` | Tables are pure functions returning a DataFrame, easy to test |
| A thesis-only analysis | `experiments/<area>/` | Runs by path or with `python -m experiments...`; never imported by `lact` |
| A CLI command | `cli/commands/<group>.py` (a `cmd_*` function registered in `register`) | A command only loads the configuration and calls one service |

## 5. Map from earlier layouts

### v0.2 -> v0.3 (October 2026 refactor)

| v0.2 | v0.3 |
| --- | --- |
| `tools/` | `scripts/` (operations) and `experiments/` (thesis analyses, figures, paper numbers); `tools` is a temporary link to `scripts` |
| `infra/physics/` | `infra/operators/` |
| `services/physics/operators.py` | `services/operators/` (`build.py`, `checks.py`) |
| `config/schema.py` | `config/schema/` (`base`, `storage`, `data`, `model`, `train`, `baseline`, `root`) |
| `core/operators/fbp.py` | `core/operators/fbp/` (`filters.py`, `operator.py`) |
| `core/models/{regformer,cg_glore,dpma,lpd,fbpconvnet}.py` | `core/models/baselines/` (DPMA: `dpma/network.py`, `dpma/model.py`) |
| `core/models/unrolled.py` | `core/models/unrolled/` (`base.py`, `learn.py`) |
| `infra/io/{box,dicom,deeplesion}.py`, `services/data/{mayo,deeplesion}.py` | `datasets/` (`box.py`, `dicom.py`, `mayo.py`, `deeplesion/`) |
| `services/data/detect.py` | `datasets/detect/` (raw data, split, Zenodo) + `services/data/detect.py` (use cases) |
| `services/training/trainer.py` | `services/training/trainer/` (`runs.py`, `guards.py`, `loop.py`) |
| `services/training/experiments.py` | `services/scheduling/` (`jobs.py`, `runner.py`, `planning.py`) |
| `services/reporting/tables.py` | `services/reporting/tables/` (`format`, `results`, `cost`, `significance`, `make`) |
| `viz/paper/figures.py` | `viz/paper/figures/` (`layout`, `images`, `metrics`, `training`) |

Every package re-exports the names of the module it replaces (e.g. `from lact.services.training.trainer import
Trainer`), and run names, the outputs layout, YAML keys, `nn.Module` attribute names and CSV columns are unchanged, so
finished runs load as before (`tests/regression`).

### v0.1 -> v0.2

| v0.1 | Now |
| --- | --- |
| `physics/geometry.py`, `phantom.py` | `core/imaging/geometry.py`, `core/imaging/phantom.py` |
| `physics/projector.py`, `fbp.py`, `opnorm.py` | `core/operators/` |
| `physics/sparse_builder.py` | `infra/operators/astra_matrix.py` (ASTRA) + `infra/operators/operator_store.py` (save/load) |
| `models/*` | `core/models/*` (`build_model(cfg)` -> `core.models.build_unrolled(...)` + `services.factories.model_for(cfg)`) |
| `baselines/tv_pdhg.py`, `fbp_baseline.py` | `core/solvers/tv_pdhg.py`, `services/evaluation/baselines.py` |
| `data/dicom_io.py` | `datasets/dicom.py` |
| `data/preprocess.py` | `core/imaging/intensity.py` (pure functions) + `services/data/prep.py` |
| `data/simulate.py`, `data/noise.py` | `services/data/simulation.py`, `core/imaging/noise.py` |
| `data/dataset.py`, `data/augment.py` | `infra/data/dataset.py`, `core/imaging/augment.py` |
| `data/split.py` | `config.SplitConfig` + `services/data/prep.check_uid_disjoint` |
| `data/manifest.py` | `infra/stores/manifest.py` |
| `train/trainer.py`, `optim.py` | `services/training/trainer/`, `services/training/optim.py` |
| `train/checkpoint.py`, `logger.py` | `infra/stores/run_store.py` (`RunStore`, `RunLogger`) + `services.factories.load_trained` |
| `eval/metrics.py`, `masks.py`, `stats.py` | `core/metrics/` |
| `eval/wedge.py`, `erf.py`, `stagewise.py`, `timing.py` | `core/analysis/` |
| `eval/evaluate.py` | `services/evaluation/evaluate.py` + `infra/stores/results_store.py` |
| `eval/aggregate.py` | `services/reporting/aggregate.py` |
| `utils/config.py` | `config/loader.py` (+ `schema/`, `defaults.py`, `experiment.py`) |
| `utils/io.py`, `seed.py`, `env.py` | `infra/io/files.py`, `infra/system/runtime.py` |
| `utils/cli.py` | `cli/common.py` |
| `scripts/00, 01, 04, 90` | `lact data inspect / preprocess / simulate / fake` |
| `scripts/02, 03, 05` | `lact ops build / check / density` |
| `scripts/10, 11, 21 pilot` | `lact train run / grid / pilot` |
| `scripts/20, 22, 23, 21 runs` | `lact eval freeze, runs, baselines / tune-tv / cross / cost` |
| `scripts/30, 31, 32` | `lact report select-lr, tables / figures / analysis` |
| `demo/app.py` | `lact demo` (`services/demo.py` + `cli/demo_app.py`) |
| `outputs/eval/per_slice/<tag>.csv` (looked up by file-name globs) | unchanged + `outputs/eval/index.json` (looked up by metadata) |

## 6. Quality checks

| Command | Content |
| --- | --- |
| `make lint` | ruff (pyflakes, pycodestyle, isort, bugbear, pyupgrade, pydocstyle with the Google convention on `src/`) |
| `make typecheck` | mypy on `src/lact` (with `check_untyped_defs`) |
| `make arch` | lint-imports: the layers, a pure `core`, ASTRA only in `infra.operators.astra_matrix`, `lact` never imports `experiments` |
| `make test-unit` | fast tests without ASTRA or data |
| `make test` | every test; pipelines and data variants run on synthetic data (ASTRA only for the matrix/FBP cross-checks) |
| `make test-regression` | finished thesis runs: 9 checkpoints, the paper tables, 246 run configurations (needs `../ket-qua`) |
| `make smoke` | every `lact` command on synthetic data |
