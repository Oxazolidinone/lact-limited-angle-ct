# Source architecture (v0.4)

## 1. Principles

- **Layers with one-way dependencies.** An upper layer may call a lower one, never the reverse:

  ```
  cli ──► services ──► datasets ──► infra ──► config ──► core
  ```

  The rule lives in `pyproject.toml` (`[tool.importlinter]`) and is checked by `lint-imports` (in `make check-all` and
  the pre-commit hooks).
- **A pure `core`.** Only numpy/scipy/torch/pandas; no file I/O, no configuration dictionaries. Functions take
  primitive values or domain objects (`FanBeamGeometry`, `SparseFanBeam`, tensors).
- **Typed configuration, checked at the boundary.** YAML is read once (`config.load_config`) and converted into a
  `Config` dataclass. Unknown keys, wrong types, a patient in two splits... fail at load time. Code inside uses
  `cfg.train.lr`, never `cfg["train"]["lr"]`.
- **One composition root.** `services/factories.py` is the only place that turns a `Config` into concrete objects
  (projector, FBP, model, dataset, results store). Other services call these functions.
- **Stores instead of file-name conventions.** `RunStore` (one training run) and `ResultsStore` (evaluation results +
  `index.json`) own the directory layout; nothing else builds paths or parses file names.
- **One entry point.** `lact <group> <command>`; the `Makefile` is only a set of shortcuts. Experiment grids run each
  run in a child process `python -m lact train run ...` to isolate GPU memory and to move easily to SLURM.

## 2. Layers

| Layer | Directory | Responsibility | May import |
| --- | --- | --- | --- |
| Domain | `core/` | `imaging/` (fan-beam geometry, phantoms, noise, intensity conversion, augmentation), `operators/` (sparse operator with autograd, `fbp/` filters and operator, ‖A‖), `models/` (`unrolled/` LEARN framework with the gd and L-BFGS rules, `regularizers/` DenseNet encoder–decoder and the 3-layer CNN of the original LEARN, `baselines/` RegFormer, DPMA, LPD, FBPConvNet, `registry`), `solvers/` (L-BFGS direction, TV-PDHG, Chambolle TV), `metrics/` (image metrics, body masks) | numpy, scipy, torch, pandas |
| Configuration | `config/` | Read/merge YAML (`_base_`, `-s` overrides), the `schema/` dataclasses (one module per group of sections), `ExperimentSpec` | `core`, yaml |
| Infrastructure | `infra/` | `io/` (JSON/SHA, HDF5 stores), `operators/` (projection matrices: ASTRA only in `astra_matrix.py`, Siddon fallback, save/load), `stores/` (`RunStore`: checkpoints, heartbeat, sessions, events; `ResultsStore`; SHA-256 manifest), `data/` (`LACTDataset`, local cache), `system/` (seed/log/device/`GracefulInterrupt`, process queue, file locks, GPU selection) | `config`, `core` |
| Datasets | `datasets/` | Raw data of each dataset: `dicom.py` (Mayo DICOM), `mayo.py` (download), `deeplesion/` (PNG reader, download, splits), `detect/` (2DeteCT raw sinograms, LION split, Zenodo partial downloads), `box.py` and `http_range.py` (shared download code) | `infra`, `config`, `core` |
| Application | `services/` | `data/` (preprocessing by `data.format`, 2DeteCT use cases, simulation + noise, synthetic data, input staging), `operators/` (build A), `training/` (`trainer/` loop, optimizer, monitoring), `scheduling/` (experiment grids, child-process runner, queue order), `evaluation/` (per-slice evaluation, FBP/TV baselines and TV tuning, validation recheck); `factories.py` (composition root) | every lower layer |
| Interface | `cli/` | Command-line parsing and calling services | every lower layer |

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
lact data stage                -> services.data.staging.stage_all             -> infra.data.cache.stage_file
lact train run                 -> services.training.trainer.Trainer           -> factories.model_for/dataset_for, RunStore
lact train grid                -> services.scheduling.expand/run_jobs         (child processes `lact train run`)
lact train status              -> services.training.monitor.run_status        -> RunStore.heartbeat/summary
lact eval tune-tv              -> services.evaluation.baselines.tune_tv       -> core.solvers.tv_pdhg, validation split
lact eval freeze               -> services.evaluation.evaluate.freeze_configs -> records the configuration before testing
lact eval runs                 -> services.evaluation.evaluate.evaluate_run   -> factories.load_trained, ResultsStore
lact eval baselines            -> services.evaluation.baselines               -> FBP and TV, ResultsStore
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
| An experiment grid | `configs/experiment/<name>.yaml` (models × geometries × seeds, lr table, overrides) | Name it in the `Makefile` grid and evaluation lists |
| An image metric | `core/metrics/image.py` (add it to `all_metrics`) | Appears in the evaluation CSVs automatically |
| A classical baseline | The algorithm in `core/solvers/`, registered in `services/evaluation/baselines.baseline_method` | Add the name to `BASELINES` |
| A configuration field | The matching dataclass in `config/schema/` + its default in YAML | Unknown keys are rejected, so declare it |
| A storage format | `infra/` (one adapter) | Services only go through the adapter |
| A dataset | Reader/downloader in `datasets/<name>/`, a branch in `services/data/prep` by `data.format`, `configs/data/<name>.yaml` + `configs/variant/<name>.yaml` | Return (HU, meta) like `datasets.dicom.read_series`; own outputs directory |
| A measurement condition (noise...) | `core/imaging/noise.py` + the `noise` section | Add `configs/variant/<name>.yaml` with its own outputs and cache |
| A CLI command | `cli/commands/<group>.py` (a `cmd_*` function registered in `register`) | A command only loads the configuration and calls one service |

## 5. Quality checks

| Command | Content |
| --- | --- |
| `make lint` | ruff (pyflakes, pycodestyle, isort, bugbear, pyupgrade, pydocstyle with the Google convention on `src/`) |
| `make typecheck` | mypy on `src/lact` (with `check_untyped_defs`) |
| `make arch` | lint-imports: the layers, a pure `core`, ASTRA only in `infra.operators.astra_matrix` |
| `make test-unit` | fast tests without ASTRA or data |
| `make test` | every test; the whole flow and the data variants run on synthetic data (ASTRA only for the matrix/FBP cross-checks) |
| `make test-regression` | finished runs: 6 trained checkpoints (one per method on 2DeteCT) reproduce their outputs and 48 saved run configurations load unchanged (needs `LACT_RESULTS_DIR`) |
| `make smoke` | data preparation, training (first-order and L-BFGS) and evaluation on synthetic data |
