# Changelog

## 0.4.0 (2026-10-17)

The repository now holds only what the comparison of LEARN-DenseNet with the published methods needs: data
preparation, training, evaluation and their tests. Run names, YAML keys, `nn.Module` attribute names and CSV columns
of the kept methods are unchanged, so their finished runs still load (`tests/regression`).

### Added

- `make smoke`: data preparation, training (first-order and L-BFGS) and evaluation on synthetic data.
- `make data-*`, `train-*`, `eval-*` targets for the joint Mayo + DeepLesion store and for 2DeteCT.

### Changed

- `base_mayo` and `base_dl` train first-order LEARN-DenseNet next to RegFormer; `lbfgs_*` grids train LEARN-DenseNet
  only. In the thesis, the first-order LEARN-DenseNet run on Mayo came from the first-phase Mayo grid (same model,
  recipe and training split).
- `--mem-table` takes any CSV with the columns `model, peak_mem_mb, sec_per_step`.

### Removed

- The other regularization blocks of the thesis benchmark (U-Net, ResNet, ResNeXt/RegNet, EfficientNet, ConvNeXt, ViT
  and their RegNet layouts) and the CG and ADMM update rules; `model.update` is `gd` or `lbfgs`.
- The first-phase experiments E0–E6 (operator checks, angular sampling, learning-rate sweep, matched sizes, 150°
  scans, cross-geometry evaluation, cost measurement) and their commands: `lact pipeline`, `lact train pilot | plan`,
  `lact ops check | density`, `lact eval cross | cost | cg-check`.
- Reporting (`lact report`, `viz`), the demo (`lact demo`), `experiments/`, `scripts/` and the operational documents.

## 0.3.0 (2026-10-17)

A refactor for readability and maintenance. Results do not change: run names, the outputs layout, YAML keys,
`nn.Module` attribute names (checkpoint `state_dict` keys), CSV columns and CLI command names are the same, and
`tests/regression` checks that finished runs reproduce their outputs, the paper tables and their loaded
configurations.

### Added

- `tests/regression`: 9 trained checkpoints (one per architecture in the thesis tables) reconstruct a fixed phantom as
  recorded before the refactor; the paper tables regenerate identically; 246 saved run configurations load to the
  same typed configuration (`make test-regression`).
- `lact.datasets`: a layer between `services` and `infra` for the raw data of each dataset (Mayo DICOM, DeepLesion,
  2DeteCT) and the shared download code (`http_range.py`).
- `experiments/`: thesis analyses, figures and paper numbers, outside the library (`lact` never imports it).
- Top-level API: `from lact import Config, load_config, FanBeamGeometry, SparseFanBeam, FanBeamFBP, model_for`
  (loaded lazily).
- `tests/unit/datasets/test_http_range.py`: Range reads, zip members and resumed downloads against a local server.
- Git history, pre-commit hooks (ruff, mypy, lint-imports) and this changelog.

### Changed

- Every comment, docstring, log/error message, CLI help text, test, script and operational document is in English.
  Report content written for the thesis (table headers, figure labels, generated summaries) stays in Vietnamese, as do
  `docs/decisions.md`, `docs/experiment_log.md` and `docs/analysis_plan.md`.
- Package layout (each package re-exports the names of the module it replaces):
  - `infra/physics` → `infra/operators`; `services/physics` → `services/operators` (`build`, `checks`);
  - `config/schema.py` → `config/schema/`; `core/operators/fbp.py` → `core/operators/fbp/`;
  - compared models → `core/models/baselines/` (DPMA split into network and model); `unrolled.py` → `unrolled/`;
  - `services/training/trainer.py` → `services/training/trainer/`; `services/training/experiments.py` →
    `services/scheduling/`;
  - `services/reporting/tables.py` → `services/reporting/tables/`; `viz/paper/figures.py` → `viz/paper/figures/`;
  - pipeline stage definitions → `services/orchestration/stages.py`;
  - `tests/unit` mirrors the `lact` package tree.
- `tools/` → `scripts/` (operations) and `experiments/` (thesis code); `scripts/lib/remote.sh` holds the SSH settings
  shared by the `mac_*` scripts.
- Lint: pydocstyle (Google convention) on `src/`; flake8-simplify, Ruff-specific and pytest-style rules everywhere.
  mypy runs with `check_untyped_defs`.
- import-linter: ASTRA may only be imported by `infra/operators/astra_matrix.py` (infra included); `lact` never imports
  `experiments`.
- The schedulers log their `==> run` markers instead of printing them.
- `mac_push.sh` no longer uploads `.regression_cache/`; `mac_pull.sh` honours `VAST_USER`.

### Fixed

- `lact.config` now imports `CacheConfig`, which its `__all__` already listed.
- Type errors found by mypy (sparse operator attributes, optional FBP operators of DPMA, `ssim` data range).

### Deprecated

- The `tools` symbolic link to `scripts/` exists for this release only.
