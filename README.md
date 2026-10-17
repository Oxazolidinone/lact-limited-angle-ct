# kltn-lact

Experiment code of the thesis **"Đánh giá và so sánh các mạng tích chập sâu trong bài toán CT góc giới hạn"**
(evaluating and comparing deep convolutional networks for limited-angle CT). Fifteen regularization blocks are placed
in the same LEARN deep-unrolling framework: the original LEARN-CNN and seven architecture families (U-Net, ResNet,
ResNeXt/RegNet, DenseNet, EfficientNet, ConvNeXt, ViT), each with two layouts (uniform and RegNet):

```
x^{k+1} = x^k − α_k A_Ω^T (A_Ω x^k − y) − β_k R_θ(x^k),   x^0 = FBP(y),   k = 0..K−1   (K = 14, shared θ)
```

Flat 2-D fan-beam geometry (DSO 600 mm, DOD 290 mm, 512 detector elements × 1.875 mm, 400 mm field of view), 0.25°
angular step, 90° / 120° scan ranges (main) and 150° (supplementary). Main data: Mayo/AAPM 2016, full dose, 3 mm,
patient-level 7/1/2 split, noise-free sinograms. The same E0–E6 workflow re-runs on **data variants** (Poisson noise,
DeepLesion as a second dataset). Phase 2 adds an L-BFGS data step, a comparison with the methods of the CG-GLORE paper,
and measured data from 2DeteCT (see [docs/PHASE2.md](docs/PHASE2.md)).

## Installation

The environment is a `.venv` inside the project directory (Python ≥ 3.10); every machine (personal computer, school
server, rented machine) works the same way:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"                               # the `lact` command + development tools
.venv/bin/lact --help
make check-all                                                  # ruff + mypy + lint-imports + pytest (uses .venv)
```

ASTRA is only needed to build the projection matrices (`lact ops build`); on machines without ASTRA (e.g. Apple
Silicon Macs) the matrices are built with a pure-NumPy Siddon implementation, which is enough for tests. Training only
needs PyTorch; MPS does not support CSR sparse matrices, so Macs train on the CPU. On rented Vast machines,
`scripts/vast/setup.sh` keeps the machine's PyTorch.

## Repository layout

```
configs/        YAML configuration: data, geometry, model, train, experiment grids, variant, env
src/lact/       the library and the `lact` command (layers: see Architecture below)
experiments/    thesis-specific analyses, figures and paper numbers built on top of lact
scripts/        operations: rented Vast machines (vast/), the school server (server/), pack_source.py
tests/          unit (fast, mirrors src/lact), integration (pipelines on synthetic data), regression (finished runs)
docs/           PIPELINE, PHASE2, VAST, SERVER, ARCHITECTURE; decisions and experiment_log (project log, Vietnamese)
data_raw/, outputs*/, cache/, .runtime/, .venv/     data, results, caches, environment (not part of the source)
```

`tools/` is a temporary symbolic link to `scripts/` so that older commands keep working; it will be removed.

## One entry point: `lact`

```
lact data     download-mayo | download-deeplesion | split-deeplesion | download-2detect | split-2detect |
              inspect | preprocess | simulate | stage | fake
lact ops      build | check | density
lact train    run | grid | pilot | plan | status
lact pipeline run | status
lact eval     freeze | runs | baselines | cross | tune-tv | recheck | cg-check | cost
lact report   select-lr | tables | figures | analysis | phase2
lact demo
```

Every command accepts `-c/--config` (extra YAML files), `-s/--set key=value` (overrides), `--no-defaults` and
`--device`. Details: `lact <group> <command> -h`. `python -m lact ...` works in place of `lact ...`.

A single training run:

```bash
lact train run -m learn_convnext -g fan_a120 -s exp=dev seed=0 train.lr=1e-4
```

Available models (`configs/model/`); every family shares the same 4-level encoder–decoder:

| Family (block type) | Original (uniform layout) | RegNet-integrated (RegNetY-200MF) |
| --- | --- | --- |
| 3-layer CNN of the original LEARN | `learn_cnn` 0.06M | – (no stages) |
| U-Net (plain VGG-style convolutions) | `learn_unet` 2.32M | `learn_unet_regnet` 2.32M |
| ResNet (BasicBlock) | `learn_resnet` 4.09M | `learn_resnet_regnet` 4.23M |
| X block (grouped-convolution bottleneck) | `learn_resnext` 2.95M | `learn_regnet` 3.14M = RegNetY-200MF |
| DenseNet (dense block) | `learn_densenet` 2.70M | `learn_densenet_regnet` 2.88M |
| EfficientNet (MBConv + SE) | `learn_efficientnet` 2.51M | `learn_efficientnet_regnet` 2.49M |
| ConvNeXt | `learn_convnext` 1.98M | `learn_convnext_regnet` 2.06M |
| ViT | `learn_vit` 3.67M | `learn_vit_regnet` 3.51M (3 RegNetY stages + Transformer core) |

The RegNet used in every experiment is **RegNetY-200MF** (X block + SE with ratio 0.25; w = [24, 56, 152, 368],
d = [1, 1, 4, 7], g = 8). "Integrating RegNet" into a family keeps its block type, takes the channels/blocks of the
four levels from the 200MF rule (wm = 2.49, 13 blocks) and adds SE 0.25 to each block; widths are scaled so that the
parameter count stays close to the original (`regnet` and `se_ratio` in the configuration, see
`core/models/regularizers/layout.py`). Parameter-matched group (~2M, E4): `learn_resnet_pm2m`, `learn_regnet_pm2m`,
`learn_vit_pm2m`. Compared methods outside the LEARN framework (phase 2): `regformer`, `cg_glore`, `learn_orig`,
`fbpconvnet`, `lpd`, `dpma` (`core/models/baselines/`). The CG-GLORE re-implementation is not included in the public
repository (its authors have not released their code); without it, `cg_glore` runs stop with a clear message and its
tests are skipped.

Run only some models: `lact pipeline run --models learn_unet learn_unet_regnet ...` (E4 names such as
`learn_regnet_pm2m` must be listed explicitly; keep the same list when resuming).

Grids on a SLURM cluster: `lact train grid configs/experiment/e3_main.yaml --index $SLURM_ARRAY_TASK_ID --run`.
Runs with a `DONE` marker are skipped; interrupted runs resume from `last.pt` (at the exact position in the epoch);
diverged runs get `DIVERGED`, runs that run out of GPU memory even with gradient checkpointing get `OOM`, and both are
reported.

## Everything with one command

After setting the data paths in `configs/local.yaml` (and downloading Mayo if needed), run on the server:

```bash
lact pipeline run --device cuda
```

The command runs **preprocessing → matrices A → E0 → sinograms → E1 / TV tuning → E2 → lr selection → E3/E4/E5 →
test evaluation / E6 → cost → tables, figures, analyses**. Existing data files and matrices are reused.

E2 has 45 runs; E3/E4/E5 have 48 runs in total (30 + 3 + 15) with all 15 models, one seed per configuration (since
05/10/2026; seed 1, 2 runs trained earlier are kept and evaluated). The pipeline chooses the learning rates only once
every E2 run has finished, and each architecture needs at least one DONE run with a finite PSNR. Diverged E2 runs are
recorded; OOM or unfinished runs block the next step. E3/E4/E5 must finish before the test split is opened.

```bash
lact pipeline run --dry-run            # show the order without running or writing anything
lact pipeline run --until prepare      # only preparation and the E0/E1/TV checks
lact pipeline run --until sweep        # up to the end of E2 and the lr selection
lact pipeline run --workers 2          # two training runs at the same time on one device
lact pipeline status                   # state of the whole workflow
lact train status                      # heartbeat and progress of each run
```

By default `--workers 1` trains one run at a time on the chosen GPU. `--workers 2` runs independent runs in parallel;
preparation, evaluation and reporting stay sequential. Increase it only when measurements show free VRAM and a higher
total throughput. The disk cache is shared and locked while files/arrays are created. Never let two servers write the
same outputs directory.

**After an interruption, run the same command again.** Stage state lives in `outputs/pipeline/main.json`, checkpoints
in each run directory. There is no time limit by default; with `--max-hours`, or on SIGINT/SIGTERM, the pipeline stops
taking new jobs, asks running jobs to save a checkpoint, waits at most 10 minutes and exits with code 3. The experiment
configuration must stay the same when resuming; GPUs, cache, workers and the time limit may change.

**Extend the grid of an existing outputs directory** (e.g. E0–E6 results of the first four architectures, now adding
LEARN–RegNet):

```bash
lact pipeline run --device cuda --extend
```

`--extend` accepts the existing outputs although the configuration list changed. DONE runs are kept (and still checked
against their saved `config.yaml`: changing a hyper-parameter of an old run is an error); only new runs are trained,
then cost, tables and figures are regenerated. Models added after the test split was opened are reported separately in
`B10_supplementary` (new backbones versus LEARN–CNN; RegNet-integrated versions versus the original of the same family).

## Data variants (noise, second dataset)

A variant is one file in `configs/variant/` that changes the **measured data** and points every output to its own
directory; the E0–E6 workflow, the architecture grid, the hyper-parameters and the evaluation stay the same. The
projection matrices are shared (same geometry).

| Variant | Data | Outputs | Command |
| --- | --- | --- | --- |
| (default) | Mayo, noise-free | `outputs/` | `lact pipeline run` |
| `mayo_poisson_1e6` | Mayo + Poisson noise, I0 = 10⁶ photons/ray | `outputs_poisson_1e6/` | `lact pipeline run -c configs/variant/mayo_poisson_1e6.yaml` |
| `mayo_poisson_1e5` | Mayo + Poisson noise, I0 = 10⁵ (strong noise) | `outputs_poisson_1e5/` | `lact pipeline run -c configs/variant/mayo_poisson_1e5.yaml` |
| `deeplesion` | DeepLesion (NIH), noise-free | `outputs_deeplesion/` | see below |
| `joint` | Mayo + DeepLesion in one store (phase 2) | `outputs_joint/` | `bash scripts/vast/phase2.sh` |
| `detect` | 2DeteCT, measured sinograms (phase 2) | `outputs_detect/` | `bash scripts/vast/detect.sh` |

Noise variants reuse `outputs/data/images.h5` (the reference images do not change), re-simulate noisy sinograms and
their FBP, and redo the TV tuning and the lr sweep on that data. Poisson noise follows Beer–Lambert:
`N ~ Poisson(I0·exp(−p))`, measurement `−ln(N/I0)` (`core/imaging/noise.py`); each slice has a fixed noise realisation
given by `noise.seed`. E0/E1 check the operator and the angular density on noise-free projections.

DeepLesion (zip files of ~4 GiB from NIH's public Box folder, with resumed downloads and MD5 checks):

```bash
lact data download-deeplesion -c configs/data/deeplesion.yaml --zips 1 2    # first 157 patients, 8 GiB
lact pipeline run -c configs/variant/deeplesion.yaml --device cuda
```

The split file `configs/data/split_deeplesion.yaml` is part of the source (generated from the real data of the first
two zip files on 05/10/2026: 1,692 / 233 / 584 train/val/test slices). Re-run `lact data split-deeplesion` only to
change the split on purpose. It picks 45/6/15 patients (train/val/test) by increasing id, following the official
`Train_Val_Test` column, keeping patients with ≥ 20 slices of 512 × 512; each patient contributes at most 40 evenly
spaced slices (~1,800 training slices, the same scale as Mayo). Tables/figures of each variant live in its outputs;
`python experiments/figures/export_results.py --outputs <outputs>` collects the plotting data of one set.

On rented Vast.ai machines (pushing code and trained runs, previewing the cost, chaining two datasets, pulling results
back): [docs/VAST.md](docs/VAST.md). Failure handling and the orchestration files: [docs/PIPELINE.md](docs/PIPELINE.md).

## Phase 2: two datasets, L-BFGS data step, comparison, measured data

After E0–E6: ResNet, U-Net, DenseNet (LEARN framework) and RegFormer are trained on Mayo and on DeepLesion (one joint
store with a common intensity scale, tested per dataset; the three LEARN backbones reuse their Mayo E3 runs). Each
model is trained a second time, also from scratch, with the L-BFGS data step (`model.update: lbfgs`) and the same
recipe. The comparison group trains the methods of the CG-GLORE paper's table (CG-GLORE, original LEARN, FBPConvNet,
Learned Primal-Dual, DPMA) on the same data, and the main models are trained again on the measured 2DeteCT data.
Settings follow the papers, without an lr sweep. Design, recipes, commands and cost estimates:
[docs/PHASE2.md](docs/PHASE2.md).

```bash
lact train grid configs/experiment/base_dl.yaml -c configs/variant/joint.yaml --dry-run      # first-order runs
lact train grid configs/experiment/lbfgs_dl.yaml -c configs/variant/joint.yaml --dry-run     # L-BFGS runs
bash scripts/vast/phase2.sh start --plan                                                    # on the rented machine
```

## Configuration

- Defaults: `configs/paths.yaml`, `configs/data/mayo_3mm.yaml`, `configs/data/split_7_1_2.yaml`,
  `configs/train/default.yaml`, `configs/baseline/*.yaml`.
- Per-machine settings: create `configs/local.yaml` (kept out of git), for example:

  ```yaml
  data:
    dicom_root: /path/to/Mayo_LDCT_2016
  cache:
    enabled: true
    dir: outputs/cache   # a local SSD when the raw data sits on a network drive
    arrays: true
    verify: sha256
  train:
    checkpoint_every_min: 10
    keep_prev_checkpoint: true
    require_gpu: true
  device: cuda
  ```
- YAML is read **once** at the boundary and converted into typed dataclasses (`lact.config.Config`); a misspelt key
  or a value of the wrong type is reported when loading, not in the middle of training.
- Models: `configs/model/*.yaml`; geometries: `configs/geometry/*.yaml`; experiment grids:
  `configs/experiment/*.yaml`.

## Workflow (as in the thesis plan)

The `make` targets are shortcuts that call `lact`; when the project has a `.venv`, `make` uses it.

| Step | Command | Result |
| --- | --- | --- |
| Inspect DICOM | `make inspect` | `outputs/data/dicom_inspect.csv` |
| Preprocessing | `make data` | `images.h5`, `stats.json`, `slices.json`, `manifest.json` |
| Projection matrices | `make operators` | `outputs/operators/A_{geom}_n{256,512}.npz/.json` |
| E0 + E1 | `make check` | `outputs/checks/e0_report.json`, `e1_angle_density.csv` |
| Simulation | `make simulate DEV=cuda` | `outputs/data/sino_fan_a{90,120,150}.h5` |
| TV tuning | `make tune-tv` | `outputs/baselines/tv_best_*.json` |
| Gate 1 | `make pilot DEV=cuda` | `outputs/tables/pilot_fan_a120.csv` (seconds/step, GPU-hours) |
| E2 lr sweep | `make e2 select-lr` | `outputs/runs/e2_sweep/best_lr.json` |
| Freeze | `make freeze` | `outputs/configs_frozen.lock` (opens the test split) |
| E3, E4, E5 | `make e3 e4 e5` | `outputs/runs/<exp>/<run_id>/` |
| Evaluation | `make eval cross` | `outputs/eval/{per_slice,preds}/`, `outputs/eval/index.json` |
| Cost | `make cost DEV=cuda` | `outputs/tables/cost_*.csv` |
| Tables, figures | `make tables figures analysis` | `outputs/tables/B1..B10.*`, `outputs/figures/F1..F10.*` |
| Demo | `make demo` | Gradio at http://localhost:7860 |

## Running on a server over SSH

The school server runs the project directly with the `.venv` inside it: see
[environment, tmux, logs and resuming checkpoints](docs/SERVER.md). Two GPUs: `lact pipeline run --gpus auto --n-gpus 2`
(picks the two idlest GPUs, one training run per GPU; see [PIPELINE.md](docs/PIPELINE.md)).

Package the source with `make source-zip`, copy `dist/kltn-lact.zip` to the server and unpack it into the project
directory. Install the environment as above, create `configs/local.yaml`, then run inside `tmux`:

```bash
tmux new -s lact
# inside the tmux session, in kltn-lact with the environment installed:
lact pipeline run --device cuda
```

Press **Ctrl+B**, release, then **D** to detach. SSH can be closed and the personal machine turned off; the job keeps
running on the server. Reattach with `tmux attach -t lact`. After a server restart, open a new tmux session and run the
pipeline command again to resume from the checkpoints.

**Protection against lost work when a job is interrupted or moves to another GPU:**

| Mechanism | Details |
|---|---|
| Mid-epoch checkpoint | `last.pt` every 10 minutes (`train.checkpoint_every_min`) with the batch position in the epoch; resuming gives exactly the same result as an uninterrupted run |
| Fallback checkpoint | the previous file is kept as `last_prev.pt`; if `last.pt` is corrupted (machine stopped while writing) the fallback is used |
| Graceful stop | SIGTERM/SIGINT (Ctrl+C) → save a checkpoint and exit with code 3; the pipeline stops its queue |
| GPU out of memory (moved to a smaller GPU) | gradient checkpointing is switched on and the step retried; if memory is still short, `OOM` is written |
| Monitoring | `heartbeat.json` (step, loss, ETA), `sessions.jsonl` (GPU of each session), `events.jsonl`; see `lact train status` |
| Cache | `data stage` copies sinograms/images/matrices into `cache.dir` and checks SHA-256; loaded arrays are saved as `.npy` and memory-mapped next time |
| Large files with the cache on | built in the local cache, then copied to the outputs through `.partial` + rename |

## Trial run without real data

```bash
make smoke        # 128 x 128 synthetic DICOM -> the whole pipeline at a small size (outputs_smoke/)
```

## Architecture

Layers; dependencies only point downwards (checked by `lint-imports`):

```
cli  ──►  services  ──►  datasets  ──►  infra | viz  ──►  config  ──►  core
```

```
src/lact/
  core/       pure domain: geometry, noise, operators, FBP, models (LEARN + baselines), solvers, metrics, analyses
  config/     YAML -> typed dataclasses (Config, ExperimentSpec)
  infra/      adapters: projection matrices (ASTRA/Siddon), HDF5 stores, Dataset, RunStore, ResultsStore, runtime
  datasets/   raw data of each dataset: Mayo DICOM, DeepLesion PNG, 2DeteCT, Box/HTTP Range downloads, splits
  viz/        figures and tables from numpy/pandas
  services/   use cases: data, operators, training, scheduling, evaluation, reporting, orchestration, demo;
              factories.py is the composition root
  cli/        the `lact` command and the demo interface
experiments/  thesis analyses, figures and paper numbers (lact never imports it)
```

Details, rules for adding code and the map from the earlier layout: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Reproducibility notes

- Each run stores `config.yaml` (merged, with a `derived` section), `env.json`, `git_hash.txt`, `metrics.csv`,
  `steps.csv` and `summary.json`.
- Data and matrices are checked against the SHA-256 in `outputs/data/manifest.json` before training.
- Evaluation results are looked up through `outputs/eval/index.json` (metadata), never by parsing file names.
- Every number in the reports comes from `outputs/tables/*.csv`; nothing is copied by hand.
- `tests/regression` pins the outputs of finished thesis runs (9 checkpoints, the paper tables, 246 run
  configurations); see `tests/regression/conftest.py`.
- The Mayo/DeepLesion/2DeteCT data are never committed; follow their terms of use and cite them (McCollough et al.
  2017 for Mayo; Yan et al. 2018 for DeepLesion; Kiss et al. 2023 for 2DeteCT).
