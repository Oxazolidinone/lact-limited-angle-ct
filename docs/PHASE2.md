# Phase 2: two datasets and the L-BFGS data step

Follows E0–E6 (the 15-backbone benchmark on Mayo). This phase answers two questions:

1. How do four representative models (ResNet, U-Net, DenseNet in the LEARN framework, and RegFormer) do on a second
   dataset (DeepLesion), and how does a model trained on one dataset do on the other?
2. Does replacing each model's data step with an L-BFGS step (a second-order approximation that needs no linear solve)
   and training FROM SCRATCH with the same recipe help?

> **Correction of 11/10/2026.** The 10/10 version had the L-BFGS runs load the checkpoints of the first-order runs and
> train 50 more epochs (`train.init_from`). The user wanted L-BFGS **trained from scratch**; the 16 old-style runs were
> removed from the results (on the rented machine: `luu-tru/2026-10-11_lbfgs_tu_checkpoint/`) and every L-BFGS run now
> starts from a random initialisation, with exactly the lr and epochs of the first-order run of the same model. The
> passages below about "attaching", "source runs" and "controls" only document the old version.
>
> **Stabilising the L-BFGS step (11/10/2026).** Training from scratch with the plain L-BFGS step failed: DenseNet on
> Mayo diverged at step ~2,300 (lr 3e-4), RegFormer on Mayo learned very slowly (21–22 dB after 12 epochs) and then
> diverged at steps ~21,000–27,000. Cause: there is no line search, and pairs (s, y) with a curvature κ = sᵀy / sᵀs
> close to 0 make H scale the direction s by about 1/κ. Every L-BFGS run now uses Powell-style damping relative to
> B₀ = I, `model.lbfgs_damping: 0.5` (H scales no direction by more than 2; `core/solvers/lbfgs.py`). The level 0.5 was
> chosen with short trial runs on the Mayo validation split, PSNR per epoch (`outputs_joint/lbfgs_stab/`):
>
> | Measure | DenseNet 120° (epochs 0–2) | RegFormer 90° (epochs 0–4) |
> | --- | --- | --- |
> | none (old version) | 19.88 then diverged | about 21 dB until epoch 12, then diverged |
> | drop pairs with κ < 0.01 | 26.76 / 29.09 / 30.35 | 16.14 / 17.53 / 16.22 / 17.62 / 17.27 |
> | damping 0.05 | 26.68 / 29.32 / 28.70 | 17.45 / 18.20 / 18.43 / 18.78 / 20.52 |
> | damping 0.2 | 27.27 / 29.55 / 29.41 | 18.79 / 21.02 / 19.98 / 18.24 / 18.26 |
> | **damping 0.5** | 27.01 / 29.59 / 28.59 | 18.87 / 22.12 / 23.19 / 23.50 / 23.65 |
>
> (First-order RegFormer at 90°: 20.41 dB after epoch 0, 23.27 dB after epoch 4.) The trial runs were only 4–6 epochs
> long, so they do not guarantee that every run survives to epoch 50. `scripts/vast/phase2_guard.sh` covers a run
> diverging while nobody watches: it re-runs `phase2.sh` once (skipping the broken run) so that the evaluation, tables
> and figures still happen and the machine still Stops itself.

The 08/10 plan (CG, ADMM, joint training on both datasets, an lr sweep for RegFormer) was dropped by the decision of
10/10; its files are in `KLTN/luu-tru/2026-10-10_ke-hoach-cu/`. The code of `model.update: cg | admm` remains, but no
grid uses it.

## 1. Experiment design

Four models × two geometries (90°, 120°) × two datasets, one run per configuration (seed 0). The two datasets are
trained **separately** (not pooled). Every model is tested on the **same test set**: 2 Mayo patients (L310, L506;
425 slices) and 300 DeepLesion slices (207 patients), reported per dataset.

| Step | Experiment | Trained on | Models | Data step | Runs | lr | Initialisation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `e3_main` (existing) | Mayo | ResNet, U-Net, DenseNet | Aᵀ (LEARN) | 6, not trained | 3e-4 | – |
| 1 | `base_mayo` | Mayo | RegFormer | FBP of the residual | 2 | 1e-4 | random |
| 1 | `base_dl` | DeepLesion | all four | as above | 8 | 3e-4 / 1e-4 | random |
| 2 | `lbfgs_mayo` | Mayo | all four | L-BFGS | 8 | 3e-4 / 1e-4 | random |
| 2 | `lbfgs_dl` | DeepLesion | all four | L-BFGS | 8 | 3e-4 / 1e-4 | random |

**26 training runs** in total (10 in step 1, 16 in step 2). Each run is 50 epochs: 85,450 steps on Mayo (1,709
slices), 100,000 steps on DeepLesion (2,000 slices).

- The three LEARN backbones are not retrained on Mayo: their E3 runs are copied to the rented machine, their validation
  PSNR is checked again on the joint store, and they serve as the baseline and as the source checkpoint. A backbone with
  several E3 runs uses the run with the highest validation PSNR (ResNet 120°: s1, ResNet 90°: s2; the others s0).
- Step 2 is compared with step 1 in pairs: same model, geometry, data, lr and epochs, both trained from scratch, matched
  per slice (table C2). The only difference is the data step, so no control run is needed.
- Seeds on Mayo: the L-BFGS runs use seed 0, while the ResNet E3 runs on the rented machine are those with the highest
  validation PSNR of the three seeds (s1, s2). To pair identical seeds, swap in the seed-0 ResNet E3 runs before
  evaluating.

## 2. Training settings: from the papers, no lr sweep

| Source | Optimiser | lr | lr schedule | Epochs | Batch | Loss | Iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LEARN (Chen et al., 2018) | – | 1e-4 (chosen among 1e-5, 1e-4, 1e-3) | – | – | – | MSE | – |
| RegFormer, public code | AdamW | 1e-4 | cosine (T_max 200) | 200 | 1 | MSE | – |
| QN-Mixer (CVPR 2024) | AdamW, weight decay 1e-2 | 1e-4 | ×0.1 after epoch 40 | 50 | 1 | MSE | 14; RegFormer 18, LEARN 30 |
| CG-GLORE (2026, the supervisor's group) | – | 1e-4 for every compared model | – | 50 | – | – | 15 (10 CG iterations); RegFormer 10 (2.7 M), LEARN 30 |
| The group's LA-CT benchmark paper | Adam | 1e-4 | cosine down to 1e-5 | 50 | 1 | MSE | K = 14 |

Sources: LEARN [arXiv:1707.09636](https://arxiv.org/abs/1707.09636); RegFormer code
[github.com/Deep-Imaging-Group/RegFormer](https://github.com/Deep-Imaging-Group/RegFormer); QN-Mixer
[arXiv:2402.17951](https://arxiv.org/abs/2402.17951); CG-GLORE arXiv:2608.15246; benchmark paper: PDF in
`KLTN/tai-lieu/`. A "–" means the paper does not state it or it was not checked.

Used in the project (`configs/train/default.yaml`, unchanged from E3): Adam, cosine down to 1e-6 after a 1-epoch
warm-up, **50 epochs, batch 1, MSE**, gradient clipping at 1.0; K = 14 for the LEARN framework, K = 10 for RegFormer
(2.68 M parameters, the configuration CG-GLORE compares against). The arithmetic mode is the same as E3
(`precision.tf32: true`, deterministic): phase 2 runs on the same GPU type as the Mayo batch (RTX 5090), so new runs and
the E3 baseline are trained and evaluated the same way; every training, evaluation and speed-measurement command sets
this mode from the configuration.

| Runs | lr | Reason |
| --- | --- | --- |
| First-order ResNet, U-Net, DenseNet on DeepLesion | 3e-4 | the lr of their Mayo (E3) runs, so the two datasets compare; no new sweep runs |
| First-order RegFormer (Mayo, DeepLesion) | 1e-4 | the original code, QN-Mixer and CG-GLORE all train RegFormer at 1e-4 |
| Every L-BFGS run | that of the first-order run of the same model (LEARN 3e-4, RegFormer 1e-4) | user decision of 11/10: trained from scratch, only the data step differs, so it compares directly with the first-order run |

## 3. Data: the joint store

`configs/variant/joint.yaml` (pulling in `configs/data/joint.yaml`): Mayo (10 patients, DICOM) and DeepLesion (2,500
key slices, PNG) share one `outputs_joint/data/images.h5` and the same sinogram files.

DeepLesion is used as in QN-Mixer ("a subset of 2000 training images and 300 testing images randomly from the official
splits") and CG-GLORE (2,000 / 200 / 300 images from the official split): **2,000 / 200 / 300 key slices** are drawn at
random (seed 0) from the three splits of the `Train_Val_Test` column of `DL_info.csv`, keeping only 512 × 512 slices of
series not flagged as noisy. The official split is patient-level, so no patient is in two splits: 1,118 training, 154
validation and 207 test patients. The slice list is `configs/data/deeplesion_keyslices.csv` (the single source of the
draw; `lact data split-deeplesion --key-slices` generates it together with `joint_patients.yaml` and
`split_joint*.yaml`). The slices are spread over all 56 `Images_png_XX.zip` files (220 GB): `lact data
download-deeplesion --key-slices` downloads each file, extracts the few dozen slices needed and deletes the file, 6 files
in parallel. NIH's `Key_slices.zip` cannot be used: its images are 8-bit RGB renderings, not HU. Open point: neither
paper says whether they drew key slices or any neighbouring slices; key slices are used here because the official
split column belongs to them. `data.format: mixed`: patient ids `DLxxxxxx` are read from `data.deeplesion_root`, the
others as DICOM from `data.dicom_root`.

Intensities are normalised with **the Mayo μ_ref** (`data.norm.mode: fixed`, 0.0379008 mm⁻¹ ≈ 974 HU), not recomputed:

- the Mayo images in the joint store equal the old Mayo store value for value, so the E3 runs can be reused and must
  reproduce their recorded validation PSNR (`lact eval recheck --on-config-data`, tolerance 0.02 dB);
- the PSNR of both datasets is on the same scale. For DeepLesion, 0.2% of the pixels exceed 974 HU and are clipped to 1
  (Mayo: 0.1%).

| Split file | Train | Validation | Test | Used for |
| --- | --- | --- | --- | --- |
| `split_joint_mayo.yaml` | Mayo 7 (1,709 slices) | L333 (244 slices) | 425 Mayo slices + 300 DeepLesion slices | `base_mayo`, `lbfgs_mayo`, re-evaluating `e3_main` |
| `split_joint_dl.yaml` | DeepLesion 2,000 slices | DeepLesion 200 slices | as above | `base_dl`, `lbfgs_dl` |
| `split_joint.yaml` | both | both | as above | building the store, TV tuning, FBP/TV baselines |

## 4. Models

A and y are normalised (‖A‖ = 1), x⁰ = FBP(y).

**LEARN framework** (`core/models/unrolled/learn.py`, K = 14, regularizer R_θ shared by the stages):
x⁺ = x − α_k Aᵀ(Ax − y) − β_k R_θ(x).

**RegFormer** (`core/models/baselines/regformer.py`; Xia et al., TRPMS 2024), rebuilt from the public code, outside the
LEARN framework: x⁺ = ReLU(x + w_k · FBP(y − Ax) + N_k(x, z)). The layers N_k alternate convolution blocks (3 layers of
5×5, 96 channels) and Swin blocks (one token per pixel, window 8, 2 layers, 3 attention heads), pass 48 feature channels
z to the next layer, and each layer has its own weights; K = 10. One difference from the original: the image output
convolution of each layer starts at zero (`zero_out`), because the original initialisation gives randomly biased output
images across seeds (details in the module docstring).

### The L-BFGS data step (`model.update: lbfgs`)

g_k = Aᵀ(Ax_k − y) is the gradient of the data term. The earlier stages of the same forward pass provide the pairs

    s_i = x_{i+1} − x_i,        y_i = g_{i+1} − g_i = AᵀA s_i

(exact, because the data term is quadratic). The two-loop L-BFGS recursion (`core/solvers/lbfgs.py`) builds
H_k ≈ (AᵀA)⁻¹ on the directions taken and applies it to g_k, without solving any linear system:

- LEARN framework: x⁺ = x − α_k H_k g_k − β_k R_θ(x);
- RegFormer: x⁺ = ReLU(x − α_k H_k g_k + N_k(x, z)); the FBP weight w_k is no longer used.

The first stage has no pairs yet, so H = I (a plain gradient step). Each stage still costs one A, Aᵀ pair like LEARN
(CG needs 1 + number of CG iterations pairs), plus a few inner products.

Four design choices, settled after attaching the step to real E3 runs at full size (256 × 256). Numbers: PSNR of
LEARN–ResNet 90° on 3 slices right after attaching, before any retraining (original model: 26.51 dB; only resetting
α = 1 without L-BFGS: 26.42 dB):

| Choice | Reason |
| --- | --- |
| H₀ = I, not γI (γ = sᵀy / yᵀy) | there is no line search; with ‖A‖ = 1, γ ≥ 1 (measured 1.0–4.4) amplifies every direction not yet remembered: 24.45 dB versus 22.69 dB (at 120° both are equal: 31.20 and 31.18 dB). QN-Mixer also uses H₀ = I |
| α_k is a new parameter (`a_qn`), initialised to 1 | the trained LEARN α grows to 4.6, which suits a gradient step but not an H-scaled step: reusing it breaks the output (3.6 dB, below FBP at 12.9 dB). α = 0.7 and 0.5 give 21.6 and 21.8 dB, worse than α = 1 |
| remember every earlier stage (`lbfgs_memory: 13`) | K is only 14 and the extra memory is negligible; 5 pairs give 24.13 dB versus 24.45 dB |
| the (s, y) pairs do not propagate gradients | as in QN-Mixer; gradients still flow through g_k, α_k, β_k and the regularizer |

The current plan trains the L-BFGS runs from scratch. Notes on the old version (loading the first-order checkpoint
through `train.init_from`, keeping β_k, resetting α_k): right after attaching, before retraining, the PSNR **drops**
because the regularizer was tuned to the old step (3 slices, real E3 runs): ResNet 90° 26.51 → 24.45 dB; U-Net 90°
27.11 → 25.19; DenseNet 90° 27.79 → 26.02; ResNet 120° 34.46 → 31.20.

How to read these trial numbers: they only compare the variants with each other and are not results to report. The
trials ran on the personal machine with 3 slices and sinograms projected directly from the 256 images with the Siddon
matrix (rented machine: ASTRA, simulated from the 512 images), so the absolute numbers differ from the result tables.
Those three slices (L310/136, L310/23, L506/191; at 120°: L310/136, L310/56, L506/148) belong to Mayo **test** patients,
because the personal machine only had the images used for figures; repeat this comparison on validation slices (L333)
on the rented machine before training step 2 and record the result here.

## 5. Running on the rented machine

On the personal machine (source code + 6 E3 runs, about 70 MB):

```bash
bash scripts/vast/mac_push_phase2.sh IP PORT ../ket-qua/vast_results_2026-10-07/outputs
```

On the Vast machine:

```bash
cd /workspace/kltn-lact && TORCH_PIN=2.11.0 bash scripts/vast/setup.sh   # PyTorch 2.11.0+cu128 as for the E3 runs
bash scripts/vast/phase2.sh start --plan       # download, build the joint store, check E3 runs, measure speed, estimate
VARIANT=joint bash scripts/vast/status.sh      # estimate / progress
bash scripts/vast/phase2.sh start --auto-stop  # train both steps -> evaluation -> tables; Stop the machine when done
```

Both steps share one queue and no run depends on another (every run trains from scratch). The queue still understands
`train.init_from`: a grid using it makes its runs wait for their source run to be DONE. With the measurements of
`--plan`, the longest runs (RegFormer) are queued first, each long run gets a GPU of its own, and runs needing more
than half the GPU memory run alone. If a source run fails, its dependent runs do not run and the script reports an
error.

`phase2.sh` options: `--until base` (step 1 only, table C1, then stop), `--control` (no longer needed since 11/10),
`--per-gpu N` (default 2). Environment variables: `MODELS="learn_resnet regformer"`, `GEOMS="fan_a90"` to run a part.
Graceful stop: `VARIANT=joint bash scripts/vast/stop.sh`; run `start` again to resume (finished runs are skipped). Pull
the results: `bash scripts/vast/mac_pull.sh IP PORT` (includes `outputs_joint`).

The steps the script runs (can be run by hand; `J` = `configs/env/server.yaml configs/variant/joint.yaml`,
`E` = `configs/experiment`):

```bash
lact data preprocess -c J configs/data/split_joint.yaml
lact ops build -c J -g fan_a90 fan_a120 ; lact data simulate -c J configs/data/split_joint.yaml -g fan_a90 fan_a120
lact eval recheck -c J configs/data/split_joint_mayo.yaml --exp e3_main --on-config-data --per-geometry 12
lact train grid E/base_mayo.yaml E/base_dl.yaml E/lbfgs_mayo.yaml E/lbfgs_dl.yaml -c J \
     --run --gpus all --per-gpu 2 --mem-table .runtime/pilot2/pilot_*.csv                  # both steps
lact train grid E/lbfgs_mayo.yaml E/lbfgs_dl.yaml -c J --exp-rename lbfgs_=ctrl_ -s model.update=gd \
     --run --gpus all --per-gpu 2                                                          # controls (optional)
lact eval runs -c J configs/data/split_joint.yaml --exp base_mayo base_dl lbfgs_mayo lbfgs_dl --split test
lact eval runs -c J configs/data/split_joint_mayo.yaml --exp e3_main --on-config-data --split test
lact eval baselines -c J configs/data/split_joint.yaml --split test -g fan_a90 fan_a120
lact report phase2 -c J configs/data/split_joint.yaml
```

Trying it on the personal machine (no GPU): `tests/integration/test_phase2.py` runs the whole chain on synthetic data.

## 6. Results

In `outputs_joint/tables/`:

- `C1_datasets`: PSNR/SSIM on each test dataset, by training data and data step (with FBP, TV). The "trained on Mayo"
  rows also have DeepLesion columns (and vice versa): those are the cross-dataset results.
- `C2_second_order`: the L-BFGS run minus its baseline, matched per slice: ΔPSNR, ΔSSIM and the number of patients that
  improve on each dataset. Column "Loại mốc": `train thường` (the run of the same model on the same data, trained from
  scratch with the old data step).
- `C3_runs`: each training run: validation PSNR, best epoch, training hours, memory, source run, GPU type used.

The `metrics.csv` of an L-BFGS run records the per-stage `alphas`, `betas` after every epoch.

Figures and comparison images: at the end of training, `phase2.sh` calls `experiments/figures/export_phase2.py`, which
collects into `outputs_joint/figdata/` the reconstructions of the illustrative slices (2 slices per Mayo patient,
1 per DeepLesion patient, chosen by FBP PSNR so they do not depend on the models), per-slice metrics, per-stage
parameters and training curves, draws PNG versions into `figdata/figures/` and writes `figdata/BAO_CAO_KET_QUA.md`
(an automatic summary in Vietnamese: test data, tables C1–C3, every figure; numbers only, commentary is written after
reviewing them). Redraw the thesis versions on the personal machine:

```bash
python experiments/figures/make_figures_phase2.py --data <outputs_joint>/figdata --out <figure directory> --lang vi --preset thesis
```

| Figure | Content |
| --- | --- |
| `fig_qual_<mayo\|dl>` | reconstructions, zoomed ROI and error maps: FBP, TV, every model first-order and with L-BFGS (trained on that dataset), both scan ranges, one slice |
| `fig_qual_cross_<mayo\|dl>` | as above, with models trained on the other dataset (cross-dataset) |
| `fig_psnr_*`, `fig_ssim_*` | per-slice metrics of each method |
| `fig_lbfgs_delta_*` | per-slice ΔPSNR of the L-BFGS run against its source run (`fig_lbfgs_vs_ctrl_*`: against the control, if any) |
| `fig_curves_*` | validation PSNR per epoch: first-order run, then the L-BFGS run |
| `fig_stage_*` | α_k, β_k per stage |

## 7. Cost

Measured in the Mayo batch (RTX 5090, 2–3 runs per GPU, 50 epochs): U-Net 4 hours per run, ResNet 7.3 hours, DenseNet
7 hours. One L-BFGS stage costs about as much as a LEARN stage, so a step-2 run takes about as long as a step-1 run.
The LEARN part of the plan (18 runs) is therefore about 55 GPU-hours. RegFormer (8 runs) had never run on the project's
GPUs: `phase2.sh start --plan` measures the seconds/step and memory of all four models in both steps and prints the
GPU-hours **before** training. Rough estimate for everything: 100–150 GPU-hours, i.e. 1–1.5 days on a 4-GPU machine.
`--control` adds about as much as step 2.

To reduce it: `MODELS=...` to run some models first, `--until base` to look at step 1 before deciding on step 2.

## 8. Limitations to state in the write-up

- The L-BFGS and first-order runs are both trained from scratch with the same recipe, so their difference is the effect
  of the data step (plus the run-to-run variation, see below). The lr of the three LEARN backbones (3e-4) was chosen
  for the plain gradient step in the Mayo batch and not re-chosen for L-BFGS.
- One run per configuration: differences smaller than the run-to-run variation (Mayo batch: median about 0.3 dB) are
  not conclusive; table C2 therefore includes the number of patients that improve.
- The lr values come from the papers and were not re-chosen for DeepLesion or for L-BFGS.
- RegFormer's original data step (FBP of the residual) is already a filtered step; switching to L-BFGS replaces a fixed
  filter with an H built from at most 9 pairs, which is not necessarily better. L-BFGS here also differs from QN-Mixer
  (that paper runs BFGS in a latent space, on encoded gradients, with FBP as A†).
- DeepLesion is treated as 400 mm images (the same assumption as Mayo) and uses the Mayo intensity scale.

## 9. Troubleshooting

- **`missing the Mayo E3 runs`**: `mac_push_phase2.sh` has not been run with the outputs directory of the Mayo batch.
- **`old Mayo runs do NOT match the joint store`**: the Mayo images or the matrices on this machine differ from the E3
  training. Check the ASTRA version (`setup.sh` pins 2.5.0) and `data.norm.mu_ref` in `configs/data/joint.yaml` (it
  must equal `mu_ref` in `outputs/data/stats.json` of the Mayo batch).
- **`train.init_from=...`: no DONE run found**: the step-1 source run has not finished or diverged; check
  `VARIANT=joint bash scripts/vast/status.sh`, then run `start` again.
- **A diverged run**: delete that run's directory, run it again with a smaller lr (`-s train.lr=5e-5`) and record the
  change.
- **GPU out of memory** (most likely for RegFormer with 2 runs per GPU): the trainer switches to per-stage gradient
  checkpointing (same results, slower); or run with `--per-gpu 1`.
- **Incomplete `images.h5`** (machine stopped during preprocessing): delete `outputs_joint/data` and run again.

## 10. Comparison with the CG-GLORE paper

The supervisor asked (12/10) for a comparison with the methods in the table of the CG-GLORE paper (BMVC 2026,
arXiv:2608.15246) in the **limited-angle setting of the thesis**, with only our best result on our side (LEARN-DenseNet
+ L-BFGS). The CG-GLORE paper studies noisy sparse-view CT, so its numbers cannot be reused: every method is trained
FROM SCRATCH on exactly the data, the 90°/120° projection operators and the splits of the existing runs (noise-free,
normalised by `mu_ref`, checkpoint chosen by validation PSNR); the training recipe is the shared recipe of the CG-GLORE
paper (see below).

| Method | Configuration | Layers | Parameters (paper) | Rebuilt from | Runs |
|---|---|---|---|---|---|
| FBPConvNet | `fbpconvnet.yaml` | – | 31.08M (31.1M) | the original paper | `cmp_*` |
| Learned Primal-Dual | `lpd.yaml` | 10 | 0.253M (0.251M) | original code (TensorFlow) | `cmp_*` |
| LEARN (original) | `learn_orig.yaml` | 30 | 1.80M (1.8M) | the project's LEARN framework, each layer with its own three 5x5 convolutions, 48 channels | `cmp_*` |
| RegFormer | `regformer.yaml` | 10 | 2.68M (2.7M) | original code | existing: `base_*` |
| CG-GLORE | `cg_glore.yaml` | 15 | 5.52M (5.5M) | the authors' code (`tai-lieu/CG-GLORE_code_goc`) | `cmp_*` |
| LEARN-DenseNet + L-BFGS | `learn_densenet.yaml` | 14 | 2.70M | this thesis | existing: `lbfgs_*` |
| DPMA | `dpma.yaml` | 300 refinement iterations | 8.06M per network (8.1M) | original code (github.com/jia-W-w/DPKA), adapted to limited angle | `dpma1_*` then `dpma_*` |
| QN-Mixer | – | 14 | (8.5M) | no source code; the original paper has one 90° number in another setting | awaiting a decision |

Grids `configs/experiment/cmp_mayo.yaml`, `cmp_dl.yaml`: 4 methods x {90°, 120°} x 2 datasets = 16 runs. DPMA has its
own grids since it trains in two stages: `dpma1_mayo`, `dpma1_dl` (network M1), then `dpma_mayo`, `dpma_dl` (M1 loaded
frozen through `train.init_from`, network M2 trained): 8 runs, 50 epochs each.

**Training recipe: SHARED, as in the CG-GLORE paper** (user decision of 13/10; `configs/train/bmvc.yaml`). That paper
trains every compared model for 50 epochs at lr 1e-4; the rest comes from the authors' code: AdamW, weight decay 1e-2,
cosine down to 1e-5, batch 1, MSE loss, no lr warm-up, no gradient clipping. It applies to all 24 `cmp_*` and `dpma*`
runs (DPMA: 50 epochs per stage, with its own loss). The original code stops early after 10 epochs without
improvement; here all 50 epochs run and the epoch with the best validation PSNR is kept. The thesis runs (`base_*`,
`lbfgs_*`) were trained with `configs/train/default.yaml` and are not retrained.

What belongs to each model still follows its authors: Learned Primal-Dual initialises the primal and dual variables to
0 with Xavier weights; LEARN initialises its kernels from N(0, 0.01²) with 30 layers as in the CG-GLORE table; DPMA keeps
the loss and the 300 refinement iterations of the original code. (The trainer still supports each author's own recipe,
e.g. FBPConvNet's SGD, but no grid uses it.)

**Image scale and PSNR/SSIM: those of the thesis** (user decision of 13/10). Training data keep the thesis scale
(divided by `mu_ref`), and every row of the comparison table uses `psnr`, `ssim` like the existing runs (peak 1 on that
scale, predictions clipped to [0, 1], averaged per slice). The existing evaluation results are therefore kept, not
recomputed (`mac_push_compare.sh` pushes the old `eval/` to the new machine). The CG-GLORE paper's metric (per-slice
min-max normalisation, then torchmetrics) is available in `core/metrics/image.py` (`psnr_minmax`, `ssim_minmax`) but is
not written into the results. The thesis runs (LEARN-DenseNet + L-BFGS, RegFormer) are not retrained either: the table
compares their existing results with the newly trained methods.

A trial of a few dozen steps at full size on a phantom (Mac, 13/10) with each author's recipe showed that the code of
FBPConvNet, Learned Primal-Dual and original LEARN learns; the shared recipe has not been tried this way (its lr is
smaller or equal, so it is less risky), and CG-GLORE and DPMA are too slow on a CPU to try.

**CG-GLORE versus the original code.** Architecture, initialisation and CG iteration count are unchanged; checked by
loading the same weights into the original `RegularizationBlock` class and `conjugate_gradient` function: the outputs
are identical (regularisation block on a 256 image, 10 CG iterations). Differences come from plugging into the
project: the project's projection operator and FBP replace ODL/ASTRA; the CG inner products and the normalisation
constant of the pseudo-inverse are computed per sample (the original always runs with batch size 1, where both
coincide). Note: "A^T" in the original code is the FBP, so the CG system is not symmetric; kept as in the original.

**DPMA (added 13/10 at the user's request).** Rebuilt from the original code in `core/models/baselines/dpma/`, with the
three parts of the original: (1) network M1 shared by sinograms and images, a three-term L1 loss; (2) network M2
correcting the image with M1 frozen, MSE loss; (3) 300 non-learned refinement iterations (preconditioned data step +
Chambolle TV) at inference. Training recipe: the shared CG-GLORE recipe, 50 epochs per stage (DPMA's original code:
lr 5e-4, weight decay 1e-5, 100 epochs). The original code targets sparse view (64 angles interpolated to 192); the
limited-angle adaptations:

- the sinogram is filled up to a 240° scan range with the same angular step (`configs/geometry/fan_a240.yaml`; the
  240° FBP of a phantom reaches 38.1 dB); the measured part is kept, the missing part is filled by re-projecting the
  FBP image instead of nearest-neighbour interpolation;
- sinograms are multiplied by 0.192 (= 76.8 / 400) to keep the sinogram/image ratio and the loss weights of the
  original code;
- the denominator FBP(A 1) of the data step in the 300 iterations is clamped at 0 from below.

Checked: the network has 8,056,321 parameters (the CG-GLORE paper reports 8.1M); the hand-written derivative of KBA
matches autograd; the GPU TV equals `skimage.restoration.denoise_tv_chambolle`, which the original code calls; the 300
refinement iterations are stable at full size. NOT yet checked by running the original code next to the rebuilt one (as
was done for CG-GLORE): the original is third-party code that does not run on this machine. The validation PSNR recorded
during training is that of the learned part (`validation_forward`); the 300 refinement iterations only run at
evaluation. `dpma1_*` runs are only stage 1 and are not tabulated. Cost: on the Mac CPU one stage-1 step takes 80 s
(CG-GLORE 98 s, LEARN-DenseNet 14 s), stage 2 takes 12 s; the GPU memory of stage 1 has not been measured, see
`--plan`.

**Running on a new rented machine** (old runs are not retrained):

```
# Mac: source code + 32 finished runs + the old evaluation results
bash scripts/vast/mac_push_compare.sh <IP> <PORT> ../ket-qua/vast_results_2026-10-12/outputs_joint
# rented machine
bash scripts/vast/phase2.sh --compare-only --plan          # build the data, check old runs, measure speed, print GPU-hours, STOP
bash scripts/vast/phase2.sh start --compare-only --auto-stop
```

The check (`lact eval recheck`) must reproduce the validation PSNR of the old Mayo runs on the rebuilt data store;
otherwise it stops, because new and old runs would no longer share the same data. Same GPU type (RTX 5090), PyTorch
version and TF32 mode as phase 2 (`scripts/vast/setup.sh`).
