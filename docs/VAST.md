# Running on rented Vast.ai machines

One rented machine runs all of E0–E6 for each dataset: download → preparation → E2 → E3/E4/E5 → evaluation, E6,
tables/figures. Every step can be re-run: finished steps and runs are skipped, unfinished runs resume from their
checkpoint.

Phase 2 (joint Mayo + DeepLesion store, RegFormer, L-BFGS data step, comparison group, 2DeteCT) has its own scripts and
guide: `scripts/vast/phase2.sh`, `scripts/vast/detect.sh`, [docs/PHASE2.md](PHASE2.md).

## Command summary

On the personal machine (in `kltn-lact`), with `IP` and `PORT` from the *Direct ssh connect* entry of the instance
card:

```bash
bash scripts/vast/mac_push.sh IP PORT ../ket-qua/vast_results_2026-10-02/main/outputs   # code + 46 trained runs (~0.45 GB)
```

On the Vast machine:

```bash
cd /workspace/kltn-lact
bash scripts/vast/setup.sh                                  # install lact + ASTRA, check the GPUs
bash scripts/vast/start.sh --plan                           # prepare + measure speed + ESTIMATE GPU-hours, no training
bash scripts/vast/status.sh                                 # estimate and progress
bash scripts/vast/start.sh --then deeplesion --auto-stop    # Mayo, then DeepLesion, then Stop the machine
```

When done, on the personal machine:

```bash
bash scripts/vast/mac_pull.sh IP PORT                       # compact copy into ../ket-qua/vast_results_<date>
```

## 1. Renting a machine

| Item | Choose | Why |
|---|---|---|
| Template | Vast's PyTorch (CUDA 12.x) | `setup.sh` keeps the installed PyTorch; the 02/10 batch used PyTorch 2.11 + CUDA 12.8 |
| GPU | RTX 5090 (32 GB) | Estimated peak training memory: LEARN–EfficientNet ≈ 22 GB, LEARN–ResNeXt ≈ 10 GB, the other models ≤ 9 GB (ConvNeXt measured at 9 GB). 24 GB cards work too, but EfficientNet then switches to gradient checkpointing and is slower |
| GPU count | 4–8 | One run per GPU; more GPUs scale almost linearly at nearly the same total price |
| CPU, RAM | ≥ 4 cores and ≥ 16 GB RAM per GPU | Each run preloads its training set into RAM |
| Disk | ≥ 200 GB | Each dataset needs about 70–75 GB (matrices A, sinograms, cache, predicted images) |
| Rental type | On-demand | Interruptible machines resume too, but are often interrupted |

## 2. Pushing code and trained runs

`scripts/vast/mac_push.sh IP PORT [OLD_OUTPUTS]` syncs the source code into `/workspace/kltn-lact`. With `OLD_OUTPUTS`
it also pushes:

- `outputs/runs`: the 46 runs of the 02/10 batch (E2–E5 of CNN, ResNet, ConvNeXt, ViT), without the `last.pt` of
  finished runs (0.45 GB instead of 1.7 GB). The pipeline sees the `DONE` markers, checks the saved `config.yaml` and
  skips them, so on Mayo only 67 new runs are trained (one seed per configuration).
- Preprocessed images need not be pushed: the rented machine downloads the raw data and rebuilds `images.h5` in a few
  minutes, then the check of existing runs (section 4) confirms that the rebuilt data matches training. With a fast
  upload link, `--with-images` pushes `outputs/data/{images.h5,stats.json,slices.json}` as is (1.5 GB).

Files already on the rented machine are never overwritten, so re-running the command midway (to update the code) is
safe. The split file `configs/data/split_deeplesion.yaml` (45/6/15 patients, generated from the real data on 05/10/2026)
ships with the code; the rented machine only downloads the data and uses exactly that file.

## 3. Installation

`bash scripts/vast/setup.sh` installs `lact` and ASTRA **2.5.0** (the version that built the matrices A of the 46 old
runs) and checks every GPU.

## 4. Previewing the cost

`bash scripts/vast/start.sh --plan` runs in tmux (it survives closing SSH), about 30 minutes for Mayo:

1. download the Mayo data (0.7 GB), build the matrices A, simulate the sinograms, E0, E1, choose the TV parameters;
2. **check existing runs**: re-evaluate the `best` checkpoint of 6 old runs on the validation split; each must
   reproduce the PSNR recorded during training (within 0.02 dB). On a mismatch it stops at once, see Troubleshooting;
3. measure the real seconds/step and peak memory of the 18 models and choose how many runs share each GPU (models that
   run out of memory are measured again with gradient checkpointing, as the real runs would do);
4. print the estimate: runs still to train, total GPU-hours, wall-clock hours with the machine's GPU count.

`bash scripts/vast/status.sh` prints the `TOTAL` and `With N GPUs` lines again. Rental cost ≈ wall-clock hours × the
machine's hourly price. To preview DeepLesion too before deciding: `VARIANT=deeplesion bash scripts/vast/start.sh
--plan` (8 GB download, about 30 more minutes), then `VARIANT=deeplesion bash scripts/vast/status.sh`.

Measured on 05/10/2026 on 4× RTX 5090 (one seed per configuration): Mayo had 67 new runs ≈ 193 GPU-hours, DeepLesion
93 runs ≈ 240 GPU-hours at pilot speed. Real training is about 1.4 times slower than the pilot (deterministic mode), but
two light runs sharing a GPU give 1.7–1.9 times the throughput (LEARN–EfficientNet needs 22.7 GB and runs alone).

## 5. The real run

```bash
bash scripts/vast/start.sh --then deeplesion --auto-stop
```

- Without `--then`: Mayo only. DeepLesion alone: `VARIANT=deeplesion bash scripts/vast/start.sh`.
- Two datasets never run at the same time on one machine (each pipeline schedules runs on every GPU); `start.sh`
  refuses when another session is running.
- `--auto-stop`: Stop the machine after the last set (GPU billing ends, the data stays on disk). It can be enabled
  later: `touch .runtime/AUTO_STOP`. If the log says "Could not Stop automatically", Stop the machine on the Vast
  website.
- `--per-gpu N`: force the maximum number of runs per GPU instead of the pilot's choice. With N > 1 the pipeline
  schedules runs by the peak memory the pilot measured (`--mem-table`): runs needing more than half the GPU memory
  (LEARN–EfficientNet, 22.7 GB) run alone, light runs share.
- `MODELS="learn_cnn learn_unet ..." bash scripts/vast/start.sh`: run only some models (keep the same list when
  resuming; add `--extend` when the list changes).
- Noisy variants (optional): `VARIANT=mayo_poisson_1e6` or `VARIANT=deeplesion_poisson_1e6`.

## 6. Monitoring

```bash
bash scripts/vast/status.sh                      # state, GPUs, runs per experiment, last log lines
VARIANT=deeplesion bash scripts/vast/status.sh   # the DeepLesion set
bash scripts/vast/progress.sh                    # overview of every queue (phase 2 and 2DeteCT)
python3 scripts/vast/watch.py                    # progress bar of each run (add --outputs outputs_deeplesion)
python3 scripts/vast/gpus.py -w 10               # which run each GPU is running
tail -f .runtime/logs/train.log                  # full log (DeepLesion: train-deeplesion.log)
bash scripts/vast/stop.sh                        # graceful stop; run start.sh again to resume
```

A finished set leaves `.runtime/TRAIN_DONE` (or `TRAIN_DONE-deeplesion`); a failure leaves `TRAIN_FAILED...` with the
reason.

## 7. Pulling the results

`bash scripts/vast/mac_pull.sh IP PORT [DEST] [--full]` pulls every `outputs*` set:

- By default (compact copy): `tables/`, `figures/`, `figdata/`, `runs/` (with `best.pt`, without `last.pt`),
  `eval/per_slice`, logs and pilot measurements. Enough to write the thesis, redraw the figures and move the runs to
  another machine later.
- `--full`: also the predicted images `eval/preds` (tens of GB) and `images.h5`.

Draw the thesis figures on the personal machine from `figdata` (collected by the pipeline's last step):

```bash
python experiments/figures/make_figures.py --data ../ket-qua/vast_results_<date>/outputs/figdata --out figs_mayo --lang vi --preset thesis
python experiments/figures/make_figures.py --data ../ket-qua/vast_results_<date>/outputs_deeplesion/figdata --out figs_deeplesion --lang vi --preset thesis
```

Once the results are pulled and checked, Destroy the machine on the Vast website (a stopped machine still bills disk).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `STOPPED: pipeline exit code ...`, machine restarted, SSH lost | Run `bash scripts/vast/start.sh ...` again with the same arguments: it resumes where it stopped |
| `The configuration differs from the pipeline saved at ...` | Code/configuration or `MODELS` changed since the previous run on that machine: add `--extend` |
| `existing runs do NOT match the data/matrices on this machine` | Check `python -c "import astra; print(astra.__version__)"` (must be 2.5.0) and whether `outputs/data/images.h5` is the file pushed from the personal machine. Fix and run again. To accept retraining all 46 old runs: rename `outputs/runs` and run again |
| A run has an `OOM` marker | It already switched to gradient checkpointing and still lacked memory: delete the `OOM` file in the run directory and run again with `--per-gpu 1` |
| An E3–E5 run has a `DIVERGED` marker | The pipeline stops at the barrier; read that run's `train.log` before deciding (it is not retried automatically) |
| DeepLesion/Mayo download interrupted | Run again: it resumes where it stopped and the zip files are checked against MD5 |
| Disk full | `df -h .`; delete the `cache/` of a finished set (it can be rebuilt), or grow the disk and run again |

## Syncing with the school server

`scripts/server/vast_push.sh` and `scripts/server/vast_pull.sh` serve the earlier set-up (E2 on the server, E3–E5 on
Vast, the project at the same absolute path as on the server). The workflow in this document needs no server.
