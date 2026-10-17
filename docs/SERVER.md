# Running directly on the server with .venv

Directory: `/home/minhna/kltn-lact`. The environment is the `.venv` **inside the project**, separate from
`/home/minhna/.venv`. Python 3.11 lives in `.runtime/python`; PyTorch 2.5.1 with CUDA 12.1 and ASTRA 2.3.1 are
installed with pip. Neither Docker nor sudo is needed. The list of installed versions is in
`.runtime/requirements-installed.txt` on the server.

Machine (checked on 28/09/2026): 2 × Xeon Silver 4210 (40 threads), 251 GiB RAM, 8 × RTX 2080 Ti 11 GiB shared with
other users; the system Python is 3.9, so the project uses its own Python 3.11 in `.runtime/python`.

`bash scripts/server/lact.sh ...` calls the project's Python, puts temporary/cache directories in `.runtime`, uses
only the GPUs listed in `.runtime/gpus.txt` and limits CPU libraries to 4 threads per process. No `source activate`
is needed. The run configuration is `configs/env/server.yaml`.

## Progress

```bash
cd /home/minhna/kltn-lact
bash scripts/server/lact.sh pipeline status -c configs/env/server.yaml
tail -f .runtime/logs/pipeline.log
```

`Ctrl+C` in `tail` only stops following the log. The pipeline state is in `outputs/pipeline/main.json`; the
checkpoints of each run in `outputs/runs/`.

## Starting or resuming

Before every (re)start, check that the two chosen GPUs may still be used and are idle:

```bash
cat .runtime/gpus.txt
.venv/bin/python -B scripts/server/pick_gpus.py --n 2
```

The check only prints its result and never changes the selection. `.runtime/gpus.txt` holds exactly two UUIDs separated
by a comma on one line. Do not reuse GPU indices from an earlier check if the state has changed.

The pipeline runs in tmux with its own socket inside the project:

```bash
tmux -S "$PWD/.runtime/tmux.sock" has-session -t lact
```

If the session exists, keep following the log; do not start another one. If there is no session:

```bash
tmux -S "$PWD/.runtime/tmux.sock" new-session -d -s lact -c "$PWD" \
  'bash scripts/server/lact.sh pipeline run -c configs/env/server.yaml --gpus all --per-gpu 1 >> .runtime/logs/pipeline.log 2>&1'
```

`--gpus all` here only uses the UUIDs that `scripts/server/lact.sh` put in `CUDA_VISIBLE_DEVICES`. At most two jobs
run at the same time, one per GPU. Order: preprocessing/matrices → E0, simulation, E1 and TV selection → E2 learning
rate selection → E3/E4/E5 → E6 evaluation and reports.

SSH can be closed and the personal machine turned off. When the **server** restarts, tmux is gone; run the command
above again to reuse the caches and resume from the checkpoints. The configuration saves a checkpoint every 10 minutes
at a training-step boundary, together with the previous checkpoint. The actual interval can be longer when one step is
slow.

## Stopping with a checkpoint

```bash
tmux -S "$PWD/.runtime/tmux.sock" send-keys -t lact C-c
tail -f .runtime/logs/pipeline.log
```

Wait for the pipeline to forward the stop signal to the jobs and for the checkpoints to be saved. Never kill the session
or force-kill processes while jobs are saving.

## Data and environment checks

Full-dose 3 mm data: `data_raw/mayo/full_3mm`, all 2,378 slices of the 10 patients. The download resumes and skips a
dataset that is already complete:

```bash
bash scripts/server/lact.sh data download-mayo --dest data_raw/mayo
```

Logs of the installation, tests, CUDA check and downloads are in `.runtime/logs/`. `outputs/` holds the durable
results; `cache/` holds data/operators shared between experiments. Keep these directories when updating the code.
