#!/usr/bin/env bash
# 2DeteCT pipeline on the rented machine, run ALONGSIDE the phase-2 queue (tmux lact-joint) without sharing GPUs:
#   1) wait for the downloader (tmux detect_dl), fetch any missing slices; check the geometry on 3 full-360° slices
#      (stop if it fails);
#   2) build the store: images.h5, matrices (ASTRA), cropped measured sinograms + FBP, cache; pick the TV parameter on
#      the validation split; freeze the configuration;
#   3) train 16 runs (configs/experiment/*_dt.yaml; K = 30 dropped for good by the user on 15/10): a grid on GPUs 1-3
#      right away, plus a grid on GPU 0 once the old queue (tmux lact-joint) has fully finished; heavy models get a GPU
#      to themselves, light models run 2 per GPU. Both grids share one run list; each run's .train.lock prevents
#      duplicates;
#   4) when all are done, evaluate on test (runs, FBP, TV) and write .runtime/DETECT_DONE. Does NOT Stop the machine.
#
#   tmux -S .runtime/tmux.sock new-session -d -s detect "bash scripts/vast/detect.sh > .runtime/logs/detect.log 2>&1"
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh" || exit 2

CFGS=(configs/env/server.yaml configs/variant/detect.yaml)
DT=("${CFGS[@]}" configs/data/split_detect.yaml)
GEOMS=(dt_a90 dt_a120)
EXPS=(lbfgs_dt base_dt cmp_dt dpma1_dt dpma_dt)
EVAL_EXPS=(lbfgs_dt base_dt cmp_dt dpma_dt)              # dpma1_dt is only stage 1 of DPMA
OUT=outputs_detect
FULL=(1 3000 4600)
MARK=.runtime/DETECT_DONE
step() { echo; echo "== $(now) $*"; }
fail() { echo "STOPPED ($(now)): $*"; exit 1; }
specs() { local e; for e in "$@"; do printf 'configs/experiment/%s.yaml ' "$e"; done; }
alive() { tmux -S "$TMUX_SOCK" has-session -t "$1" 2>/dev/null; }
rm -f "$MARK"

# ---------------------------------------------------------------------------- 1. raw data
step "waiting for the 2DeteCT downloader (tmux detect_dl)"
while alive detect_dl; do sleep 60; done
for k in 1 2 3 4 5; do          # fetch failed slices again (each rerun downloads only what is missing)
    lact data download-2detect -c configs/env/server.yaml configs/data/detect.yaml --full "${FULL[@]}" --workers 48 && break
    [[ $k == 5 ]] && fail "data still incomplete after 5 attempts"
    sleep 120
done

step "checking the geometry on full-360° slices (${FULL[*]})"
"$PY" experiments/analysis/detect_geometry_check.py --raw-dir data_raw/2detect --slices "${FULL[@]}" \
    --out "$OUT/geometry_check" || fail "geometry check failed"
"$PY" - "$OUT/geometry_check/report.json" <<'EOF' || fail "geometry does not match the configuration (see $OUT/geometry_check)"
import json, sys
import yaml
rep = json.load(open(sys.argv[1]))["slices"]
d = yaml.safe_load(open("configs/data/detect.yaml"))["data"]["detect"]
bad = []
for s, r in rep.items():
    b = r["ranking"][0]
    ok = (b["transform"] == d["image_transform"] and b["flip_det"] == d["flip_det"] and b["rel_residual"] < 0.2
          and abs(b["scale_c"] - 1) < 0.15)
    print(f"slice {s}: {b['transform']}/{b['flip_det']} residual {b['rel_residual']:.4f} c={b['scale_c']:.3f} -> {'pass' if ok else 'FAIL'}")
    bad += [] if ok else [s]
sys.exit(1 if bad else 0)
EOF

# ---------------------------------------------------------------------------- 2. data store
gpu_most_free() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
step "building the 2DeteCT store ($OUT)"
[[ -f "$OUT/data/images.h5" ]] || lact data preprocess -c "${DT[@]}" || fail "preprocess failed"
lact ops build -c "${CFGS[@]}" -g "${GEOMS[@]}" dt_a90_to240 dt_a120_to240 --sizes 256 || fail "matrix build failed"
CUDA_VISIBLE_DEVICES=$(gpu_most_free) lact data simulate -c "${DT[@]}" -g "${GEOMS[@]}" || fail "measured-sinogram processing failed"
lact data stage -c "${DT[@]}" -g "${GEOMS[@]}" || fail "cache staging failed"
for g in "${GEOMS[@]}"; do
    [[ -f "$OUT/baselines/tv_best_$g.json" ]] \
        || CUDA_VISIBLE_DEVICES=$(gpu_most_free) lact eval tune-tv -c "${DT[@]}" -g "$g" || fail "TV tuning failed ($g)"
done
[[ -f "$OUT/configs_frozen.lock" ]] \
    || lact eval freeze -c "${CFGS[@]}" --note "2DeteCT: each method's Mayo/DeepLesion training recipe; TV chosen on the validation split" \
    || fail "could not create the configuration lock"

# ---------------------------------------------------------------------------- 3. train
# User, 15/10: the old runs (tmux lact-joint) must finish completely, undisturbed; GPUs without old runs start 2DeteCT
# right away. HEAVY models (CG-GLORE, RegFormer, DPMA, LEARN-DenseNet + L-BFGS) run ALONE on a GPU, LIGHT models
# (original LEARN, LPD, FBPConvNet) run 2 per GPU: enforced by a dedicated schedule table (memory > half a GPU = runs
# alone; sec/step so that long runs start first; on 16/10 the user wanted LEARN-DenseNet + L-BFGS to start BEFORE DPMA,
# so its sec/step is set to 0.50 > DPMA's 0.41, not a measured value). GPU 0 (running old runs) is opened only once the
# lact-joint session has fully ended.
SCHED=.runtime/pilot_dt/sched.csv
mkdir -p "$(dirname "$SCHED")"
cat > "$SCHED" <<'EOF'
model,K,sec_per_step,peak_mem_mb,n_params,n_train_slices,epochs,gpu_hours_per_run,grad_checkpoint,update,metric,cg_iters_cap,cg_iters_used
cg_glore,15,1.13,20000,0,2000,50,0,False,,,0,0
regformer,10,0.45,20000,0,2000,50,0,False,,,0,0
dpma,1,0.41,24230,0,2000,50,0,False,,,0,0
learn_densenet,14,0.50,20000,0,2000,50,0,False,,,0,0
learn_orig,30,0.11,2500,0,2000,50,0,False,,,0,0
lpd,10,0.034,2500,0,2000,50,0,False,,,0,0
fbpconvnet,1,0.009,1800,0,2000,50,0,False,,,0,0
EOF
grid_status() { lact train grid $(specs "${EXPS[@]}") -c "${CFGS[@]}" 2>/dev/null; }
grid_on() {     # $1 = log tag, the rest = GPUs
    local tag=$1; shift
    echo "$(now) opening 2DeteCT grid on GPU $*"
    lact train grid $(specs "${EXPS[@]}") --mem-table "$SCHED" -c "${CFGS[@]}" --run --gpus "$@" --per-gpu 2 \
        >> ".runtime/logs/detect_grid_$tag.log" 2>&1
}
step "train 16 2DeteCT runs: GPUs 1-3 now, GPU 0 when the old queue (tmux lact-joint) has fully finished"
grid_on gpu123 1 2 3 &
p123=$!
while alive lact-joint; do sleep 120; done
grid_on gpu0 0 &
p0=$!
wait "$p123" "$p0"
for k in 1 2; do        # unfinished runs (e.g. locked by the other grid when their turn came): one grid on all 4 GPUs
    n_open=$(grid_status | grep -cE " (todo|partial) ")
    (( n_open == 0 )) && break
    echo "$(now) $n_open runs not finished: opening grid on all 4 GPUs (attempt $k)"
    grid_on all 0 1 2 3
done
grep -E " (diverged|oom) " <<< "$(grid_status)" && echo "(some runs diverged / ran out of memory: still evaluating the finished runs)"

# ---------------------------------------------------------------------------- 4. evaluation
step "2DeteCT test evaluation"
i=0
for e in "${EVAL_EXPS[@]}"; do
    CUDA_VISIBLE_DEVICES=$(( i % 4 )) lact eval runs -c "${DT[@]}" --exp "$e" --split test &
    i=$(( i + 1 ))
done
for g in "${GEOMS[@]}"; do
    CUDA_VISIBLE_DEVICES=$(( i % 4 )) lact eval baselines -c "${DT[@]}" --split test -g "$g" &
    i=$(( i + 1 ))
done
wait
date '+%F %T' > "$MARK"
echo "DONE ($(now)). Results: $OUT/eval (per_slice), $OUT/runs. The machine does NOT Stop itself."
