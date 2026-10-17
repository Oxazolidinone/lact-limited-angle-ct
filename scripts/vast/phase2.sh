#!/usr/bin/env bash
# Phase 2 on ONE rented machine (docs/PHASE2.md): ResNet, U-Net, DenseNet (LEARN framework) and RegFormer on the two
# datasets Mayo and DeepLesion; every model is trained TWICE, both FROM SCRATCH: plain and L-BFGS data step.
#   first-order:   base_mayo (RegFormer; the three LEARN backbones reuse their E3 runs) + base_dl (all four)
#   L-BFGS:        lbfgs_mayo + lbfgs_dl, same recipe as the first-order runs (same lr and epochs), NO checkpoint loaded
#   comparison:    cmp_mayo + cmp_dl, the four methods of the comparison table of the CG-GLORE paper (CG-GLORE, original
#                  LEARN, FBPConvNet, Learned Primal-Dual), also trained from scratch on the same data (docs/PHASE2.md,
#                  section 10); and DPMA, trained in two stages as in the original code: dpma1_* (network M1), then
#                  dpma_* (loads M1 frozen, trains M2)
# Everything is in ONE queue. (The 10/10 version had the L-BFGS runs load the first-order checkpoints; the user asked
# for training from scratch on 11/10. The queue still understands train.init_from if a grid uses it: such a run waits
# for its source run to be DONE.)
# No lr sweep: settings follow the papers (see docs/PHASE2.md, section 2). Afterwards, evaluate on test on EACH dataset
# and write tables C1-C3. Re-runnable: finished steps and runs are skipped, unfinished runs resume from checkpoints.
#
#   bash scripts/vast/phase2.sh start [options]     # runs in tmux: the rented machine keeps going after SSH closes
#   bash scripts/vast/phase2.sh [options]           # runs directly
#   VARIANT=joint bash scripts/vast/status.sh       # progress  |  VARIANT=joint bash scripts/vast/stop.sh   # graceful stop
#
# Options:
#   --plan         only prepare data + measure speed + print the GPU-hour estimate, then STOP (no training)
#   --until base   run only step 1, then evaluate (table C1) and stop, before any L-BFGS run
#   --control      NO LONGER NEEDED since 11/10 (L-BFGS runs train from scratch and compare directly with the
#                  first-order runs): re-run the L-BFGS grid as ctrl_* with the old data step, i.e. repeat the
#                  first-order runs. Only meaningful when the L-BFGS grid uses train.init_from.
#   --compare-only only measure and train the comparison group (cmp_*); used on a new machine once the base_* / lbfgs_*
#                  runs were pushed with scripts/vast/mac_push_compare.sh. Evaluation and tables still cover every
#                  experiment with runs on the machine.
#   --per-gpu N    runs trained at the same time on each GPU (default 2)
#   --auto-stop    Stop the machine when everything is done (enable while running: touch .runtime/AUTO_STOP)
# Environment variables:
#   MODELS="learn_resnet regformer"   run only these models (default: every model of the grids)
#   GEOMS="fan_a90 fan_a120"          geometries
set -uo pipefail
export VARIANT=joint
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

if [[ "${1:-}" == "start" ]]; then
    shift
    if tmux -S "$TMUX_SOCK" has-session -t "$SESSION" 2>/dev/null; then
        echo "Already running (tmux '$SESSION'). See: VARIANT=joint bash scripts/vast/status.sh"
        exit 0
    fi
    others=$(tmux -S "$TMUX_SOCK" list-sessions -F '#S' 2>/dev/null | grep '^lact' | tr '\n' ' ' || true)
    if [[ -n "$others" ]]; then
        echo "Another session is running: $others. Wait for it to finish (two pipelines would compete for GPUs)." >&2
        exit 1
    fi
    args=""
    (( $# )) && printf -v args '%q ' "$@"
    tmux -S "$TMUX_SOCK" new-session -d -s "$SESSION" -c "$LACT_ROOT" \
        "MODELS='${MODELS:-}' GEOMS='${GEOMS:-}' bash scripts/vast/phase2.sh $args >> '$LOG' 2>&1"
    echo "Started. Log: $LOG"
    echo "Progress: VARIANT=joint bash scripts/vast/status.sh   |   Graceful stop: VARIANT=joint bash scripts/vast/stop.sh"
    exit 0
fi

plan_only=0; until_base=0; control=0; auto_stop=0; per_gpu=2; compare_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan) plan_only=1; shift ;;
        --until) [[ "${2:-}" == "base" ]] || { echo "--until only accepts base" >&2; exit 2; }; until_base=1; shift 2 ;;
        --control) control=1; shift ;;
        --compare-only) compare_only=1; shift ;;
        --per-gpu) per_gpu="$2"; shift 2 ;;
        --auto-stop) auto_stop=1; shift ;;
        *) echo "Invalid argument: $1" >&2; exit 2 ;;
    esac
done
read -r -a GEOM_A <<< "${GEOMS:-fan_a90 fan_a120}"
read -r -a MODELS_A <<< "${MODELS:-}"
GEOM_MATCH=("${GEOM_A[@]/#/_}")                             # grids keep only runs of these geometries: _fan_a90_ ...
GEOM_MATCH=("${GEOM_MATCH[@]/%/_}")
SPEC=configs/experiment
BASE=(base_mayo base_dl)                                    # step 1: first-order training
QN=(lbfgs_mayo lbfgs_dl)                                    # L-BFGS data step, trained from scratch
CMP=(cmp_mayo cmp_dl dpma1_mayo dpma1_dl dpma_mayo dpma_dl)  # comparison table of the CG-GLORE paper, from scratch
CMP_EVAL=(cmp_mayo cmp_dl dpma_mayo dpma_dl)                                # dpma1_* is only stage 1 of DPMA: not tabulated
LAST=()                                                     # runs LAST (the K = 30 group was dropped on 15/10); empty = skip
TRAIN=("${BASE[@]}" "${QN[@]}" "${CMP[@]}")                 # grids that are measured and queued
[[ "$compare_only" == "1" ]] && TRAIN=("${CMP[@]}")
CTRL=()                                                     # --control: same grids as QN, renamed, old data step
[[ "$control" == "1" ]] && CTRL=("${QN[@]/#lbfgs_/ctrl_}")
CTRL_SETS=(model.update=gd)
JOINT=("${CFGS[@]}" configs/data/split_joint.yaml)          # data-level commands: validation/test splits of both sets
MAYO_ON_JOINT=("${CFGS[@]}" configs/data/split_joint_mayo.yaml)
n_gpu=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
step() { echo; echo "== $(date '+%F %T') $*"; }
fail() { echo "$*" > "$FAILED_FILE"; echo "STOPPED: $*"; exit 1; }
rm -f "$DONE_FILE" "$FAILED_FILE"
(( n_gpu > 0 )) || fail "no GPU found (nvidia-smi -L)"
models_opt=()
[[ ${#MODELS_A[@]} -gt 0 ]] && models_opt=(--models "${MODELS_A[@]}")
if [[ "$auto_stop" == "1" ]]; then     # install while the network is up: on 07/10 auto-stop failed as the network dropped
    "$PY" -m pip install -q vastai >/dev/null 2>&1 || echo "(could not install vastai: remember to Stop the machine by hand)"
fi
out_root=$("$PY" -c 'import sys; from lact.config import load_config; print(load_config(sys.argv[1:]).paths.outputs)' \
    "${CFGS[@]}")

# Run the commands of the CMDS array on the GPUs, at most one per GPU at a time; return 1 if any command failed.
on_gpus() {
    local i=0 bad=0 pids=() p
    for cmd in "${CMDS[@]}"; do
        CUDA_VISIBLE_DEVICES=$(( i % n_gpu )) bash -c "$cmd" &
        pids+=($!)
        i=$(( i + 1 ))
        if (( i % n_gpu == 0 )); then
            for p in "${pids[@]}"; do wait "$p" || bad=1; done
            pids=()
        fi
    done
    for p in ${pids[@]+"${pids[@]}"}; do wait "$p" || bad=1; done
    return $bad
}
lact_cmd() { printf '%q ' "$PY" -m lact "$@"; }
specs() { local s; for s in "$@"; do printf '%s ' "$SPEC/$s.yaml"; done; }

# ---------------------------------------------------------------------------- data
step "download the Mayo and DeepLesion data (skipping what is already there)"
lact data download-mayo -c "$CFG" --delete-zip || fail "Mayo download failed"
# DeepLesion: only the extracted key slices (configs/data/deeplesion_keyslices.csv). They are spread over all 56 zip
# files (220 GiB): each file is downloaded, a few dozen slices are extracted, then it is deleted; 6 files in parallel.
lact data download-deeplesion -c "${CFGS[@]}" --key-slices || fail "DeepLesion download failed"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
step "joint store: preprocessing (Mayo intensity scale), matrix A, simulation ${GEOM_A[*]}"
if [[ ! -f "$out_root/data/images.h5" ]]; then
    lact data preprocess -c "${JOINT[@]}" || fail "preprocessing failed"
fi
lact ops build -c "${CFGS[@]}" -g "${GEOM_A[@]}" || fail "matrix build failed"
# DPMA fills the sinogram up to a 240° scan range (configs/geometry/fan_a240.yaml): only the reconstruction-grid matrix
# is needed
rec_size=$("$PY" -c 'import sys; from lact.config import load_config; print(load_config(sys.argv[1:]).data.img_size_rec)' \
    "${CFGS[@]}")
lact ops build -c "${CFGS[@]}" -g fan_a240 --sizes "$rec_size" || fail "240° matrix build (DPMA) failed"
CUDA_VISIBLE_DEVICES=0 lact data simulate -c "${JOINT[@]}" -g "${GEOM_A[@]}" || fail "simulation failed"
lact data stage -c "${CFGS[@]}" -g "${GEOM_A[@]}" || fail "cache staging failed"

# The three LEARN backbones are not retrained on Mayo: their E3 runs (pushed by scripts/vast/mac_push_phase2.sh) are both
# the first-order Mayo baseline and the source the lbfgs_mayo runs used to load. They must reproduce their validation
# PSNR on the joint store; a mismatch means the Mayo images/matrices here differ from training => stop BEFORE any GPU.
compgen -G "$out_root/runs/e3_main/*/DONE" >/dev/null \
    || fail "missing the Mayo E3 runs in $out_root/runs/e3_main (on the Mac: bash scripts/vast/mac_push_phase2.sh IP PORT <Mayo outputs>)"
step "check the old Mayo runs (e3_main) against the joint store"
CUDA_VISIBLE_DEVICES=0 lact eval recheck -c "${MAYO_ON_JOINT[@]}" --exp e3_main --per-geometry 12 --on-config-data \
    || fail "old Mayo runs do NOT match the joint store (see docs/PHASE2.md, Troubleshooting)"

step "choose the TV parameter on the joint validation split, freeze the configuration"
CMDS=()
for g in "${GEOM_A[@]}"; do
    [[ -f "$out_root/baselines/tv_best_$g.json" ]] || CMDS+=("$(lact_cmd eval tune-tv -c "${JOINT[@]}" -g "$g")")
done
if [[ ${#CMDS[@]} -gt 0 ]]; then on_gpus || fail "TV tuning failed"; fi
[[ -f "$out_root/configs_frozen.lock" ]] \
    || lact eval freeze -c "${CFGS[@]}" --note "phase 2: settings from the papers, no lr sweep; TV chosen on the validation split" \
    || fail "could not create the configuration lock"

# ---------------------------------------------------------------------------- speed measurement, estimate
pilot_dir=.runtime/pilot2
plan_file=".runtime/pilot/plan$TAG.txt"
mkdir -p "$pilot_dir" .runtime/pilot
sig="${GEOM_A[*]}|${MODELS_A[*]-}|${TRAIN[*]} ${CTRL[*]-}"
if [[ "$(cat "$pilot_dir/SIGNATURE" 2>/dev/null)" != "$sig" ]]; then
    step "measure speed (pilot): sec/step and memory of each run group"
    rm -f "$pilot_dir"/pilot_*.csv
    CMDS=()
    for g in "${GEOM_A[@]}"; do
        for e in "${TRAIN[@]}"; do
            CMDS+=("$(lact_cmd train pilot -c "${CFGS[@]}" --geometry "$g" -s "paths.tables=$pilot_dir" \
                --spec "$SPEC/$e.yaml" --steps 60 ${models_opt[@]+"${models_opt[@]}"})")
        done
        for e in ${CTRL[@]+"${QN[@]}"}; do      # control runs: the step-2 grid with the old data step
            CMDS+=("$(lact_cmd train pilot -c "${CFGS[@]}" --geometry "$g" -s "paths.tables=$pilot_dir" "${CTRL_SETS[@]}" \
                --spec "$SPEC/$e.yaml" --tag _ctrl --steps 60 ${models_opt[@]+"${models_opt[@]}"})")
        done
    done
    on_gpus > "$pilot_dir/pilot.log" 2>&1 || echo "(a pilot measurement failed: see $pilot_dir/pilot.log; the estimate may be incomplete)"
    echo "$sig" > "$pilot_dir/SIGNATURE"
fi
gpu_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
"$PY" - "$pilot_dir" "$n_gpu" "$per_gpu" "${gpu_mb:-0}" <<'EOF' | tee "$plan_file"
import sys
from pathlib import Path
import pandas as pd
d, n_gpu, per_gpu, gpu_gb = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]) / 1024
rows = []
for f in sorted(d.glob("pilot_*.csv")):
    name = f.stem[len("pilot_"):]
    ctrl = name.endswith("_ctrl")                      # control run: lbfgs_* grid measured with the old data step
    exp, geom = (name[: -len("_ctrl")] if ctrl else name).rsplit("_fan_", 1)
    exp = exp.replace("lbfgs_", "ctrl_", 1) if ctrl else exp
    for r in pd.read_csv(f).itertuples():
        rows.append({"exp": exp, "geom": "fan_" + geom, "model": r.model, "sec": r.sec_per_step,
                     "hours": r.gpu_hours_per_run, "mem_gb": r.peak_mem_mb / 1024})
if not rows:
    sys.exit("No pilot measurements yet.")
df = pd.DataFrame(rows)
print("Sec/step and peak memory (pilot, 1 run/GPU):")
for (exp, geom), g in df.groupby(["exp", "geom"], sort=False):
    print(f"  {exp:10s} {geom}: " + " | ".join(f"{r.model.replace('learn_', '')} {r.sec:.2f}s, {r.mem_gb:.1f} GB"
                                             for r in g.itertuples()))
SLOW = 1.4          # real training (deterministic, validation every epoch) is slower than the pilot; ~1.4x on 05/10
total = 0.0
print("Estimate (multiplied by 1.4 for real training):")
for exp, g in df.groupby("exp", sort=False):
    h = g.hours.sum() * SLOW
    total += h
    print(f"  {exp:10s} {len(g):3d} runs  {h:6.0f} GPU-hours")
print(f"  TOTAL {len(df)} runs, {total:.0f} GPU-hours")
print(f"  With {n_gpu} GPUs: about {total / n_gpu:.0f} hours if {per_gpu} runs/GPU do not reduce the total throughput. "
      f"Rental cost ~ hours x the machine's hourly price.")
heavy = df[df.mem_gb * per_gpu > 0.9 * gpu_gb] if gpu_gb else df[:0]
if len(heavy):
    print(f"  NOTE: {sorted(set(heavy.model))} use up to {heavy.mem_gb.max():.1f} GB per run; {per_gpu} runs do not fit one "
          f"{gpu_gb:.0f} GB GPU: runs will switch to gradient checkpointing (slower); prefer --per-gpu 1.")
EOF
if [[ "$plan_only" == "1" ]]; then
    step "--plan: stopping here, nothing trained. Real run: bash scripts/vast/phase2.sh start --auto-stop"
    exit 0
fi

# ---------------------------------------------------------------------------- training
t=$(( $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc) / (n_gpu * per_gpu) ))
(( t < 1 )) && t=1
(( t > 8 )) && t=8
export OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t
# Pilot measurements: the longest runs (RegFormer) are queued first so each long run gets its own GPU and short runs
# fill in afterwards; runs needing over half the GPU memory run alone.
mem=()
for f in "$pilot_dir"/pilot_*.csv; do [[ -f "$f" ]] && mem+=("$f"); done
[[ ${#mem[@]} -gt 0 ]] && mem=(--mem-table "${mem[@]}")
grid() {
    lact train grid "$@" ${mem[@]+"${mem[@]}"} --match "${GEOM_MATCH[@]}" -c "${CFGS[@]}" --run --gpus all \
        --per-gpu "$per_gpu"
}
check_grid() {   # $1 = exit code of the grid, $2 = name
    [[ $1 -eq 3 ]] && { echo "Stopped midway ($2): re-run scripts/vast/phase2.sh start to resume."; exit 3; }
    [[ $1 -eq 0 ]] || fail "$2: some runs failed, diverged, or could not run because their source run broke (VARIANT=joint bash scripts/vast/status.sh; see that run's log)"
}
evaluate() {     # test evaluation: the given experiments (one GPU each), the Mayo E3 runs, FBP/TV; then write the tables
    CMDS=()
    for e in "$@"; do               # skip experiments with no finished run on this machine (e.g. --compare-only)
        compgen -G "$out_root/runs/$e/*/DONE" >/dev/null || { echo "(no finished run of $e: skipped)"; continue; }
        CMDS+=("$(lact_cmd eval runs -c "${JOINT[@]}" --exp "$e" --split test)")
    done
    CMDS+=("$(lact_cmd eval runs -c "${MAYO_ON_JOINT[@]}" --exp e3_main --on-config-data --split test)")
    for g in "${GEOM_A[@]}"; do CMDS+=("$(lact_cmd eval baselines -c "${JOINT[@]}" --split test -g "$g")"); done
    on_gpus || fail "evaluation failed"
    lact report phase2 -c "${JOINT[@]}" --geoms "${GEOM_A[@]}" || fail "writing the tables failed"
}

if [[ "$until_base" == "1" ]]; then
    step "step 1 only, first-order training (${BASE[*]}): $n_gpu GPUs x $per_gpu runs, $t CPU threads/run"
    # shellcheck disable=SC2046
    grid $(specs "${BASE[@]}") ${models_opt[@]+"${models_opt[@]}"}
    check_grid $? "step 1"
    step "test evaluation of step 1 on each dataset"
    evaluate "${BASE[@]}"
    step "--until base: stopping here. Table: $out_root/tables/C1_datasets.md. Continue with L-BFGS: bash scripts/vast/phase2.sh start"
    exit 0
fi

MAIN=(); for e in "${TRAIN[@]}"; do [[ " ${LAST[*]} " == *" $e "* ]] || MAIN+=("$e"); done
step "train (${MAIN[*]}) in one queue: $n_gpu GPUs x $per_gpu runs, $t CPU threads/run"
# shellcheck disable=SC2046
grid $(specs "${MAIN[@]}") ${models_opt[@]+"${models_opt[@]}"}
check_grid $? "train"
step "test evaluation and tables for the main runs (before the extra group that runs last)"
evaluate "${BASE[@]}" "${QN[@]}" cmp_mayo cmp_dl dpma_mayo dpma_dl
if [[ ${#LAST[@]} -gt 0 ]]; then
    step "extra group run last (${LAST[*]}): failures here do not block the results of the main runs"
    # shellcheck disable=SC2046
    grid $(specs "${LAST[@]}") ${models_opt[@]+"${models_opt[@]}"}
    rc=$?; [[ $rc -eq 3 ]] && { echo "Stopped midway (extra group): re-run scripts/vast/phase2.sh start to resume."; exit 3; }
    [[ $rc -eq 0 ]] || echo "(some runs of the extra group failed: still evaluating the finished runs)"
fi
if [[ ${#CTRL[@]} -gt 0 ]]; then
    step "step-2 control: train further like the L-BFGS runs but keep the old data step (${CTRL[*]})"
    # shellcheck disable=SC2046
    grid $(specs "${QN[@]}") ${models_opt[@]+"${models_opt[@]}"} --exp-rename lbfgs_=ctrl_ -s "${CTRL_SETS[@]}"
    check_grid $? "control"
fi
step "test evaluation on each dataset, tables C1-C3"
evaluate "${BASE[@]}" "${QN[@]}" "${CMP_EVAL[@]}" ${CTRL[@]+"${CTRL[@]}"}

# Collect the plotting data (reconstructions of the illustrative slices, per-slice metrics, per-stage parameters,
# training curves) into figdata to pull to a personal machine; the full predictions (eval/preds) are too large to pull.
# The script is invoked at this point, so an update made while training is still picked up.
step "collect the plotting data into $out_root/figdata"
if [[ -f experiments/figures/export_phase2.py ]]; then
    "$PY" experiments/figures/export_phase2.py --outputs "$out_root" --out "$out_root/figdata" \
        || echo "(could not collect figdata: the command above can be re-run later)"
else
    echo "(experiments/figures/export_phase2.py not found: skipped)"
fi

date '+%F %T' > "$DONE_FILE"
echo "DONE. Results: $out_root/tables/{C1_datasets,C2_second_order,C3_runs}.* and $out_root/{runs,eval}"
[[ -f .runtime/AUTO_STOP ]] && auto_stop=1
if [[ "$auto_stop" == "1" && -n "${CONTAINER_ID:-}" ]]; then
    echo "Stopping machine $CONTAINER_ID automatically (data stays on disk)"
    "$PY" -m pip install -q vastai >/dev/null 2>&1 || true
    "$(dirname -- "$PY")/vastai" --api-key "${CONTAINER_API_KEY:-}" stop instance "$CONTAINER_ID" \
        || echo "Could not Stop automatically: Stop it on the Vast website"
fi
