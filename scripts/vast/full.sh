#!/usr/bin/env bash
# The whole E0-E6 pipeline on ONE rented machine (no server needed): download data -> prepare (A, E0, simulation, E1,
# TV) -> measure speed to choose runs/GPU -> E2 -> E3/E4/E5 -> test evaluation, E6, cost, tables/figures.
# Called through scripts/vast/start.sh (runs in tmux). Re-runnable: finished steps/runs are skipped.
#   bash scripts/vast/full.sh [--per-gpu N] [--auto-stop] [--extend] [--plan] [--then VARIANT]
#   VARIANT=mayo_poisson_1e6 bash scripts/vast/full.sh      # data variant: configs/variant/<name>.yaml
#   MODELS="learn_unet learn_unet_regnet" bash scripts/vast/full.sh   # run only some models (kept when resuming)
# --extend: accept existing outputs after the grid was extended (e.g. LEARN-RegNet added); train only the missing runs.
# --plan: only prepare data, check old runs, measure speed and print the GPU-hour estimate, then STOP (no training).
# --then VARIANT: once this set has fully finished, start that variant automatically (e.g. --then deeplesion) so the
#   machine does not sit idle; --auto-stop then applies to the last set.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

per_gpu=""
auto_stop=0
plan_only=0
then_variant=""
extend=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --per-gpu) per_gpu="$2"; shift 2 ;;
        --auto-stop) auto_stop=1; shift ;;
        --extend) extend=(--extend); shift ;;
        --plan) plan_only=1; shift ;;
        --then) then_variant="$2"; shift 2 ;;
        *) echo "Invalid argument: $1" >&2; exit 2 ;;
    esac
done
# Arrays may be empty: always write ${a[@]+"${a[@]}"} (bash < 4.4 treats an empty array as unset under `set -u`).
models=()
# shellcheck disable=SC2206
[[ -n "${MODELS:-}" ]] && models=(--models $MODELS)
n_gpu=$(nvidia-smi -L | wc -l)
threads_for() {  # CPU threads per process when running $1 processes/GPU
    local t=$(( $(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc) / (n_gpu * $1) ))
    (( t < 1 )) && t=1
    (( t > 8 )) && t=8
    echo "$t"
}
step() { echo; echo "== $(date '+%F %T') $*"; }
fail() { echo "$*" > "$FAILED_FILE"; echo "STOPPED: $*"; exit 1; }
rm -f "$DONE_FILE" "$FAILED_FILE"
[[ -z "$then_variant" || -f "configs/variant/$then_variant.yaml" ]] || fail "--then: configs/variant/$then_variant.yaml does not exist"

if [[ "$VARIANT" == deeplesion* ]]; then
    step "download DeepLesion (skipping extracted files) and split it (if not done yet)"
    lact data download-deeplesion -c "$CFG" configs/data/deeplesion.yaml --zips ${DEEPLESION_ZIPS:-1 2} \
        || fail "DeepLesion download failed"
    if [[ ! -f configs/data/split_deeplesion.yaml ]]; then
        lact data split-deeplesion -c "$CFG" configs/data/deeplesion.yaml || fail "DeepLesion split failed"
    fi
else
    step "download the Mayo data (skipped if complete)"
    lact data download-mayo -c "$CFG" --delete-zip || fail "Mayo download failed"
fi
[[ -z "$VARIANT" || -f "configs/variant/$VARIANT.yaml" ]] || fail "configs/variant/$VARIANT.yaml does not exist"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
step "prepare: preprocessing, matrix A, E0, simulation, E1, TV selection"
lact pipeline run -c "${CFGS[@]}" --gpus all --per-gpu 1 --until prepare ${extend[@]+"${extend[@]}"} ${models[@]+"${models[@]}"} || fail "preparation step failed"

# Runs brought over from another machine (an earlier batch) must reproduce their validation PSNR on the data just built
# here; otherwise stop BEFORE spending GPU time on new runs trained on data that differs from the old runs.
step "check existing runs against the data on this machine"
CUDA_VISIBLE_DEVICES=0 lact eval recheck -c "${CFGS[@]}" \
    || fail "existing runs do NOT match the data/matrices on this machine (see docs/VAST.md, Troubleshooting)"

pilot_csv=.runtime/pilot/k1/pilot_fan_a120.csv
if [[ ! -f .runtime/pilot/per_gpu.txt || ! -f "$pilot_csv" ]]; then
    step "measure speed (pilot): choose runs/GPU, estimate GPU-hours"
    VARIANT="$VARIANT" bash scripts/vast/pilot.sh || echo "(pilot failed: using 1 run/GPU)"
elif [[ -f "$pilot_csv" ]]; then
    step "estimate the remaining training (from the existing pilot measurements)"
    lact train plan -c "${CFGS[@]}" --pilot "$pilot_csv" --n-gpus "$n_gpu" \
        --gain "$(cat .runtime/pilot/gain.txt 2>/dev/null || echo 1)" ${models[@]+"${models[@]}"} 2>&1 \
        | grep -v "Warning\|does not exist yet" | tee ".runtime/pilot/plan$TAG.txt" || true
fi
[[ -n "$per_gpu" ]] || per_gpu=$(cat .runtime/pilot/per_gpu.txt 2>/dev/null || echo 1)
if [[ "$plan_only" == "1" ]]; then
    step "--plan: stopping here, nothing trained. See the estimate above (or .runtime/pilot/plan$TAG.txt)."
    echo "Real run: bash scripts/vast/start.sh ${extend[*]}"
    exit 0
fi
t=$(threads_for "$per_gpu")
export OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t OPENBLAS_NUM_THREADS=$t
# Several runs/GPU: schedule by the peak memory the pilot measured (runs needing over half a GPU run alone, light runs
# share).
mem=()
[[ "$per_gpu" -gt 1 && -f "$pilot_csv" ]] && mem=(--mem-table "$pilot_csv")
# Weak GPUs on this machine (e.g. a degraded PCIe link: two shared runs are slower than running them one after the
# other) take only one run:
#   SOLO_GPUS="2" bash scripts/vast/start.sh ...   or write the GPU indices to .runtime/solo_gpus.txt
SOLO_GPUS="${SOLO_GPUS:-$(cat .runtime/solo_gpus.txt 2>/dev/null || true)}"
# shellcheck disable=SC2206
[[ "$per_gpu" -gt 1 && -n "$SOLO_GPUS" ]] && mem+=(--solo-gpus $SOLO_GPUS)
step "E2 -> E3/E4/E5 -> evaluation, report: $n_gpu GPUs x up to $per_gpu runs, $t CPU threads/run"
lact pipeline run -c "${CFGS[@]}" --gpus all --per-gpu "$per_gpu" ${mem[@]+"${mem[@]}"} \
    ${extend[@]+"${extend[@]}"} ${models[@]+"${models[@]}"}
code=$?
step "pipeline finished, exit code $code"
if [[ $code -ne 0 ]]; then
    fail "pipeline exit code $code (see the log; re-run scripts/vast/start.sh to resume)"
fi
out_root=$("$PY" -c 'import sys; from lact.config import load_config; print(load_config(sys.argv[1:]).paths.outputs)' \
    "${CFGS[@]}")
step "collect the thesis plotting data into $out_root/figdata (compact, to pull to a personal machine)"
"$PY" experiments/figures/export_results.py --outputs "$out_root" --out "$out_root/figdata" \
    || echo "(skipped) could not collect figdata; the command above can be re-run later"
date '+%F %T' > "$DONE_FILE"
echo "DONE. Results: $out_root/{tables,figures,runs,figdata}"
if [[ -n "$then_variant" ]]; then
    step "continue with variant $then_variant (own tmux session, log and status: VARIANT=$then_variant bash scripts/vast/status.sh)"
    next_args=()
    [[ "$auto_stop" == "1" ]] && next_args=(--auto-stop)
    LACT_CHAIN=1 VARIANT="$then_variant" MODELS="${MODELS:-}" bash scripts/vast/start.sh ${next_args[@]+"${next_args[@]}"}
    exit 0
fi
# Toggle at any time while running: touch .runtime/AUTO_STOP (or delete that file).
[[ -f .runtime/AUTO_STOP ]] && auto_stop=1
if [[ "$auto_stop" == "1" && -n "${CONTAINER_ID:-}" ]]; then
    echo "Stopping machine $CONTAINER_ID automatically (data stays on disk)"
    "$PY" -m pip install -q vastai >/dev/null 2>&1 || true
    "$(dirname -- "$PY")/vastai" --api-key "${CONTAINER_API_KEY:-}" stop instance "$CONTAINER_ID" \
        || echo "Could not Stop automatically: Stop it on the Vast website"
fi
