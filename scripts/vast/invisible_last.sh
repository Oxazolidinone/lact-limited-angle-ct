#!/usr/bin/env bash
# Run the two two-stage runs (experiments/analysis/invisible_two_stage.py) LAST, without competing for GPUs with the
# RegFormer runs of the main queue (scripts/vast/phase2.sh). The main queue Stops the machine when done, so "last" here
# means: wait until both lbfgs_dl RegFormer runs are on a GPU (by then no run of the queue is waiting for a slot), then
# run on the GPUs WITHOUT any RegFormer run. The two-stage runs resume from their stage-1 checkpoint. If the main queue
# finishes first, exit without running anything.
#
#   setsid nohup bash scripts/vast/invisible_last.sh > .runtime/logs/invisible_last.log 2>&1 < /dev/null &
#   bash scripts/vast/invisible_last.sh --dry      # only print the status and the GPUs it would pick now; runs nothing
export VARIANT=joint
# shellcheck source=scripts/vast/env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh" || exit 2
GEOMS_INV=(fan_a120 fan_a90)          # the longer run starts first, on the least busy GPU

reg_started() {     # $1 = geometry: the lbfgs_dl RegFormer run has a process, or is already DONE
    compgen -G "outputs_joint/runs/lbfgs_dl/lbfgs_dl_regformer_${1}_*/DONE" > /dev/null && return 0
    pgrep -f "lact train run.*regformer\.yaml.*${1}\.yaml.*exp=lbfgs_dl" > /dev/null
}

free_gpus() {       # indices of GPUs without a RegFormer run (two-stage runs included), least busy first
    local idx uuid n reg pid
    while IFS=', ' read -r idx uuid; do
        n=0; reg=0
        while read -r pid; do
            [[ -z "$pid" ]] && continue
            n=$((n + 1))
            tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "regformer\|invisible_two_stage" && reg=1
        done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                     | awk -F', ' -v u="$uuid" '$1 == u {print $2}')
        [[ $reg -eq 0 ]] && echo "$n $idx"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader) | sort -n | awk '{print $2}'
}

if [[ "${1:-}" == "--dry" ]]; then
    for g in fan_a90 fan_a120; do
        if reg_started "$g"; then echo "lbfgs_dl RegFormer $g: on a GPU"; else echo "lbfgs_dl RegFormer $g: NOT running yet"; fi
    done
    echo "GPUs without RegFormer now (least busy first): $(free_gpus | tr '\n' ' ')"
    exit 0
fi

echo "$(now) waiting for both lbfgs_dl RegFormer runs to reach a GPU"
until reg_started fan_a90 && reg_started fan_a120; do
    [[ -f "$DONE_FILE" ]] && { echo "$(now) the main queue finished first: exiting without the two-stage runs"; exit 0; }
    sleep 60
done
sleep 180           # let the run that just started show up on its GPU
mapfile -t gpus < <(free_gpus)
until [[ ${#gpus[@]} -gt 0 ]]; do
    [[ -f "$DONE_FILE" ]] && { echo "$(now) the main queue has finished: exiting"; exit 0; }
    sleep 60
    mapfile -t gpus < <(free_gpus)
done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 LACT_JOBS_PER_GPU=2
i=0
for geom in "${GEOMS_INV[@]}"; do
    gpu=${gpus[$i]:-${gpus[0]}}
    i=$((i + 1))
    echo "$(now) starting the two-stage run $geom on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu LACT_GPU=$gpu setsid nohup "$PY" experiments/analysis/invisible_two_stage.py \
        -c "${CFGS[@]}" --geom "$geom" >> ".runtime/logs/invisible2_$geom.log" 2>&1 < /dev/null &
done
echo "$(now) started; logs: .runtime/logs/invisible2_<geometry>.log"
