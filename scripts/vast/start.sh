#!/usr/bin/env bash
# Run the full pipeline (scripts/vast/full.sh) in tmux so the rented machine keeps running after SSH closes.
#   bash scripts/vast/start.sh [--per-gpu N] [--auto-stop] [--extend] [--plan] [--then VARIANT]
#   VARIANT=mayo_poisson_1e6 bash scripts/vast/start.sh     # data variant (configs/variant/<name>.yaml)
# Without --per-gpu: measure the speed after the preparation step, then choose 1 or 2 runs/GPU.
# --extend: existing outputs whose grid was just extended (e.g. LEARN-RegNet added): train only the missing runs.
# --plan: only prepare + measure speed + print the GPU-hour estimate, then stop (see scripts/vast/status.sh); no training.
# --then VARIANT: when this set finishes, start that variant automatically, e.g. --then deeplesion (see docs/VAST.md).
# --auto-stop: Stop the machine when everything is done (data stays, GPU billing ends). Can be enabled while running:
#   touch .runtime/AUTO_STOP
# Re-run after a failure/restart: finished steps and runs are skipped, unfinished runs resume from their checkpoint.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if tmux -S "$TMUX_SOCK" has-session -t "$SESSION" 2>/dev/null; then
    echo "Already running (tmux '$SESSION'). See: bash scripts/vast/status.sh"
    exit 0
fi
# Two datasets at once would compete for GPUs (each pipeline schedules runs on EVERY GPU).
# LACT_CHAIN: set when full.sh --then chains the next variant.
others=$(tmux -S "$TMUX_SOCK" list-sessions -F '#S' 2>/dev/null | grep '^lact' | tr '\n' ' ' || true)
if [[ -n "$others" && -z "${LACT_CHAIN:-}" ]]; then
    echo "Another session is running: $others. Wait for it, or start the first set with --then <variant> to chain them." >&2
    exit 1
fi
args=""
(( $# )) && printf -v args '%q ' "$@"
tmux -S "$TMUX_SOCK" new-session -d -s "$SESSION" -c "$LACT_ROOT" \
    "VARIANT='$VARIANT' MODELS='${MODELS:-}' bash scripts/vast/full.sh $args >> '$LOG' 2>&1"
echo "Started. Log: $LOG"
echo "Progress: bash scripts/vast/status.sh   |   Graceful stop: bash scripts/vast/stop.sh"
