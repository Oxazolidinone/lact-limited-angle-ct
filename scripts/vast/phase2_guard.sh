#!/usr/bin/env bash
# Safety net for `phase2.sh start --auto-stop` when nobody watches the machine (overnight runs).
# phase2.sh stops BEFORE evaluation if a run diverged / failed, and then does not Stop the machine: it idles at a cost.
# This script waits for the queue to end; if it ends without the TRAIN_DONE marker file (and was not stopped by hand):
#   1) re-run `phase2.sh start --auto-stop` ONCE: DIVERGED/OOM runs are skipped, the remaining runs are evaluated,
#      tables written and figures collected, then the machine Stops itself;
#   2) if the second attempt still does not finish, Stop the machine (Stop, not Destroy: the data stays on disk).
#
#   tmux new-session -d -s guard "bash scripts/vast/phase2_guard.sh [phase2.sh options] > .runtime/logs/phase2_guard.log 2>&1"
# Options (e.g. --compare-only) are passed unchanged to the re-run so it runs the same group of runs as the first time.
export VARIANT=joint
# shellcheck source=scripts/vast/env.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh" || exit 2

alive() { tmux -S "$TMUX_SOCK" has-session -t "$SESSION" 2>/dev/null; }
wait_end() { while alive; do sleep 60; done; }
stopped_by_hand() { tail -n 5 "$LOG" 2>/dev/null | grep -q "Stopped midway"; }

echo "$(now) watching queue $SESSION"
wait_end
[[ -f "$DONE_FILE" ]] && { echo "$(now) queue finished normally"; exit 0; }
stopped_by_hand && { echo "$(now) queue was stopped by hand: doing nothing"; exit 0; }
echo "$(now) queue ended without $(basename "$DONE_FILE"): re-running once (diverged runs are skipped)"
tail -n 3 "$LOG"
bash scripts/vast/phase2.sh start --auto-stop "$@"
sleep 30
wait_end
[[ -f "$DONE_FILE" ]] && { echo "$(now) second attempt finished"; exit 0; }
stopped_by_hand && { echo "$(now) second attempt was stopped by hand: doing nothing"; exit 0; }
echo "$(now) second attempt still did not finish: Stopping the machine so it does not idle (data stays on disk)"
tail -n 3 "$LOG"
if [[ -n "${CONTAINER_ID:-}" ]]; then
    "$(dirname -- "$PY")/vastai" --api-key "${CONTAINER_API_KEY:-}" stop instance "$CONTAINER_ID" \
        || echo "$(now) could not Stop automatically: Stop it on the Vast website"
else
    echo "$(now) no CONTAINER_ID: Stop it on the Vast website"
fi
