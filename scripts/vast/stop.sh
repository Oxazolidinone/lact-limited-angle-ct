#!/usr/bin/env bash
# Graceful stop: runs save a checkpoint and exit. Re-run scripts/vast/start.sh to continue.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if tmux -S "$TMUX_SOCK" has-session -t "$SESSION" 2>/dev/null; then
    tmux -S "$TMUX_SOCK" send-keys -t "$SESSION" C-c
    echo "Stop signal sent. Wait for the log to report the checkpoint save: tail -f $LOG"
else
    echo "No running session."
fi
