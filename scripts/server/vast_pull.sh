#!/usr/bin/env bash
# Run ON THE SERVER once the Vast machine has finished training: pull the E3–E5 runs into outputs/runs/.
#   bash scripts/server/vast_pull.sh <IP> <PORT>
# Then re-run the pipeline on the server (see docs/VAST.md): it skips the DONE runs, freezes the configuration, then
# runs the test evaluation, E6 and cost, and exports tables/figures.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: bash scripts/server/vast_pull.sh <IP> <PORT>" >&2; exit 2; }
host="root@$1"
port="$2"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$root"
key="${VAST_KEY:-$HOME/.ssh/vast}"
rsync_ssh="ssh -i $key -p $port -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o LogLevel=ERROR"

if tmux -S "$root/.runtime/tmux.sock" has-session -t lact 2>/dev/null; then
    echo "The pipeline is running on the server: stop it before pulling." >&2
    exit 1
fi
mkdir -p outputs/runs .runtime/logs
for e in e3_main e4_matched e5_a150; do
    echo "== $e"
    rsync -a --partial --info=progress2 -e "$rsync_ssh" "$host:$root/outputs/runs/$e" outputs/runs/
done
rsync -a -e "$rsync_ssh" "$host:$root/.runtime/logs/train.log" .runtime/logs/vast_train.log || true
rsync -a -e "$rsync_ssh" "$host:$root/.runtime/pilot" .runtime/vast_pilot || true
n_done=$(find outputs/runs/e3_main outputs/runs/e4_matched outputs/runs/e5_a150 -maxdepth 2 -name DONE 2>/dev/null | wc -l)
echo "$n_done runs DONE on the server (all 15 models with one seed make 48; plus any old seed 1, 2 runs)."
