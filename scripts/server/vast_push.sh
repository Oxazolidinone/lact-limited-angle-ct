#!/usr/bin/env bash
# Run ON THE SERVER once E2 has finished and the pipeline has stopped: push the code + data E3–E5 need to the rented
# Vast machine.
#   bash scripts/server/vast_push.sh <IP> <PORT>          (IP/PORT of "Direct ssh connect" on the Vast instance card)
# Optional variables: VAST_KEY (default ~/.ssh/vast).
# The Vast machine receives the project at the SAME absolute path as on the server, so per-run configurations match
# when pulled back.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: bash scripts/server/vast_push.sh <IP> <PORT>" >&2; exit 2; }
host="root@$1"
port="$2"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$root"
key="${VAST_KEY:-$HOME/.ssh/vast}"
ssh_cmd=(ssh -i "$key" -p "$port" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o LogLevel=ERROR)

[[ -f outputs/runs/e2_sweep/best_lr.json ]] || { echo "No best_lr.json yet: E2 has not finished." >&2; exit 1; }
if tmux -S "$root/.runtime/tmux.sock" has-session -t lact 2>/dev/null; then
    echo "The pipeline is running on the server: stop it before pushing (files may be changing)." >&2
    exit 1
fi

files=(outputs/data/images.h5 outputs/data/stats.json outputs/data/slices.json outputs/data/manifest.json)
for g in fan_a90 fan_a120 fan_a150; do
    files+=("outputs/data/sino_$g.h5" "outputs/operators/A_${g}_n256.npz" "outputs/operators/A_${g}_n256.json")
done
for f in "${files[@]}"; do [[ -f "$f" ]] || { echo "Missing $f" >&2; exit 1; }; done
du -ch "${files[@]}" | tail -1

"${ssh_cmd[@]}" "$host" "mkdir -p '$root/.runtime' && echo '$root' > '$root/.runtime/server_root.txt'"
rsync_ssh="${ssh_cmd[*]}"
echo "== source code"
rsync -a --partial -e "$rsync_ssh" \
    --exclude='.git/' --exclude='.venv/' --exclude='.runtime/' --exclude='cache/' --exclude='outputs/' \
    --exclude='outputs_smoke/' --exclude='data_raw/' --exclude='dist/' --exclude='__pycache__/' \
    --exclude='.mypy_cache/' --exclude='.ruff_cache/' --exclude='.pytest_cache/' --exclude='.import_linter_cache/' \
    --exclude='*.egg-info/' --exclude='*.swp' ./ "$host:$root/"
echo "== data, matrix A"
rsync -a --partial --info=progress2 -R -e "$rsync_ssh" "${files[@]}" "$host:$root/"
echo "== E2 results (without checkpoints)"
rsync -a --partial -R -e "$rsync_ssh" --exclude='*.pt' --exclude='tb/' outputs/runs/e2_sweep "$host:$root/"
for e in e3_main e4_matched e5_a150; do
    if [[ -d "outputs/runs/$e" ]]; then
        echo "== $e has unfinished runs on the server: pushing them too so the rented machine resumes them"
        rsync -a --partial -R -e "$rsync_ssh" "outputs/runs/$e" "$host:$root/"
    fi
done
echo "Done. On the Vast machine: cd $root && bash scripts/vast/setup.sh"
