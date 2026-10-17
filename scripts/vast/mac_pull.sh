#!/usr/bin/env bash
# Run ON THE PERSONAL MACHINE (Mac/Linux): pull the results back from the Vast machine.
#   bash scripts/vast/mac_pull.sh <IP> <PORT> [DEST] [--full]
# DEST defaults to ../ket-qua/vast_results_<today> (the results folder next to the project). Re-runnable: only new or
# changed files are pulled.
# By default pulls the COMPACT copy of every outputs* set (tables, figures, figdata, runs with best.pt, per-slice
# metrics, logs): enough to write the thesis, redraw figures (experiments/figures/make_figures.py) and move the runs to
# another machine later.
# --full: also pull the predicted images (eval/preds, tens of GB) and images.h5.
# DONE runs come without last.pt; UNFINISHED runs include last.pt so they can resume on another machine (stop the
# machine with scripts/vast/stop.sh before pulling). Never pulled: sinograms, matrix A, caches (all rebuildable).
# Optional variables: VAST_KEY, VAST_ROOT (as in mac_push.sh).
set -eo pipefail
full=0
args=()
for a in "$@"; do
    if [[ "$a" == "--full" ]]; then full=1; else args+=("$a"); fi
done
if [[ ${#args[@]} -lt 2 || ${#args[@]} -gt 3 ]]; then
    echo "Usage: bash scripts/vast/mac_pull.sh <IP> <PORT> [DEST] [--full]" >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/remote.sh"
vast_remote "${args[0]}" "${args[1]}"
dest="${args[2]:-$root/../ket-qua/vast_results_$(date +%F)}"
mkdir -p "$dest"
dest="$(cd "$dest" && pwd -P)"

skip=(--exclude='*.tmp' --exclude='tb/' --exclude='.train.lock' --exclude='sino_*.h5' --exclude='/operators/')
if [[ "$full" == "0" ]]; then
    skip+=(--exclude='/eval/preds/' --exclude='/data/images.h5')
fi
# shellcheck disable=SC2086,SC2029
sets=$(ssh $ssh_opts "$host" "cd '$remote' && ls -d outputs*/ 2>/dev/null | tr -d '/'")
[[ -n "$sets" ]] || { echo "No outputs* directory found at $host:$remote" >&2; exit 1; }
for name in $sets; do
    echo "== $name -> $dest/$name"
    mkdir -p "$dest/$name"
    done_skip="$(mktemp)"       # last.pt of the DONE runs
    # shellcheck disable=SC2086,SC2029
    ssh $ssh_opts "$host" "cd '$remote/$name/runs' 2>/dev/null && find . -maxdepth 3 -name DONE" \
        | sed -e 's|^\./||' -e 's|/DONE$||' \
        | awk '{print "/runs/" $0 "/last.pt"; print "/runs/" $0 "/last_prev.pt"}' > "$done_skip" || true
    rsync -a --partial -e "ssh $ssh_opts" "${skip[@]}" --exclude-from="$done_skip" "$host:$remote/$name/" "$dest/$name/"
    rm -f "$done_skip"
done
echo "== logs, pilot measurements, split files"
mkdir -p "$dest/runtime" "$dest/configs_data"
rsync -a -e "ssh $ssh_opts" --exclude='tmp/' --exclude='cache/' --exclude='tmux.sock' "$host:$remote/.runtime/" "$dest/runtime/" || true
rsync -a -e "ssh $ssh_opts" "$host:$remote/configs/data/" "$dest/configs_data/" || true
rsync -a -e "ssh $ssh_opts" "$host:$remote/CODE_VERSION" "$dest/" || true
for name in $sets; do
    n=$(find "$dest/$name/runs" -maxdepth 3 -name DONE 2>/dev/null | wc -l | tr -d ' ')
    echo "$name: $n runs DONE, $(du -sh "$dest/$name" | cut -f1)"
done
echo "Pulled into $dest$([[ "$full" == "0" ]] && echo " (compact copy; add --full for the predicted images)")"
