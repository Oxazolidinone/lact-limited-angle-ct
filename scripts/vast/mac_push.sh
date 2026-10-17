#!/usr/bin/env bash
# Run ON THE PERSONAL MACHINE (Mac/Linux): push the source code, plus the runs trained in an earlier batch if given, to
# the Vast machine.
#   bash scripts/vast/mac_push.sh <IP> <PORT> [OLD_OUTPUTS_DIR] [--with-images]
# IP/PORT: the "Direct ssh connect" entry on the Vast instance card (form: ssh -p PORT root@IP).
# OLD_OUTPUTS_DIR (optional): the outputs of an earlier batch on the Mayo set, e.g.
#   ../ket-qua/vast_results_2026-10-07/outputs
# When given, the script also pushes outputs/runs (every trained run; DONE runs without last.pt since only best.pt is
# needed) so that the new machine does NOT retrain those runs. Preprocessed images (images.h5, 1.5 GB) are NOT pushed by
# default: the rented machine downloads the raw data and rebuilds them in a few minutes, then the "check existing runs"
# step of scripts/vast/full.sh confirms the rebuilt data matches training. --with-images: also push
# outputs/data/{images.h5,stats.json,slices.json}.
# Files already on the rented machine are never overwritten, so re-running it midway (e.g. to update the code) is safe.
# Optional variables: VAST_KEY (own SSH key), VAST_ROOT (project directory on the rented machine, default
# /workspace/kltn-lact), VAST_USER (login account, default root).
set -eo pipefail
with_images=0
args=()
for a in "$@"; do
    if [[ "$a" == "--with-images" ]]; then with_images=1; else args+=("$a"); fi
done
if [[ ${#args[@]} -lt 2 || ${#args[@]} -gt 3 ]]; then
    echo "Usage: bash scripts/vast/mac_push.sh <IP> <PORT> [OLD_OUTPUTS_DIR] [--with-images]" >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/remote.sh"
vast_remote "${args[0]}" "${args[1]}"
old="${args[2]:-}"
cd "$root"

if [[ -n "$old" ]]; then
    old="$(cd "$old" && pwd -P)"
    need="runs"
    [[ "$with_images" == "1" ]] && need="runs data/images.h5 data/stats.json data/slices.json"
    for f in $need; do
        [[ -e "$old/$f" ]] || { echo "Missing $old/$f: this is not the outputs directory of an earlier batch." >&2; exit 1; }
    done
    n_done=$(find "$old/runs" -maxdepth 3 -name DONE | wc -l | tr -d ' ')
    echo "Old outputs: $old ($n_done runs DONE)"
fi
[[ -f configs/data/split_deeplesion.yaml ]] || echo "NOTE: configs/data/split_deeplesion.yaml does not exist yet (the rented machine will split DeepLesion itself)."

echo "== checking the connection and rsync on the rented machine"
# shellcheck disable=SC2086,SC2029
ssh $ssh_opts "$host" "mkdir -p '$remote/outputs/data' && { command -v rsync >/dev/null || { apt-get update -qq && apt-get install -y -qq rsync; }; }"

echo "== source code -> $host:$remote"
python3 scripts/pack_source.py >/dev/null      # writes CODE_VERSION: every training run records the code version it used
# --delete: code/config files removed on this machine are also removed on the rented machine (stale configurations
# would change the experiment grids). The --exclude entries (results, data, caches, environments) are never touched.
keep_split=()
[[ -f configs/data/split_deeplesion.yaml ]] || keep_split=(--exclude='/configs/data/split_deeplesion.yaml')
rsync -a --partial --delete -e "ssh $ssh_opts" \
    --exclude='.git/' --exclude='.venv/' --exclude='/.runtime/' --exclude='/cache/' --exclude='/outputs*/' \
    --exclude='/.regression_cache/' \
    --exclude='/data_raw/' --exclude='/dist/' --exclude='__pycache__/' --exclude='.mypy_cache/' \
    --exclude='.ruff_cache/' --exclude='.pytest_cache/' --exclude='.import_linter_cache/' --exclude='*.egg-info/' \
    --exclude='.DS_Store' --exclude='/configs/local.yaml' ${keep_split[@]+"${keep_split[@]}"} ./ "$host:$remote/"

if [[ -n "$old" ]]; then
    echo "== trained runs (files already on the rented machine are not overwritten)"
    # last.pt is only needed to resume unfinished runs: skipped for runs with a DONE marker (~3/4 less data to push)
    skip_list="$(mktemp)"
    (cd "$old/runs" && find . -maxdepth 3 -name DONE | sed -e 's|^\./||' -e 's|/DONE$||' \
        | awk '{print "/runs/" $0 "/last.pt"; print "/runs/" $0 "/last_prev.pt"}') > "$skip_list"
    rsync -a --ignore-existing -e "ssh $ssh_opts" --exclude-from="$skip_list" --exclude='tb/' --exclude='*.tmp' \
        --exclude='.train.lock' "$old/runs" "$host:$remote/outputs/"
    rm -f "$skip_list"
    if [[ "$with_images" == "1" ]]; then
        echo "== preprocessed images (1.5 GB)"
        rsync -a --ignore-existing --progress -e "ssh $ssh_opts" \
            "$old/data/images.h5" "$old/data/stats.json" "$old/data/slices.json" "$host:$remote/outputs/data/"
    fi
    # shellcheck disable=SC2086,SC2029
    ssh $ssh_opts "$host" "echo \"On the rented machine: \$(find '$remote/outputs/runs' -maxdepth 3 -name DONE | wc -l) runs DONE\""
fi
cat <<MSG
Done. On the Vast machine:
  ssh -p $port $host
  cd $remote && bash scripts/vast/setup.sh
  bash scripts/vast/start.sh --plan      # prepare + measure speed + print the GPU-hour estimate, NO training yet
(details: docs/VAST.md)
MSG
