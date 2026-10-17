#!/usr/bin/env bash
# Run ON THE PERSONAL MACHINE (Mac/Linux): prepare a NEW Vast machine for the CG-GLORE comparison group
# (docs/PHASE2.md, section 10).
#   bash scripts/vast/mac_push_compare.sh <IP> <PORT> <OLD_JOINT_OUTPUTS_DIR>
# OLD_JOINT_OUTPUTS_DIR: the outputs_joint pulled after phase 2, e.g. ../ket-qua/vast_results_2026-10-12/outputs_joint
# 1) push the source code (as scripts/vast/mac_push.sh);
# 2) push into outputs_joint on the rented machine what the new machine must NOT redo:
#      runs/{e3_main,base_*,lbfgs_*}  every finished run (config, summary, best.pt, DONE marker, log; no last.pt)
#      eval/                          the test evaluation of those runs (kept as is, not recomputed)
#      baselines/, configs_frozen.lock  the chosen TV parameters and the configuration lock
#    and data_raw/deeplesion_key (the extracted DeepLesion key slices, if present on this machine) to avoid downloading
#    220 GiB again; the rented machine downloads the Mayo set itself.
#    Then on the rented machine:  bash scripts/vast/phase2.sh start --compare-only --auto-stop
#    The machine downloads the raw data, rebuilds the joint store, CHECKS the old Mayo runs against the new store (stops
#    before using GPUs on a mismatch), trains only the 16 cmp_* runs, then evaluates and writes tables covering both the
#    old and the new runs.
# Files already on the rented machine are not overwritten. Optional variables: VAST_KEY, VAST_ROOT, VAST_USER (as in
# mac_push.sh).
set -eo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: bash scripts/vast/mac_push_compare.sh <IP> <PORT> <OLD_JOINT_OUTPUTS_DIR>" >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/remote.sh"
vast_remote "$1" "$2"
old="$(cd "$3" && pwd -P)"
for e in e3_main base_mayo base_dl lbfgs_mayo lbfgs_dl; do
    compgen -G "$old/runs/$e/*/DONE" >/dev/null || { echo "No finished run of $e in $old/runs: wrong directory?" >&2; exit 1; }
done
[[ -d "$old/eval" ]] || { echo "Missing $old/eval" >&2; exit 1; }
cd "$root"
bash scripts/vast/mac_push.sh "$1" "$2"

n=$(find "$old/runs" -maxdepth 3 -name DONE | wc -l | tr -d ' ')
echo "== $n finished runs + eval + baselines -> $host:$remote/outputs_joint"
# shellcheck disable=SC2086,SC2029
ssh $ssh_opts "$host" "mkdir -p '$remote/outputs_joint/runs'"
for e in e3_main base_mayo base_dl lbfgs_mayo lbfgs_dl; do
    rsync -a --ignore-existing -e "ssh $ssh_opts" --exclude='last.pt' --exclude='last_prev.pt' --exclude='tb/' \
        --exclude='*.tmp' --exclude='.train.lock' "$old/runs/$e" "$host:$remote/outputs_joint/runs/"
done
for d in eval baselines; do
    [[ -d "$old/$d" ]] && rsync -a --ignore-existing -e "ssh $ssh_opts" "$old/$d" "$host:$remote/outputs_joint/"
done
[[ -f "$old/configs_frozen.lock" ]] \
    && rsync -a --ignore-existing -e "ssh $ssh_opts" "$old/configs_frozen.lock" "$host:$remote/outputs_joint/"
# The extracted DeepLesion key slices (584 MB, with 56 .keyslices_XX.json marker files): if present, the rented machine
# need not download the 56 zip files (220 GiB) again to extract them; `lact data download-deeplesion --key-slices` skips
# any zip whose marker file exists.
if [[ -f data_raw/deeplesion_key/DL_info.csv ]]; then
    echo "== DeepLesion key slices -> $host:$remote/data_raw/deeplesion_key"
    # shellcheck disable=SC2086,SC2029
    ssh $ssh_opts "$host" "mkdir -p '$remote/data_raw'"
    rsync -a --ignore-existing -e "ssh $ssh_opts" --exclude='zips/*.zip' data_raw/deeplesion_key "$host:$remote/data_raw/"
else
    echo "(no data_raw/deeplesion_key on this machine: the rented machine will download and extract the key slices, which takes a few hours)"
fi
echo "Done. On the rented machine: cd $remote && bash scripts/vast/phase2.sh --compare-only --plan   (see the estimate), then"
echo "                             bash scripts/vast/phase2.sh start --compare-only --auto-stop"
