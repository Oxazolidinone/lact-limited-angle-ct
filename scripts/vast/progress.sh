#!/usr/bin/env bash
# Full progress overview on the rented machine (run from anywhere on it, including Vast's Jupyter terminal):
#   bash /workspace/kltn-lact/scripts/vast/progress.sh
#   watch -n 60 bash /workspace/kltn-lact/scripts/vast/progress.sh      # refresh every minute (Ctrl-C to quit)
# Shows: tmux sessions, which run each GPU is running, the 2DeteCT download, the current step of scripts/vast/detect.sh,
# running runs (progress, sec/step, hours left, best validation PSNR) and finished / partial / pending counts per queue.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh" >/dev/null 2>&1 || { echo "could not source scripts/vast/env.sh"; exit 2; }
JOINT=(configs/env/server.yaml configs/variant/joint.yaml)
DETECT=(configs/env/server.yaml configs/variant/detect.yaml)
OLD=(cmp_mayo cmp_dl dpma1_mayo dpma1_dl dpma_mayo dpma_dl)
NEW=(lbfgs_dt base_dt cmp_dt dpma1_dt dpma_dt)
specs() { local e; for e in "$@"; do [[ -f "configs/experiment/$e.yaml" ]] && printf 'configs/experiment/%s.yaml ' "$e"; done; }

echo "=== $(date -u '+%F %T') UTC  ($(TZ=Asia/Ho_Chi_Minh date '+%H:%M') Vietnam time)"
echo "tmux: $(tmux -S "$TMUX_SOCK" ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"

echo; echo "--- GPU (which runs are on it)"
"$PY" - <<'PY'
import re, subprocess
q = lambda a: subprocess.run(["nvidia-smi", f"--query-{a}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
gpus = [l.split(", ") for l in q("gpu=index,uuid,utilization.gpu,memory.used,memory.total").strip().splitlines()]
apps = [l.split(", ") for l in q("compute-apps=gpu_uuid,pid,used_memory").strip().splitlines() if l]
def what(pid):
    try:
        a = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
    except OSError:
        return f"pid {pid}"
    exp = re.search(r"exp=(\S+)", a); geo = re.search(r"geometry/(\w+)\.yaml", a); mod = re.search(r"model/(\w+)\.yaml", a)
    if exp:
        return f"{exp.group(1)} {mod.group(1) if mod else ''} {geo.group(1) if geo else ''}".strip()
    return "evaluation/other: " + " ".join(a.split()[3:6])
for idx, uuid, util, used, tot in gpus:
    runs = [what(p) for u, p, _ in apps if u == uuid]
    print(f"GPU {idx}: {util:>3}% {int(used)/1024:5.1f}/{int(tot)/1024:.0f} GB  " + (" | ".join(runs) if runs else "IDLE"))
PY

echo; echo "--- 2DeteCT download"
n=$(ls data_raw/2detect/*.npz 2>/dev/null | wc -l)
echo "$n / 2670 slices downloaded$(tmux -S "$TMUX_SOCK" has-session -t detect_dl 2>/dev/null && echo ' (downloading)')"
[[ -f .runtime/logs/detect_download.log ]] && tail -n 1 .runtime/logs/detect_download.log | cut -c1-140

echo; echo "--- scripts/vast/detect.sh (2DeteCT)"
if [[ -f .runtime/DETECT_DONE ]]; then echo "DONE at $(cat .runtime/DETECT_DONE)"; fi
grep -E "^== |STOPPED|slice [0-9]+:|opening grid|DONE|mu_ref" .runtime/logs/detect.log 2>/dev/null | tail -n 5 | cut -c1-140

echo; echo "--- running runs (progress, sec/step, hours left, best validation PSNR)"
for v in JOINT DETECT; do
    declare -n C=$v
    # `lact train status` columns: exp run_id status epoch step total_steps progress_% sec_per_step eta_h best_val_psnr
    lact train status -c "${C[@]}" 2>/dev/null | awk '$3 == "running" || $3 == "partial" {
        printf "%-48s %-8s ep %-3s %5.1f%%  %5.3f s/step  left %5.1f h  best %.2f\n", $2, $3, $4, $7, $8, $9, $10}'
done

echo; echo "--- queues (number of runs)"
count() { lact train grid $(specs "$@") -c "${CFG_A[@]}" "${MATCH[@]}" 2>/dev/null \
    | awk '/^\[/ {s[$2]++} END {for (k in s) printf "%s %d  ", k, s[k]; print ""}'; }
CFG_A=("${JOINT[@]}"); MATCH=(--match _fan_a90_ _fan_a120_); echo "Mayo/DeepLesion (comparison): $(count "${OLD[@]}")"
CFG_A=("${DETECT[@]}"); MATCH=(); echo "2DeteCT:                      $(count "${NEW[@]}")"
echo; df -h "$LACT_ROOT" | tail -n 1 | awk '{print "disk: used " $3 " / " $2 ", free " $4}'
