#!/usr/bin/env bash
# Progress on the rented machine: bash scripts/vast/status.sh        (variant: VARIANT=<name> bash scripts/vast/status.sh)
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if tmux -S "$TMUX_SOCK" has-session -t "$SESSION" 2>/dev/null; then echo "Status: RUNNING"; else echo "Status: not running"; fi
[[ -f "$DONE_FILE" ]] && echo "DONE at $(cat "$DONE_FILE")"
[[ -f "$FAILED_FILE" ]] && echo "FAILED: $(cat "$FAILED_FILE")"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
"$PY" - "${CFGS[@]}" <<'PY'
import json
import sys
from lact.config import load_config
p = load_config(sys.argv[1:]).paths.outputs / "pipeline" / "main.json"
if p.exists():
    s = json.loads(p.read_text())
    st = s.get("stages", {})
    print("Pipeline:", s.get("status"), "| stages:", ", ".join(f"{k}={v.get('status')}" for k, v in st.items()))
    jobs = s.get("jobs", {})
    if jobs:
        from collections import Counter
        by = Counter((k.split("_")[0], v.get("status")) for k, v in jobs.items())
        print("Runs:", ", ".join(f"{e} {st_}: {n}" for (e, st_), n in sorted(by.items())))
PY
lact train status -c "${CFGS[@]}" 2>/dev/null | grep -v "^\s*$" | grep -E "running|partial|exp " | head -20
[[ -f ".runtime/pilot/plan$TAG.txt" ]] && { echo "--- estimate at start (pilot)"; grep -E "TOTAL|With" ".runtime/pilot/plan$TAG.txt"; }
df -h "$LACT_ROOT" | tail -1
echo "--- last log lines"; tail -n 4 "$LOG" 2>/dev/null
