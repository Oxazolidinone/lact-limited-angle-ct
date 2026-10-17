#!/usr/bin/env bash
# Run ON THE PERSONAL MACHINE (Mac/Linux): prepare the Vast machine for phase 2 (docs/PHASE2.md).
#   bash scripts/vast/mac_push_phase2.sh <IP> <PORT> <MAYO_OUTPUTS_DIR>
# 1) push the source code (as scripts/vast/mac_push.sh);
# 2) push the E3 runs trained on Mayo for ResNet, U-Net, DenseNet x {90°, 120°} from MAYO_OUTPUTS_DIR
#    (e.g. ../ket-qua/vast_results_2026-10-07/outputs) into outputs_joint/runs/e3_main on the rented machine. Phase 2
#    does NOT retrain these three backbones on Mayo: the E3 runs are both the first-order baseline and the checkpoint
#    the lbfgs_mayo runs used to load. When a configuration has several runs, the one with the highest VALIDATION PSNR
#    is taken. Only config.yaml, summary.json, best.pt, the DONE marker and logs are pushed (~60 MB). Without the third
#    argument only the source code is pushed (when the E3 runs are already on the rented machine).
# Optional variables: VAST_KEY, VAST_ROOT, VAST_USER (as in mac_push.sh); ARCHS (model.arch list, default the three
# backbones above).
set -eo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/vast/mac_push_phase2.sh <IP> <PORT> [MAYO_OUTPUTS_DIR]" >&2
    exit 2
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/remote.sh"
vast_remote "$1" "$2"
old="${3:-}"
ARCHS="${ARCHS:-learn_unet learn_resnet learn_densenet}"
cd "$root"
bash scripts/vast/mac_push.sh "$1" "$2"

if [[ -n "$old" ]]; then
    old="$(cd "$old" && pwd -P)"
    [[ -d "$old/runs/e3_main" ]] || { echo "Missing $old/runs/e3_main: this is not the outputs of the Mayo batch." >&2; exit 1; }
    list="$(mktemp)"
    # shellcheck disable=SC2086
    python3 - "$old/runs/e3_main" $ARCHS > "$list" <<'PY'
import json
import sys
from pathlib import Path

root, archs = Path(sys.argv[1]), set(sys.argv[2:])
best = {}
for d in sorted(root.iterdir()):
    f = d / "summary.json"
    if not (f.is_file() and (d / "DONE").is_file() and (d / "best.pt").is_file()):
        continue
    s = json.loads(f.read_text())
    key = (s.get("arch"), s.get("geometry"))
    if s.get("arch") in archs and s.get("geometry") in ("fan_a90", "fan_a120") and s.get("best_val_psnr") is not None:
        if key not in best or (s["best_val_psnr"], -int(s.get("seed", 0))) > best[key][0]:
            best[key] = ((s["best_val_psnr"], -int(s.get("seed", 0))), d.name)
for (arch, geom), (score, name) in sorted(best.items()):
    print(name)
    print(f"  {arch:16s} {geom}: {name} (validation PSNR {score[0]:.2f})", file=sys.stderr)
PY
    n=$(wc -l < "$list" | tr -d ' ')
    [[ "$n" -gt 0 ]] || { echo "No E3 run found for: $ARCHS" >&2; rm -f "$list"; exit 1; }
    echo "== $n E3 runs (Mayo) -> $host:$remote/outputs_joint/runs/e3_main"
    # shellcheck disable=SC2086,SC2029
    ssh $ssh_opts "$host" "mkdir -p '$remote/outputs_joint/runs/e3_main'"
    while IFS= read -r name; do
        rsync -a --ignore-existing -e "ssh $ssh_opts" --exclude='last.pt' --exclude='last_prev.pt' --exclude='tb/' \
            --exclude='*.tmp' --exclude='.train.lock' "$old/runs/e3_main/$name" "$host:$remote/outputs_joint/runs/e3_main/"
    done < "$list"
    rm -f "$list"
fi
cat <<MSG
Done. On the Vast machine:
  ssh -p $port $host
  cd $remote && bash scripts/vast/setup.sh
  bash scripts/vast/phase2.sh start --plan     # prepare + measure speed + print the GPU-hour estimate, NO training yet
  VARIANT=joint bash scripts/vast/status.sh    # view the estimate / progress
(details: docs/PHASE2.md)
MSG
