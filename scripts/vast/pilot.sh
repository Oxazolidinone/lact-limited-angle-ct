#!/usr/bin/env bash
# Measure the real speed on the rented GPU (~15 minutes): bash scripts/vast/pilot.sh   (variant: VARIANT=<name> bash ...)
#  1) 1 process on GPU 0: sec/step and peak memory of the 18 models used in E2–E5
#  2) k = 2, 3 processes sharing GPU 0 with the two slowest models and the most memory-hungry one: how much throughput
#     improves. Models without enough memory to share k runs do not count towards the choice: the pipeline runs them
#     alone (lact pipeline run --mem-table) and the remaining models share.
#  3) estimate the GPU-hours still to train (`lact train plan`) from the unfinished runs of the grid
# Writes: .runtime/pilot/per_gpu.txt (runs/GPU, read by scripts/vast/full.sh), .runtime/pilot/plan<-variant>.txt.
# Does not touch outputs/.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
out="$LACT_ROOT/.runtime/pilot"
mkdir -p "$out"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
ALL=(learn_cnn learn_unet learn_unet_regnet learn_resnet learn_resnet_regnet learn_resnext learn_regnet
     learn_densenet learn_densenet_regnet learn_efficientnet learn_efficientnet_regnet learn_convnext
     learn_convnext_regnet learn_vit learn_vit_regnet learn_resnet_pm2m learn_regnet_pm2m learn_vit_pm2m)
STEPS=120
csv=pilot_fan_a120.csv
n_gpu=$(nvidia-smi -L | wc -l)
cores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
gpu_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')

echo "== 1 process on GPU 0 (${#ALL[@]} models)"
CUDA_VISIBLE_DEVICES=0 lact train pilot -c "${CFGS[@]}" -s "paths.tables=$out/k1" \
    --models "${ALL[@]}" --steps "$STEPS" 2>&1 | grep -v Warning | tail -n 20

# Conservative measure for GPU sharing: the two slowest models and the most memory-hungry one.
read -r -a HEAVY <<< "$("$PY" - "$out/k1/$csv" <<'EOF'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1])
pick = list(d.sort_values("sec_per_step", ascending=False).model[:2]) + [d.loc[d.peak_mem_mb.idxmax(), "model"]]
print(" ".join(dict.fromkeys(pick)))
EOF
)"
echo "Models used to measure GPU sharing: ${HEAVY[*]}"

for k in 2 3; do
    echo "== $k processes sharing GPU 0"
    for i in $(seq 1 "$k"); do
        CUDA_VISIBLE_DEVICES=0 lact train pilot -c "${CFGS[@]}" -s "paths.tables=$out/k${k}_$i" \
            --models "${HEAVY[@]}" --steps "$STEPS" > "$out/k${k}_$i.log" 2>&1 &
    done
    wait || echo "(a measuring process with k=$k failed, usually out of memory: skipping this level)"
done

"$PY" - "$out" "$cores" "$n_gpu" "$gpu_mb" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
out, cores, n_gpu, gpu_mb = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
csv = "pilot_fan_a120.csv"
k1 = pd.read_csv(out / "k1" / csv).set_index("model")
print(k1[["sec_per_step", "peak_mem_mb", "gpu_hours_per_run", "n_params"]].round(3).to_string())
budget = 0.85 * gpu_mb
best_k, best_gain = 1, 1.0
for k in (2, 3):
    files = [out / f"k{k}_{i}" / csv for i in range(1, k + 1)]
    if not all(f.is_file() for f in files):
        print(f"k={k}: could not measure (see {out}/k{k}_*.log)")
        continue
    tk = pd.concat([pd.read_csv(f) for f in files]).groupby("model")["sec_per_step"].mean()
    gains = {m: k * k1.loc[m, "sec_per_step"] / tk[m] for m in tk.index}     # throughput relative to 1 process
    fit = {m: g for m, g in gains.items() if k * k1.loc[m, "peak_mem_mb"] <= budget}
    alone = sorted(set(gains) - set(fit))
    cpu_ok = k * n_gpu <= cores
    print(f"k={k}: " + ", ".join(f"{m} x{g:.2f}" for m, g in gains.items()) + f"; CPU {'enough' if cpu_ok else 'SHORT'}"
          + (f"; not enough memory to share (will run alone): {alone}" if alone else ""))
    # raise k only when every model that can share gains a further >= 10% over the best level so far
    if fit and cpu_ok and min(fit.values()) >= best_gain * 1.10:
        best_k, best_gain = k, min(fit.values())
n_alone = int((best_k * k1.peak_mem_mb > budget).sum()) if best_k > 1 else 0
(out / "per_gpu.txt").write_text(f"{best_k}\n")
(out / "gain.txt").write_text(f"{best_gain:.3f}\n")
print(f"Chose --per-gpu {best_k} (throughput x{best_gain:.2f} for the shared models"
      + (f"; {n_alone} heavy models run alone" if n_alone else "") + "). The pilot disables deterministic mode, so "
      "real training is slower (batch of 05/10: about 1.4x).")
EOF

echo "== estimate of the remaining training"
lact train plan -c "${CFGS[@]}" --pilot "$out/k1/$csv" --n-gpus "$n_gpu" --gain "$(cat "$out/gain.txt")" \
    ${MODELS:+--models $MODELS} 2>&1 | grep -v "Warning\|does not exist yet" | tee "$out/plan$TAG.txt"
