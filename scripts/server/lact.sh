#!/usr/bin/env bash
# Usage: bash scripts/server/lact.sh pipeline status -c configs/env/server.yaml
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$project_root"
export LACT_ROOT="$project_root"
export TMPDIR="$project_root/.runtime/tmp"
export XDG_CACHE_HOME="$project_root/.runtime/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLBACKEND=Agg
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true   # see scripts/vast/env.sh

# ASTRA needs to find the CUDA runtime that pip installed into .venv; no system CUDA is required.
site_packages="$("$project_root/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cuda_libraries=()
for library_dir in "$site_packages"/nvidia/*/lib; do
    [[ ! -d "$library_dir" ]] || cuda_libraries+=("$library_dir")
done
export LD_LIBRARY_PATH="$(IFS=:; echo "${cuda_libraries[*]-}")"

# Do not re-pick GPUs when resuming a session. This file holds the UUIDs checked on the server.
# An empty value passed explicitly from outside still means CPU only.
if [[ ! ${CUDA_VISIBLE_DEVICES+x} ]]; then
    if [[ ! -f .runtime/gpus.txt ]]; then
        echo 'Missing .runtime/gpus.txt: write the UUIDs of the GPUs allowed for use, separated by commas.' >&2
        exit 2
    fi
    IFS= read -r CUDA_VISIBLE_DEVICES < .runtime/gpus.txt
    export CUDA_VISIBLE_DEVICES
fi
exec "$project_root/.venv/bin/python" -m lact "$@"
