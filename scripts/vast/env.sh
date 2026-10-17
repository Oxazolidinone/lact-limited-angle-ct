# shellcheck shell=bash
# Shared variables for the scripts on the rented Vast machine. Do not run directly: `source scripts/vast/env.sh`.
#
# Every command runs with configs/env/server.yaml (paths relative to the project directory), so the project can live
# anywhere (e.g. /workspace/kltn-lact) and runs copied from another machine are still picked up (see docs/VAST.md).
# Only when syncing both ways with the server via scripts/server/vast_push.sh must the path match the server's.

LACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
export LACT_ROOT
cd "$LACT_ROOT" || return 2

# scripts/server/vast_push.sh writes the project's path on the server here.
if [[ -f .runtime/server_root.txt ]]; then
    IFS= read -r LACT_SERVER_ROOT < .runtime/server_root.txt
    if [[ "$LACT_ROOT" != "$LACT_SERVER_ROOT" ]]; then
        echo "The project is at $LACT_ROOT but at $LACT_SERVER_ROOT on the server: they must match." >&2
        return 2 2>/dev/null || exit 2
    fi
fi

# Python of the Vast PyTorch template lives in /venv/main; elsewhere use python3 from PATH.
if [[ -x /venv/main/bin/python ]]; then
    PY=/venv/main/bin/python
else
    PY="$(command -v python3)"
fi
export PY

export TMPDIR="$LACT_ROOT/.runtime/tmp"
export XDG_CACHE_HOME="$LACT_ROOT/.runtime/cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$LACT_ROOT/.runtime/logs"
# The array cache (cache.arrays) keeps one file open per array per patient. Datasets with thousands of patients
# (DeepLesion: 1,118 training patients) exceed the default limit of 1024 files and the run dies while loading data
# ("Too many open files", 10/10/2026): raise the soft limit, which every child process inherits.
ulimit -n 65536 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

CFG=configs/env/server.yaml
# Data variant (configs/variant/<name>.yaml), e.g. VARIANT=mayo_poisson_1e6 bash scripts/vast/start.sh
# Each variant has its own outputs, tmux session, log and marker files; leave VARIANT unset for the noise-free Mayo set.
VARIANT="${VARIANT:-}"
CFGS=("$CFG")
TAG=""
if [[ -n "$VARIANT" ]]; then
    CFGS+=("configs/variant/$VARIANT.yaml")
    TAG="-$VARIANT"
fi
TMUX_SOCK="$LACT_ROOT/.runtime/tmux.sock"
SESSION="lact$TAG"
LOG="$LACT_ROOT/.runtime/logs/train$TAG.log"
DONE_FILE="$LACT_ROOT/.runtime/TRAIN_DONE$TAG"
FAILED_FILE="$LACT_ROOT/.runtime/TRAIN_FAILED$TAG"

lact() { "$PY" -m lact "$@"; }
now() { date -u '+%F %T UTC'; }
