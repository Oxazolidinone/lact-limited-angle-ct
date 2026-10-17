#!/usr/bin/env bash
# Install the environment on the rented machine (Vast PyTorch template, keeping its PyTorch): bash scripts/vast/setup.sh
# TORCH_PIN=2.11.0 [TORCH_CUDA=cu128] bash scripts/vast/setup.sh : install that PyTorch version first (the one the E3 runs
# were trained with; newer Vast templates ship another version with different cuDNN/cuSPARSE). New runs and the E3
# baseline then use the same libraries.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# The ASTRA version that built matrix A for the 02/10/2026 runs: keep it so that rebuilt matrices match exactly.
# (Other version: ASTRA_PIN=astra-toolbox bash scripts/vast/setup.sh; "check existing runs" then flags any mismatch.)
ASTRA_PIN="${ASTRA_PIN:-astra-toolbox==2.5.0}"
install() {
    if command -v uv >/dev/null; then
        uv pip install --python "$PY" -q "$@"
    else
        "$PY" -m pip install -q "$@"
    fi
}
if [[ -n "${TORCH_PIN:-}" ]]; then
    install "torch==$TORCH_PIN" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA:-cu128}"
fi
install -e . "$ASTRA_PIN" || {
    echo "Could not install $ASTRA_PIN on this machine: using the latest ASTRA." >&2
    install -e . astra-toolbox
}
for tool in tmux rsync; do
    command -v "$tool" >/dev/null || { apt-get update -qq && apt-get install -y -qq "$tool"; } \
        || echo "$tool is missing and could not be installed: apt-get install -y $tool" >&2
done
"$PY" -c "import torch, astra; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, '| ASTRA', astra.__version__)"
"$PY" scripts/vast/check_env.py --gpu-only
echo "Next: bash scripts/vast/start.sh --plan   (see docs/VAST.md)"
