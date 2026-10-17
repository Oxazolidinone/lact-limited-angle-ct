"""Pick N idle GPUs on the host (reads nvidia-smi only; never initialises CUDA or occupies a GPU).

Runs with the host's Python (3.9 on the server), no project installation needed:

    python3 scripts/server/pick_gpus.py --n 2
    # prints the status table and export lines with the UUIDs (store them in .runtime/gpus.txt, see docs/SERVER.md)

By default only UNUSED GPUs qualify: used memory <= 1024 MiB, load <= 10% (maximum over 3 samples) and no compute
process. With too few GPUs it fails instead of taking busy ones. On a shared server a GPU that is idle at check time
is not necessarily free to use: confirm with the administrators if there is an allocation policy.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "lact" / "infra" / "system" / "gpus.py"


def _load_gpus():
    # Load the file directly instead of importing the lact package (which needs torch and Python >= 3.10).
    spec = importlib.util.spec_from_file_location("lact_gpus", _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lact_gpus"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2, help="Number of GPUs needed (default 2)")
    ap.add_argument("--max-used-mb", type=int, default=1024)
    ap.add_argument("--max-util", type=int, default=10)
    ap.add_argument("--allow-shared", action="store_true", help="Also accept GPUs in use by others (not recommended)")
    ap.add_argument("--min-free-mb", type=int, default=9216)
    a = ap.parse_args(argv)
    G = _load_gpus()
    try:
        gpus = G.query_gpus()
        chosen = G.pick_free_gpus(a.n, G.visible_gpus(gpus), a.max_used_mb, a.max_util,
                                  a.allow_shared, a.min_free_mb)
    except (OSError, subprocess.SubprocessError, ValueError, G.GpuSelectionError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(G.format_table(gpus, chosen), file=sys.stderr)
    names = ["LACT_GPU_UUID"] + [f"LACT_GPU_UUID_{i}" for i in range(2, a.n + 1)]
    for name, uuid in zip(names, chosen):
        print(f"export {name}={uuid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
