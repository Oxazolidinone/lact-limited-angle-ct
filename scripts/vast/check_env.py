"""Check a rented machine before training E3–E5: usable GPU, data/matrix/E2 files present, paths matching the server.

Usage: python scripts/vast/check_env.py [--gpu-only]   (exits with 1 on errors)
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

GEOMS = ["fan_a90", "fan_a120", "fan_a150"]
CFG = "configs/env/server.yaml"


def check_gpus() -> list[str]:
    import torch

    errors = []
    if not torch.cuda.is_available():
        return ["PyTorch sees no GPU"]
    for d in range(torch.cuda.device_count()):
        dev = torch.device("cuda", d)
        try:
            x = torch.randn(1, 1, 64, 64, device=dev, requires_grad=True)
            torch.nn.Conv2d(1, 8, 3, padding=1).to(dev)(x).square().mean().backward()
            n = 1000
            crow = torch.arange(n + 1, device=dev, dtype=torch.int32) * 4
            col = torch.randint(0, n, (4 * n,), device=dev, dtype=torch.int32)
            A = torch.sparse_csr_tensor(crow, col, torch.rand(4 * n, device=dev), size=(n, n))
            (A @ torch.randn(n, 1, device=dev)).sum().item()
            print(f"  GPU{d}: {torch.cuda.get_device_name(dev)}, "
                  f"{torch.cuda.get_device_properties(dev).total_memory / 2**30:.0f} GiB: OK")
        except Exception as exc:
            errors.append(f"GPU{d}: {exc}")
    return errors


def check_files() -> list[str]:
    from lact.config import load_config, load_geometry
    from lact.infra.operators.operator_store import operator_path

    cfg = load_config([CFG])
    need = [cfg.data.images_h5, cfg.data.stats_json, cfg.data.slices_json, cfg.paths.manifest,
            cfg.paths.runs / "e2_sweep" / "best_lr.json"]
    for g in GEOMS:
        op = operator_path(load_geometry(g), cfg.data.img_size_rec, cfg.paths.operators)
        need += [cfg.paths.sino_file(g), op, op.with_suffix(".json")]
    errors = []
    for p in need:
        if p.is_file():
            print(f"  {p.relative_to(cfg.paths.outputs.parent)}: {p.stat().st_size / 2**20:,.0f} MB")
        else:
            errors.append(f"Missing {p}")
    return errors


def main() -> int:
    root = Path(os.environ.get("LACT_ROOT", Path.cwd())).resolve()
    errors = []
    marker = root / ".runtime" / "server_root.txt"
    if marker.is_file() and marker.read_text().strip() != str(root):
        errors.append(f"Project at {root}, on the server at {marker.read_text().strip()}: they must match")
    print("GPU:")
    errors += check_gpus()
    if "--gpu-only" not in sys.argv:
        print("Data:")
        errors += check_files()
    free = shutil.disk_usage(root).free / 2**30
    print(f"Free disk space: {free:.0f} GiB")
    if free < 40:
        errors.append("Less than 40 GiB of free disk space")
    for e in errors:
        print("ERROR:", e)
    print("OK" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
