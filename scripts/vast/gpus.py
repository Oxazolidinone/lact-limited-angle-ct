"""Which GPU runs which training run. Read-only; never touches processes.

    python3 scripts/vast/gpus.py          # print once
    python3 scripts/vast/gpus.py -w 10    # refresh every 10 seconds, Ctrl+C to quit (training keeps running)

Standard library only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path


def smi(fields: str, kind: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


def hms(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def heartbeat(cwd: Path, exp: str, model: str, geom: str, lr: str, seed: str) -> dict:
    """Heartbeat of a running run. Each data variant has its own outputs tree (outputs, outputs_deeplesion, ...) with
    the same run names: take the most recently written file."""
    try:
        lr_f = float(lr)
    except ValueError:
        return {}
    found: list[Path] = []
    for rd in cwd.glob(f"outputs*/runs/{exp}/{exp}_{model}_{geom}_lr*_s{seed}"):
        try:
            if abs(float(rd.name.rsplit("_lr", 1)[1].rsplit("_s", 1)[0]) - lr_f) <= 1e-12 * max(1.0, lr_f):
                found.append(rd / "heartbeat.json")
        except (ValueError, IndexError):
            continue
    for hb in sorted((p for p in found if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return json.loads(hb.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return {}


def proc(pid: str) -> dict | None:
    try:
        argv = [a.decode() for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
        cwd = Path(os.readlink(f"/proc/{pid}/cwd"))
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return None
    p = {"cwd": cwd, "stopped": state == "T", "model": "", "geom": "", "exp": "", "seed": "", "lr": ""}
    for a in argv:
        if a.endswith(".yaml") and ("/model/" in a or "/models/" in a):
            p["model"] = Path(a).stem
        elif a.endswith(".yaml") and "/geometry/" in a:
            p["geom"] = Path(a).stem
        elif a.startswith("exp="):
            p["exp"] = a[4:]
        elif a.startswith("seed="):
            p["seed"] = a[5:]
        elif a.startswith("train.lr="):
            p["lr"] = a[9:]
    i = argv.index("lact") if "lact" in argv else 0
    p["cmd"] = " ".join(argv[i + 1:i + 3]) or " ".join(argv[:2])
    return p


def render() -> str:
    now = dt.datetime.now(dt.timezone.utc)
    lines = [f"{now:%H:%M} UTC ({now + dt.timedelta(hours=7):%H:%M} VN)"]
    gpus = smi("index,uuid,name,utilization.gpu,memory.used,memory.total", "gpu")
    apps: dict[str, list[tuple[str, float]]] = {}
    for pid, uuid, mem in smi("pid,gpu_uuid,used_memory", "compute-apps"):
        apps.setdefault(uuid, []).append((pid, float(mem or 0)))
    for idx, uuid, name, util, used, total in gpus:
        lines.append(f"\nGPU {idx}  {name.replace('NVIDIA GeForce ', '')}  {uuid[:12]}  load {util}%  "
                     f"VRAM {float(used) / 1024:.1f}/{float(total) / 1024:.1f} GB")
        rows = []
        for pid, mem in sorted(apps.get(uuid, []), key=lambda t: int(t[0])):
            p = proc(pid)
            if p is None:
                continue
            if not p["exp"]:
                rows.append(f"  pid {pid:>6}  {p['cmd']}  {mem / 1024:.1f} GB")
                continue
            hb = heartbeat(p["cwd"], p["exp"], p["model"], p["geom"], p["lr"], p["seed"])
            step, tot, sps = hb.get("step", 0), hb.get("total_steps") or 0, hb.get("sec_per_step")
            prog = (f"{100 * step / tot:5.1f}% ({step}/{tot})  {sps:.3f}s/step  {hms((tot - step) * sps)} left"
                    if tot and sps else "starting")
            flag = "  [PAUSED]" if p["stopped"] else ""
            rows.append(f"  pid {pid:>6}  {p['exp']:<16} {p['model']:<20} {p['geom']:<9} "
                        f"lr {p['lr']:<7} s{p['seed']}  {mem / 1024:4.1f} GB  {prog}{flag}")
        lines += rows or ["  (idle)"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-w", "--watch", type=float, default=0, help="refresh every N seconds (0 = print once)")
    a = ap.parse_args()
    if a.watch <= 0:
        print(render())
        return
    try:
        while True:
            print("\033[H\033[J" + render(), flush=True)
            time.sleep(a.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
