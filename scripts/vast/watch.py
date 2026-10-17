"""tqdm-style progress bars for every training run, read from heartbeat.json (never touches running processes).

    python3 scripts/vast/watch.py              # refresh every 5 seconds, Ctrl+C to quit (training keeps running)
    python3 scripts/vast/watch.py --once       # print once
    python3 scripts/vast/watch.py --exp e3_main e4_matched
    python3 scripts/vast/watch.py --outputs outputs_poisson_1e6      # the outputs tree of a data variant

Standard library only; runs with the system python3.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs"
STATE = ROOT / "outputs" / "pipeline" / "main.json"
BLOCKS = " ▏▎▍▌▋▊▉█"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def bar(frac: float, width: int) -> str:
    frac = min(max(frac, 0.0), 1.0)
    full, rem = divmod(frac * width, 1)
    s = "█" * int(full)
    if len(s) < width:
        s += BLOCKS[int(rem * 8)] + " " * (width - len(s) - 1)
    return s


def hms(sec: float | None) -> str:
    if sec is None or sec != sec or sec < 0:
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def short(run_id: str, exp: str) -> str:
    """e2_sweep_learn_convnext_fan_a120_lr3e-05_s0 -> convnext a120 lr3e-05 s0"""
    s = run_id[len(exp) + 1:] if run_id.startswith(exp + "_") else run_id
    return s.replace("learn_", "").replace("fan_", "").replace("_", " ")


def collect(exps: list[str] | None) -> dict[str, list[dict]]:
    state = _load(STATE)
    expected = list(state.get("jobs", {}))
    names = exps or sorted({p.name for p in RUNS.iterdir() if p.is_dir()} if RUNS.exists() else set())
    out: dict[str, list[dict]] = {}
    for exp in names:
        rows = {}
        d = RUNS / exp
        ids = {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()
        ids |= {j for j in expected if j.startswith(exp + "_")}
        for rid in sorted(ids):
            rd = d / rid
            hb, summ = _load(rd / "heartbeat.json"), _load(rd / "summary.json")
            status = ("done" if (rd / "DONE").exists() else "diverged" if (rd / "DIVERGED").exists()
                      else "oom" if (rd / "OOM").exists() else hb.get("status") or "todo")
            total = hb.get("total_steps") or summ.get("total_steps") or 0
            step = total if status == "done" and total else hb.get("step", 0)
            age = None
            if hb.get("time"):
                with contextlib.suppress(ValueError):
                    age = (dt.datetime.now() - dt.datetime.fromisoformat(hb["time"])).total_seconds()
            rows[rid] = {"id": rid, "status": status, "step": step, "total": total, "epoch": hb.get("epoch"),
                         "spe": hb.get("steps_per_epoch"), "sps": hb.get("sec_per_step"), "age": age,
                         "best": summ.get("best_val_psnr", hb.get("best_val_psnr")), "loss": hb.get("loss")}
        if rows:
            out[exp] = list(rows.values())
    return out


def gpu_line() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        parts = []
        for line in r.stdout.strip().splitlines():
            i, u, m, t = (x.strip() for x in line.split(","))
            parts.append(f"GPU{i} {u:>3}% {int(m) / 1024:4.1f}/{int(t) / 1024:.0f}G")
        return "   ".join(parts)
    except Exception:
        return ""


def render(data: dict[str, list[dict]]) -> str:
    cols = shutil.get_terminal_size((140, 40)).columns
    width = max(10, min(40, cols - 95))
    lines = [f"{dt.datetime.now():%H:%M:%S}   {gpu_line()}", ""]
    for exp, rows in data.items():
        tot = sum(r["total"] for r in rows if r["total"])
        done_steps = sum(min(r["step"], r["total"]) for r in rows if r["total"])
        n_done = sum(r["status"] == "done" for r in rows)
        frac = done_steps / tot if tot else n_done / len(rows)
        etas = [(r["total"] - r["step"]) * r["sps"] for r in rows
                if r["status"] not in ("done", "diverged", "oom") and r["sps"] and r["total"]]
        lines.append(f"{exp:<22} {frac * 100:5.1f}%|{bar(frac, width)}| {n_done}/{len(rows)} runs done"
                     f"   ~{hms(max(etas) if etas else None)} left")
        for r in rows:
            f = r["step"] / r["total"] if r["total"] else (1.0 if r["status"] == "done" else 0.0)
            rate = f"{1 / r['sps']:5.2f} it/s" if r["sps"] else "   -- it/s"
            eta = hms((r["total"] - r["step"]) * r["sps"]) if r["sps"] and r["total"] else "--:--:--"
            ep = f"ep {r['epoch'] + 1:>2}/{r['total'] // r['spe']}" if r["epoch"] is not None and r["spe"] else " " * 8
            best = f"{r['best']:5.2f}dB" if isinstance(r["best"], (int, float)) else "   -- "
            flag = {"done": " done", "diverged": " DIVERGED", "oom": " OUT OF VRAM",
                    "todo": " waiting"}.get(r["status"], "")
            if r["status"] not in ("done", "todo") and r["age"] and r["age"] > 600:
                flag += f" (no update for {int(r['age'] // 60)} min)"
            lines.append(f"  {short(r['id'], exp):<26} {f * 100:5.1f}%|{bar(f, width)}| "
                         f"{r['step']:>6}/{r['total'] or '?':<6} {ep} [{eta}, {rate}] best {best}{flag}")
        lines.append("")
    if not data:
        lines.append("No training runs yet.")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="*", default=None, help="e.g. e2_sweep e3_main (default: all)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=5.0, help="seconds between refreshes")
    ap.add_argument("--outputs", default="outputs", help="outputs directory to watch (default: outputs)")
    a = ap.parse_args()
    global RUNS, STATE
    RUNS, STATE = ROOT / a.outputs / "runs", ROOT / a.outputs / "pipeline" / "main.json"
    if a.once:
        print(render(collect(a.exp)))
        return 0
    try:
        while True:
            text = render(collect(a.exp))
            sys.stdout.write("\033[H\033[2J" + text + "\n(Ctrl+C to quit — training keeps running)\n")
            sys.stdout.flush()
            time.sleep(a.every)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
