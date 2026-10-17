"""lact demo: the demo UI (Gradio)."""
from __future__ import annotations

from ..common import load_cfg


def cmd_demo(a) -> int:
    from ...services.demo import DemoBackend
    from ..demo_app import build_ui

    dev = "cpu" if a.device == "auto" else a.device
    build_ui(DemoBackend(load_cfg(a), a.split, dev, a.seed)).launch(server_port=a.port)
    return 0


def register(sub, parent) -> None:
    c = sub.add_parser("demo", parents=[parent], help="Demo application: pick a slice, angle and method")
    c.add_argument("--split", default="test", choices=["val", "test"])
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--port", type=int, default=7860)
    c.set_defaults(func=cmd_demo)
