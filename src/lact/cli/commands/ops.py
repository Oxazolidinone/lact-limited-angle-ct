"""lact ops build: the sparse projection matrices A of each geometry."""
from __future__ import annotations

from ...services import operators
from ..common import GEOMS_MAIN, add_geometries, geometries, load_cfg


def cmd_build(a) -> int:
    operators.build_operators(load_cfg(a), geometries(a.geometries), a.sizes, a.projector, a.overwrite)
    return 0


def register(sub, parent) -> None:
    p = sub.add_parser("ops", help="Projection operators: build the matrices")
    s = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    c = s.add_parser("build", parents=[parent], help="Build the sparse projection matrix A (ASTRA) and ||A||_2")
    add_geometries(c, GEOMS_MAIN)
    c.add_argument("--sizes", nargs="+", type=int, default=None, help="Default: img_size_rec and img_size_sim")
    c.add_argument("--projector", default="line_fanflat", choices=["line_fanflat", "strip_fanflat"])
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(func=cmd_build)
