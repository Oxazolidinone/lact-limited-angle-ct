"""Paper / thesis quality figures and tables (see style.py for the shared rules)."""
from . import stats
from .figures import (
    angle_lines,
    curves,
    error_split,
    lr_sweep,
    metric_strip,
    paired_forest,
    pareto,
    qualitative,
    stage_params,
    visibility,
)
from .latex import results_table
from .style import CTX, PRESETS, color, figure, geom_label, label, marker, panel_label, save, t, use

__all__ = [
    "CTX",
    "PRESETS",
    "angle_lines",
    "color",
    "curves",
    "error_split",
    "figure",
    "geom_label",
    "label",
    "lr_sweep",
    "marker",
    "metric_strip",
    "paired_forest",
    "panel_label",
    "pareto",
    "qualitative",
    "results_table",
    "save",
    "stage_params",
    "stats",
    "t",
    "use",
    "visibility",
]
