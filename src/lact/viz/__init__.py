"""Presentation layer: figures and tables (numpy/pandas in; unaware of configurations and stores)."""
from .panels import line_profile, recon_grid, roi_by_fbp_error
from .plots import (
                    alphas_from_metrics,
                    heatmaps,
                    interaction_plot,
                    metric_by_arch,
                    operator_check_figure,
                    pareto_plot,
                    per_slice_plot,
                    stagewise_plot,
                    training_curves,
                    wedge_fraction_plot,
)
from .style import ARCH_COLORS, ARCH_LABELS, apply_style, color_of, geom_label, label_of
from .tables import to_latex, to_markdown, write_table
from .windows import WINDOWS, apply_window, to_hu, window_image

__all__ = [
                    "ARCH_COLORS",
                    "ARCH_LABELS",
                    "WINDOWS",
                    "alphas_from_metrics",
                    "apply_style",
                    "apply_window",
                    "color_of",
                    "geom_label",
                    "heatmaps",
                    "interaction_plot",
                    "label_of",
                    "line_profile",
                    "metric_by_arch",
                    "operator_check_figure",
                    "pareto_plot",
                    "per_slice_plot",
                    "recon_grid",
                    "roi_by_fbp_error",
                    "stagewise_plot",
                    "to_hu",
                    "to_latex",
                    "to_markdown",
                    "training_curves",
                    "wedge_fraction_plot",
                    "window_image",
                    "write_table",
]
