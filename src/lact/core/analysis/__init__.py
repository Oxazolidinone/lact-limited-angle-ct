"""In-depth tensor analyses: frequency domain, visibility, receptive field, per-stage behaviour, compute cost."""
from .cost import benchmark_fbp, benchmark_reconstruction, model_cost, regularizer_flops
from .erf import effective_receptive_field, erf_radius
from .spectral import coverage_map, error_energy_by_region, region_masks
from .stagewise import stagewise_metrics, summarize_stagewise
from .visibility import VisibilityOperator, visibility_gates

__all__ = [
           "VisibilityOperator",
           "benchmark_fbp",
           "benchmark_reconstruction",
           "coverage_map",
           "effective_receptive_field",
           "erf_radius",
           "error_energy_by_region",
           "model_cost",
           "region_masks",
           "regularizer_flops",
           "stagewise_metrics",
           "summarize_stagewise",
           "visibility_gates",
]
