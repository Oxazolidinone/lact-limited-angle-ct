"""Paper-quality figures. Each function takes plain numpy/pandas (unaware of configurations and stores) and returns
a Figure.

Per-slice metric tables use the columns geometry, method, pid, idx, [seed], <metric> (psnr, ssim, rmse_hu, ...).
Call ``style.use(...)`` before plotting; save with ``style.save(fig, out, name)``.
"""
from .images import qualitative, roi_by_structure, visibility
from .metrics import angle_lines, error_split, metric_strip, paired_forest, pareto
from .training import curves, lr_sweep, sci, stage_params

__all__ = ["angle_lines", "curves", "error_split", "lr_sweep", "metric_strip", "paired_forest", "pareto", "qualitative",
    "roi_by_structure", "sci", "stage_params", "visibility"]
