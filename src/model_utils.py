"""Deprecated shim for model utilities.

Use ``src.models.exposure`` or ``src.models.preprocessing`` instead.
"""

from __future__ import annotations

from src.models.exposure import (
    calibrate_linear_exposure,
    exposure_edge_share,
    exposure_turnover,
    linear_exposure_transform,
    sharpe_ratio,
)
from src.models.preprocessing import robust_clip_zscore_block, sanitize_predictions

__all__ = [
    "calibrate_linear_exposure",
    "exposure_edge_share",
    "exposure_turnover",
    "linear_exposure_transform",
    "robust_clip_zscore_block",
    "sanitize_predictions",
    "sharpe_ratio",
]
