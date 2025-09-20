"""Feature preprocessing helpers for model training."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_clip_zscore_block(df: pd.DataFrame, z_clip: float) -> pd.DataFrame:
    """Clip z-score style columns to ``[-z_clip, z_clip]`` in place."""

    z_cols = [
        column
        for column in df.columns
        if column.endswith("_rz5") or column.endswith("_rz10") or column.endswith("_rz20")
    ]
    if not z_cols:
        return df

    arr = df.loc[:, z_cols].to_numpy()
    np.clip(arr, -float(z_clip), float(z_clip), out=arr)
    df.loc[:, z_cols] = arr
    return df


def sanitize_predictions(values: np.ndarray, clip_low: float, clip_high: float) -> np.ndarray:
    """Replace NaN/Inf values and clip predictions to a stable range."""

    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip_high, neginf=clip_low)
    return np.clip(arr, clip_low, clip_high)


__all__ = ["robust_clip_zscore_block", "sanitize_predictions"]
