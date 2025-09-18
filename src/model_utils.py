"""Utility helpers shared across training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


def robust_clip_zscore_block(df: pd.DataFrame, z_clip: float) -> pd.DataFrame:
    """Clip ``*_rz{w}`` z-score features to ``[-z_clip, z_clip]`` inplace-safely."""

    z_cols = [
        c
        for c in df.columns
        if c.endswith("_rz5") or c.endswith("_rz10") or c.endswith("_rz20")
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

