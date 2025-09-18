from __future__ import annotations

import numpy as np
import pandas as pd

from src.model_utils import robust_clip_zscore_block, sanitize_predictions


def test_robust_clip_zscore_block_clips_only_zscore_columns() -> None:
    df = pd.DataFrame(
        {
            "feature_rz5": [0.0, 10.0, -12.0, 1.5],
            "feature_rz10": [5.0, -5.0, 0.0, 2.0],
            "other": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = robust_clip_zscore_block(df.copy(), z_clip=3.0)

    assert np.all(out["feature_rz5"].to_numpy() <= 3.0)
    assert np.all(out["feature_rz5"].to_numpy() >= -3.0)
    assert np.all(out["feature_rz10"].to_numpy() <= 3.0)
    assert np.all(out["feature_rz10"].to_numpy() >= -3.0)
    assert out["other"].equals(df["other"])


def test_robust_clip_zscore_block_no_z_columns_returns_input() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = robust_clip_zscore_block(df.copy(), z_clip=1.0)
    pd.testing.assert_frame_equal(out, df)


def test_sanitize_predictions_handles_nan_and_inf() -> None:
    arr = np.array([np.nan, np.inf, -np.inf, 5.0, -5.0])
    sanitized = sanitize_predictions(arr, clip_low=-1.0, clip_high=1.0)
    expected = np.array([0.0, 1.0, -1.0, 1.0, -1.0])
    assert np.allclose(sanitized, expected)
