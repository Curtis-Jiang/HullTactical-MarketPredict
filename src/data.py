from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

def load_raw_data(
    path: str | Path,
    *,
    expected_columns: Sequence[str] | None = None,
    required_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """加载原始 CSV 并执行基础校验与缺失统计。"""

    csv_path = Path(path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    rows, cols = df.shape
    LOGGER.info("Loaded raw data from %s with shape (%d, %d)", csv_path, rows, cols)

    if expected_columns is not None:
        if list(df.columns) != list(expected_columns):
            raise ValueError(
                "Columns in raw data do not match expected order.\n"
                f"Expected: {list(expected_columns)}\n"
                f"Found: {df.columns.tolist()}"
            )

    if required_columns is not None:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    total_cells = rows * cols
    if total_cells == 0:
        LOGGER.warning("Raw data is empty: %s", csv_path)
        return df

    total_missing = int(df.isna().sum().sum())
    overall_ratio = total_missing / total_cells
    LOGGER.info(
        "Overall missing values: %d (%.4f%%)",
        total_missing,
        overall_ratio * 100,
    )

    if rows > 0:
        col_missing_ratio = (df.isna().sum() / rows).sort_values(ascending=False)
        cols_with_missing = col_missing_ratio[col_missing_ratio > 0]
        if not cols_with_missing.empty:
            top_missing = cols_with_missing.head(10)
            ratio_str = top_missing.map(lambda r: f"{r:.2%}").to_string()
            LOGGER.info("Top columns by missing ratio:\n%s", ratio_str)

    return df
