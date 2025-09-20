"""Custom cross-validation utilities."""

from __future__ import annotations

from typing import Generator, List, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples


class PurgedExpandingTimeSeriesSplit(BaseCrossValidator):
    """Time-series splitter with expanding window and optional embargo."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 1500,
        initial_train_size: int = 4000,
        embargo: int = 0,
        step_size: int | None = None,
        max_train_size: int | None = None,
        min_test_size_ratio: float = 0.6,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if test_size < 1:
            raise ValueError("test_size must be positive")
        if initial_train_size < 1:
            raise ValueError("initial_train_size must be positive")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size must be None or >= 1")
        if not (0.0 < min_test_size_ratio <= 1.0):
            raise ValueError("min_test_size_ratio must lie in (0, 1]")

        self.n_splits = n_splits
        self.test_size = test_size
        self.initial_train_size = initial_train_size
        self.embargo = embargo
        self.step_size = step_size
        self.max_train_size = max_train_size
        self.min_test_size_ratio = min_test_size_ratio

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803 - sklearn API
        return self.n_splits

    def split(
        self, X, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:  # noqa: D401
        n_samples = _num_samples(X)
        if n_samples <= self.initial_train_size:
            raise ValueError(
                "Not enough samples for the requested initial_train_size."
            )
        if n_samples <= self.initial_train_size + max(self.test_size, 1):
            raise ValueError(
                "Not enough samples to produce a split with the given test_size."
            )

        min_val_length = max(1, int(np.ceil(self.test_size * self.min_test_size_ratio)))
        last_start_allowed = n_samples - min_val_length
        if last_start_allowed <= self.initial_train_size:
            raise ValueError(
                "Series too short to allocate validation windows under the requested configuration."
            )

        starts = self._compute_validation_starts(n_samples)
        produced = 0
        for val_start in starts:
            if produced >= self.n_splits:
                break

            val_end = min(n_samples, val_start + self.test_size)
            if val_end - val_start < min_val_length:
                continue

            if val_start < self.initial_train_size:
                continue

            train_end = max(0, val_start - self.embargo)
            if train_end <= 0:
                continue

            if self.max_train_size is not None and self.max_train_size < train_end:
                train_start = train_end - self.max_train_size
            else:
                train_start = 0

            train_idx = np.arange(train_start, train_end, dtype=int)
            val_idx = np.arange(val_start, val_end, dtype=int)
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            produced += 1
            yield train_idx, val_idx

        if produced == 0:
            raise ValueError(
                "Unable to generate any validation folds with the provided configuration."
            )

    def _compute_validation_starts(self, n_samples: int) -> List[int]:
        if self.step_size and self.step_size > 0:
            raw = self.initial_train_size + np.arange(self.n_splits) * self.step_size
        else:
            upper = n_samples - self.test_size
            if upper <= self.initial_train_size:
                upper = n_samples - 1
            raw = np.linspace(self.initial_train_size, upper, self.n_splits)

        starts: List[int] = []
        prev = self.initial_train_size - 1
        min_start = self.initial_train_size + self.embargo
        max_start = n_samples - 1
        for candidate in raw:
            start = int(np.floor(candidate))
            if start <= prev:
                start = prev + 1
            if start < min_start:
                start = min_start
            if start > max_start:
                break
            starts.append(start)
            prev = start
        return starts


__all__ = ["PurgedExpandingTimeSeriesSplit"]
