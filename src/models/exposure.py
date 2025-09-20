"""Exposure calibration and risk diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


def sharpe_ratio(returns: np.ndarray) -> float:
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    if std <= 1e-12:
        return 0.0
    return mean / std


def exposure_turnover(exposures: np.ndarray) -> float:
    arr = np.asarray(exposures, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def exposure_edge_share(
    exposures: np.ndarray,
    *,
    low: float = 0.0,
    high: float = 2.0,
    tol: float = 1e-6,
) -> float:
    arr = np.asarray(exposures, dtype=float)
    if arr.size == 0:
        return 0.0
    low_mask = arr <= (low + tol)
    high_mask = arr >= (high - tol)
    return float(np.mean(low_mask | high_mask))


def linear_exposure_transform(
    preds: np.ndarray,
    *,
    center: float,
    scale: float,
    offset: float = 1.0,
    low: float = 0.0,
    high: float = 2.0,
) -> np.ndarray:
    arr = np.asarray(preds, dtype=float)
    exposures = offset + scale * (arr - center)
    return np.clip(exposures, low, high)


@dataclass(slots=True)
class ExposureCalibration:
    scale: float
    center: float
    offset: float
    sharpe: float
    edge_share: float
    turnover: float
    sigma: float


def calibrate_linear_exposure(
    preds: np.ndarray,
    returns: np.ndarray,
    candidate_scales: Iterable[float],
    *,
    offset: float = 1.0,
    low: float = 0.0,
    high: float = 2.0,
    max_edge_share: float = 0.3,
    max_turnover: float = 0.5,
) -> Dict[str, float]:
    preds_arr = np.asarray(preds, dtype=float)
    returns_arr = np.asarray(returns, dtype=float)
    scales = np.asarray(list(candidate_scales), dtype=float)
    if preds_arr.size == 0 or returns_arr.size == 0:
        raise ValueError("Training data for exposure calibration is empty")
    if scales.size == 0:
        raise ValueError("candidate_scales must contain at least one value")

    center = float(np.median(preds_arr))
    best: Dict[str, float] | None = None
    for scale in scales:
        exposures = linear_exposure_transform(
            preds_arr,
            center=center,
            scale=float(scale),
            offset=offset,
            low=low,
            high=high,
        )
        edge_share = exposure_edge_share(exposures, low=low, high=high)
        turnover = exposure_turnover(exposures)
        if edge_share > max_edge_share or turnover > max_turnover:
            continue
        fold_returns = exposures * returns_arr
        sharpe = sharpe_ratio(fold_returns)
        if best is None or sharpe > best["sharpe"]:
            best = {
                "scale": float(scale),
                "center": center,
                "offset": offset,
                "sharpe": sharpe,
                "edge_share": edge_share,
                "turnover": turnover,
                "sigma": float(np.std(fold_returns, ddof=0)),
            }

    if best is not None:
        return best

    fallback_scale = float(scales.min())
    exposures = linear_exposure_transform(
        preds_arr,
        center=center,
        scale=fallback_scale,
        offset=offset,
        low=low,
        high=high,
    )
    fold_returns = exposures * returns_arr
    return {
        "scale": fallback_scale,
        "center": center,
        "offset": offset,
        "sharpe": sharpe_ratio(fold_returns),
        "edge_share": exposure_edge_share(exposures, low=low, high=high),
        "turnover": exposure_turnover(exposures),
        "sigma": float(np.std(fold_returns, ddof=0)),
    }


__all__ = [
    "ExposureCalibration",
    "calibrate_linear_exposure",
    "exposure_edge_share",
    "exposure_turnover",
    "linear_exposure_transform",
    "sharpe_ratio",
]
