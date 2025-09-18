#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inference script that reuses saved training artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.model_utils import robust_clip_zscore_block, sanitize_predictions


def resolve_processed_dir(
    processed_root: Path | None, processed_dir: str | None, cfg: Dict[str, object]
) -> Path:
    cfg_root = Path(cfg.get("root", "data/processed")) if cfg else Path("data/processed")
    cfg_version = cfg.get("version") if cfg else None
    cfg_path = cfg.get("path") if cfg else None

    if processed_root is not None:
        cfg_root = processed_root

    if processed_dir:
        candidate = Path(processed_dir)
        if not candidate.is_absolute():
            candidate = cfg_root / candidate
        return candidate.expanduser().resolve()

    if cfg_path:
        return Path(str(cfg_path)).expanduser().resolve()

    if cfg_version is None:
        raise ValueError("Processed directory not provided and missing from config.")

    return (cfg_root / str(cfg_version)).expanduser().resolve()


def load_config(exp_dir: Path) -> Dict[str, object]:
    cfg_path = exp_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference from a saved experiment.")
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment dir")
    parser.add_argument("--processed_root", type=str, default=None)
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument(
        "--out_sub",
        type=str,
        default=None,
        help="Optional output CSV path (defaults to experiment/submission_infer.csv)",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment).expanduser().resolve()
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    config = load_config(exp_dir)
    processed_cfg = config.get("processed", {})

    processed_root = (
        Path(args.processed_root).expanduser().resolve() if args.processed_root else None
    )
    processed_dir = resolve_processed_dir(processed_root, args.processed_dir, processed_cfg)
    print(f"[info] using processed dir: {processed_dir}")

    test_path = processed_dir / "test_l1.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test features missing: {test_path}")

    test_df = pd.read_parquet(test_path)

    features = config.get("features", {})
    feat_cols: List[str] = list(features.get("columns", []))
    if not feat_cols:
        raise ValueError("No feature columns stored in config.json")

    missing_cols = sorted(set(feat_cols) - set(test_df.columns))
    if missing_cols:
        raise ValueError(f"Missing columns in processed test data: {missing_cols[:10]}")

    X_test = test_df[feat_cols].astype("float32")

    drop_zscore = bool(config.get("drop_zscore", False))
    if not drop_zscore:
        z_clip = float(config.get("z_clip", 8.0))
        X_test = robust_clip_zscore_block(X_test, z_clip)

    assert np.isfinite(X_test.to_numpy()).all(), "Test features contain NaN/Inf"

    model_cfg = config.get("model", {})
    folds: List[Dict[str, object]] = list(model_cfg.get("folds", []))
    if not folds:
        raise ValueError("No fold model metadata found in config.json")

    weights = config.get("weights", {})
    w_lgb = float(weights.get("lgb", 0.7))
    w_ridge = float(weights.get("ridge", 0.3))

    clip_low, clip_high = [float(v) for v in config.get("clip_range", [-1.0, 1.0])]

    preds = np.zeros(len(X_test), dtype="float64")
    for fold in folds:
        lgb_path = exp_dir / str(fold.get("lgb_model"))
        ridge_path = exp_dir / str(fold.get("ridge_model"))

        if not lgb_path.exists():
            raise FileNotFoundError(f"LightGBM model missing: {lgb_path}")
        if not ridge_path.exists():
            raise FileNotFoundError(f"Ridge model missing: {ridge_path}")

        booster = lgb.Booster(model_file=str(lgb_path))
        best_iter = fold.get("best_iteration")
        num_iteration = int(best_iter) if best_iter else None
        pred_lgb = booster.predict(X_test, num_iteration=num_iteration)

        model_ridge = joblib.load(ridge_path)
        pred_ridge = model_ridge.predict(X_test)

        fold_pred = sanitize_predictions(w_lgb * pred_lgb + w_ridge * pred_ridge, clip_low, clip_high)
        preds += fold_pred / len(folds)

    preds = sanitize_predictions(preds, clip_low, clip_high).astype("float32")

    if "date_id" not in test_df.columns:
        raise ValueError("'date_id' column missing in processed test data")

    submission = pd.DataFrame({"date_id": test_df["date_id"], "prediction": preds})

    out_path = Path(args.out_sub).expanduser() if args.out_sub else exp_dir / "submission_infer.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)
    print(f"[ok] submission written to {out_path}")


if __name__ == "__main__":
    main()

