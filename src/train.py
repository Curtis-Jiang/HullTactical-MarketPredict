#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training entry-point for the Hull Tactical Market Prediction project."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from src.config import cfg_hash, load_config
from src.cv import PurgedExpandingTimeSeriesSplit
from src.pipelines.training import (
    TrainingParams,
    prepare_training_data,
    drop_near_constant_features,
    run_training,
)


def set_seed(seed: int = 42) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_latest_processed(processed_root: Path) -> Path:
    candidates = sorted(processed_root.glob("htmp_v*"))
    if not candidates:
        raise FileNotFoundError(
            f"No processed dir found in {processed_root}. Run build_features.py first."
        )
    return candidates[-1]


def build_cv_splitter(cv_cfg: Dict[str, object]):
    cv_type = str(cv_cfg.get("type", "purged_timeseries")).lower()
    if cv_type != "purged_timeseries":
        return TimeSeriesSplit(n_splits=int(cv_cfg.get("n_splits", 5)))
    step_size = cv_cfg.get("step_size")
    step_size = int(step_size) if step_size is not None else None
    max_train_size = cv_cfg.get("max_train_size")
    max_train_size = int(max_train_size) if max_train_size is not None else None
    return PurgedExpandingTimeSeriesSplit(
        n_splits=int(cv_cfg.get("n_splits", 5)),
        test_size=int(cv_cfg.get("test_size", cv_cfg.get("val_size", 1500))),
        initial_train_size=int(cv_cfg.get("initial_train_size", 4000)),
        embargo=int(cv_cfg.get("embargo", 0)),
        step_size=step_size,
        max_train_size=max_train_size,
        min_test_size_ratio=float(cv_cfg.get("min_test_size_ratio", 0.6)),
    )


def prepare_experiment_dir(log_dir: Path, name: str, payload: Dict[str, object]) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_hash = cfg_hash(payload)
    base = f"{timestamp}_{name}_{exp_hash}"
    exp_dir = log_dir / base
    counter = 1
    while exp_dir.exists():
        exp_dir = log_dir / f"{base}_{counter}"
        counter += 1
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def serialize_models(exp_dir: Path, folds) -> List[Dict[str, object]]:
    details: List[Dict[str, object]] = []
    for fold in folds:
        lgb_path = exp_dir / f"model_lgb_fold{fold.fold}.txt"
        ridge_path = exp_dir / f"model_ridge_fold{fold.fold}.joblib"
        logit_path = exp_dir / f"model_logit_fold{fold.fold}.joblib"
        cat_path = exp_dir / f"model_cat_fold{fold.fold}.cbm"
        hgb_path = exp_dir / f"model_hgb_fold{fold.fold}.joblib"
        clf_path = exp_dir / f"model_clf_fold{fold.fold}.joblib"
        vol_path = exp_dir / f"model_vol_fold{fold.fold}.joblib"

        fold.booster.save_model(str(lgb_path), num_iteration=fold.best_iteration)
        joblib.dump(fold.ridge, ridge_path)
        joblib.dump(fold.logit, logit_path)
        joblib.dump(fold.classifier, clf_path)
        joblib.dump(fold.vol_model, vol_path)
        fold.cat.save_model(str(cat_path))
        joblib.dump(fold.hgb, hgb_path)

        detail = {
            "fold": fold.fold,
            "train_size": fold.train_size,
            "valid_size": fold.valid_size,
            "best_iteration": fold.best_iteration,
            "metrics": fold.metrics,
            "exposure": fold.exposure_metrics,
            "lgb_model": lgb_path.name,
            "ridge_model": ridge_path.name,
            "logit_model": logit_path.name,
            "clf_model": clf_path.name,
            "vol_model": vol_path.name,
            "cat_model": cat_path.name,
            "hgb_model": hgb_path.name,
        }
        details.append(detail)
    return details


def save_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM + Ridge ensemble with exposure calibration."
    )
    parser.add_argument("--processed_root", type=str, default="data/processed")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/experiments")
    parser.add_argument("--experiment_name", type=str, default="lgb_ridge")
    parser.add_argument("--cv_config", type=str, default="configs/cv.yml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z_clip", type=float, default=8.0)
    parser.add_argument("--pred_clip_sigma", type=float, default=5.0)
    parser.add_argument("--drop_zscore", action="store_true")
    parser.add_argument("--keep_zscore", action="store_true")
    parser.add_argument("--w1", type=float, default=0.7)
    parser.add_argument("--w2", type=float, default=0.3)
    parser.add_argument("--w_cat", type=float, default=0.4)
    parser.add_argument("--w_tabnet", type=float, default=0.3)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--logreg_c", type=float, default=1.0)
    parser.add_argument("--logreg_max_iter", type=int, default=5000)
    parser.add_argument("--signal_blend", type=float, default=0.5, help="Weight for classification signal in exposure blend (0=regression only, 1=classification only)")
    parser.add_argument("--signal_tanh_gain", type=float, default=1.0, help="Gain parameter for tanh squashing of exposure signal (<=0 to disable)")
    parser.add_argument("--class_blend", type=float, default=0.6, help="Blend weight for LightGBM classifier probabilities (0=logistic only, 1=LightGBM only)")
    parser.add_argument("--exposure_offset", type=float, default=1.0)
    parser.add_argument("--exposure_min_scale", type=float, default=0.5)
    parser.add_argument("--exposure_max_scale", type=float, default=80.0)
    parser.add_argument("--exposure_scale_steps", type=int, default=24)
    parser.add_argument("--exposure_scale_floor", type=float, default=0.05)
    parser.add_argument("--exposure_clip_low", type=float, default=0.0)
    parser.add_argument("--exposure_clip_high", type=float, default=2.0)
    parser.add_argument("--edge_cap", type=float, default=0.3)
    parser.add_argument("--turnover_cap", type=float, default=0.5)
    parser.add_argument("--vol_ratio_cap", type=float, default=1.2)
    parser.add_argument("--exposure_decay", type=float, default=0.8)
    parser.add_argument("--exposure_max_iter", type=int, default=20)
    parser.add_argument("--risk_aversion", type=float, default=20.0, help="Risk aversion factor applied to volatility-adjusted exposures")
    parser.add_argument("--tabnet_max_epochs", type=int, default=400)
    parser.add_argument("--tabnet_patience", type=int, default=30)
    parser.add_argument("--tabnet_batch_size", type=int, default=1024)
    parser.add_argument("--tabnet_virtual_batch_size", type=int, default=128)
    parser.add_argument("--tabnet_feature_limit", type=int, default=256)
    parser.add_argument("--out_sub", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics")

    processed_root = Path(args.processed_root).expanduser().resolve()
    if args.processed_dir:
        ver_dir = Path(args.processed_dir)
        if not ver_dir.is_absolute():
            ver_dir = processed_root / ver_dir
        ver_dir = ver_dir.resolve()
    else:
        ver_dir = load_latest_processed(processed_root)
    print(f"[info] using processed dir: {ver_dir}")

    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(ver_dir / "train_l1.parquet")
    test_df = pd.read_parquet(ver_dir / "test_l1.parquet")

    target = "forward_returns"
    drop_cols = {"date_id", "is_scored", target, "row_id", "id", "index"}
    drop_zscore = bool(args.drop_zscore and not args.keep_zscore)

    bundle = prepare_training_data(
        train_df,
        test_df,
        target=target,
        drop_cols=drop_cols,
        drop_zscore=drop_zscore,
        z_clip=args.z_clip,
        pred_clip_sigma=args.pred_clip_sigma,
    )

    cv_paths: List[Path] = []
    base_cv = Path("configs/cv.base.yml")
    if base_cv.exists():
        cv_paths.append(base_cv)
    cv_cfg_path = Path(args.cv_config)
    if cv_cfg_path.exists():
        cv_paths.append(cv_cfg_path)

    cv_cfg = load_config(*cv_paths) if cv_paths else {}
    cv_settings = dict(cv_cfg.get("cv", {}))
    splitter = build_cv_splitter(cv_settings)
    splits = list(splitter.split(bundle.X))

    bundle, dropped_features = drop_near_constant_features(bundle, splits)
    if dropped_features:
        print(f"[filter] drop union near-constant: {len(dropped_features)}")

    scale_steps = max(1, int(args.exposure_scale_steps))
    candidate_scales = np.linspace(
        float(args.exposure_min_scale),
        float(args.exposure_max_scale),
        scale_steps,
        dtype=float,
    )

    lgb_params = dict(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_gain_to_split=0.0,
        min_data_in_leaf=20,
        random_state=args.seed,
    )

    cat_params = dict(
        depth=6,
        learning_rate=0.03,
        iterations=1500,
        l2_leaf_reg=3.0,
        random_state=args.seed,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
    )

    hgb_params = dict(
        max_depth=7,
        learning_rate=0.05,
        max_iter=600,
        min_samples_leaf=40,
        l2_regularization=0.1,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
    )

    lgb_clf_params = dict(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        min_data_in_leaf=20,
        objective="binary",
        random_state=args.seed,
    )

    vol_lgb_params = dict(
        n_estimators=1200,
        learning_rate=0.02,
        num_leaves=15,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_data_in_leaf=40,
        random_state=args.seed,
    )

    params = TrainingParams(
        lgb_params=lgb_params,
        cat_params=cat_params,
        hgb_params=hgb_params,
        weight_cat=args.w_cat,
        weight_hgb=args.w_tabnet,
        hgb_feature_limit=args.tabnet_feature_limit,
        ridge_alpha=args.ridge_alpha,
        logreg_c=args.logreg_c,
        logreg_max_iter=args.logreg_max_iter,
        signal_blend=args.signal_blend,
        signal_tanh_gain=args.signal_tanh_gain,
        class_blend=args.class_blend,
        weight_lgb=args.w1,
        weight_ridge=args.w2,
        candidate_scales=candidate_scales,
        exposure_offset=args.exposure_offset,
        exposure_clip_low=args.exposure_clip_low,
        exposure_clip_high=args.exposure_clip_high,
        edge_cap=args.edge_cap,
        turnover_cap=args.turnover_cap,
        vol_ratio_cap=args.vol_ratio_cap,
        exposure_decay=args.exposure_decay,
        exposure_max_iter=args.exposure_max_iter,
        exposure_scale_floor=args.exposure_scale_floor,
        lgb_clf_params=lgb_clf_params,
        vol_lgb_params=vol_lgb_params,
        risk_aversion=args.risk_aversion,
    )

    artifacts = run_training(bundle, splits, params, dropped_features=dropped_features)

    hash_payload = {
        "processed_dir": ver_dir.name,
        "seed": args.seed,
        "cv": cv_settings,
        "weights": {"w1": args.w1, "w2": args.w2},
        "weight_cat": args.w_cat,
        "weight_tabnet": args.w_tabnet,
        "signal_blend": args.signal_blend,
        "signal_tanh_gain": args.signal_tanh_gain,
        "class_blend": args.class_blend,
        "drop_zscore": drop_zscore,
        "z_clip": args.z_clip,
        "pred_clip_sigma": args.pred_clip_sigma,
        "ridge_alpha": args.ridge_alpha,
        "logreg": {"c": args.logreg_c, "max_iter": args.logreg_max_iter},
        "risk_aversion": args.risk_aversion,
        "exposure": {
            "offset": args.exposure_offset,
            "clip": [args.exposure_clip_low, args.exposure_clip_high],
            "candidate_scales": candidate_scales.tolist(),
            "edge_cap": args.edge_cap,
            "turnover_cap": args.turnover_cap,
            "vol_ratio_cap": args.vol_ratio_cap,
            "decay": args.exposure_decay,
            "max_iter": args.exposure_max_iter,
            "scale_floor": args.exposure_scale_floor,
        },
        "cat_params": cat_params,
        "tabnet_params": tabnet_params,
        "tabnet_fit": {
            "max_epochs": args.tabnet_max_epochs,
            "patience": args.tabnet_patience,
            "batch_size": args.tabnet_batch_size,
            "virtual_batch_size": args.tabnet_virtual_batch_size,
            "feature_limit": args.tabnet_feature_limit,
        },
        "lgb_clf_params": lgb_clf_params,
        "vol_lgb_params": vol_lgb_params,
    }

    exp_dir = prepare_experiment_dir(log_dir, args.experiment_name, hash_payload)
    print(f"[info] saving experiment artifacts to {exp_dir}")

    fold_details = serialize_models(exp_dir, artifacts.folds)
    metrics_payload = artifacts.metrics
    save_metrics(exp_dir / "metrics.json", metrics_payload)

    target_stats = {
        "mean": float(bundle.y.mean()),
        "std": float(bundle.y.std()),
        "min": float(bundle.y.min()),
        "max": float(bundle.y.max()),
    }

    experiment_config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "target": target,
        "n_splits": len(splits),
        "weights": {"lgb": args.w1, "ridge": args.w2},
        "weight_cat": args.w_cat,
        "weight_tabnet": args.w_tabnet,
        "signal_blend": args.signal_blend,
        "signal_tanh_gain": args.signal_tanh_gain,
        "class_blend": args.class_blend,
        "drop_zscore": drop_zscore,
        "z_clip": args.z_clip,
        "pred_clip_sigma": args.pred_clip_sigma,
        "clip_range": [bundle.clip_low, bundle.clip_high],
        "target_stats": target_stats,
        "processed": {
            "root": str(processed_root),
            "version": ver_dir.name,
            "path": str(ver_dir),
        },
        "features": {
            "columns": bundle.feature_columns,
            "dropped_in_cv": dropped_features,
        },
        "cv": {
            "type": cv_settings.get("type", "purged_timeseries"),
            "config": cv_settings,
        },
        "model": {
            "lgb_params": lgb_params,
            "ridge": {"alpha": args.ridge_alpha, "with_mean": True, "with_std": True},
            "cat": cat_params,
            "tabnet": tabnet_params,
            "logistic": {"C": args.logreg_c, "max_iter": args.logreg_max_iter},
            "lgb_classifier": lgb_clf_params,
            "vol_lgb": vol_lgb_params,
            "tabnet_fit": {
                "max_epochs": args.tabnet_max_epochs,
                "patience": args.tabnet_patience,
                "batch_size": args.tabnet_batch_size,
                "virtual_batch_size": args.tabnet_virtual_batch_size,
                "feature_limit": args.tabnet_feature_limit,
            },
            "folds": [
                {
                    "fold": detail["fold"],
                    "train_size": detail["train_size"],
                    "valid_size": detail["valid_size"],
                    "best_iteration": detail["best_iteration"],
                    "lgb_model": detail["lgb_model"],
                    "ridge_model": detail["ridge_model"],
                    "logit_model": detail["logit_model"],
                    "clf_model": detail["clf_model"],
                    "vol_model": detail["vol_model"],
                    "cat_model": detail.get("cat_model"),
                    "tabnet_model": detail.get("tabnet_model"),
                }
                for detail in fold_details
            ],
        },
        "exposure": hash_payload["exposure"],
        "tabnet_fit": hash_payload["tabnet_fit"],
    }
    save_metrics(exp_dir / "config.json", experiment_config)

    manifest_path = ver_dir / "manifest.json"
    if manifest_path.exists():
        shutil.copy(manifest_path, exp_dir / "feature_manifest.json")

    oof_df = pd.DataFrame(
        {
            "date_id": train_df["date_id"].reset_index(drop=True),
            "target": bundle.y.astype("float32"),
            "oof_prediction": artifacts.oof_predictions.astype("float32"),
            "oof_exposure": artifacts.oof_exposures.astype("float32"),
            "is_scored": bundle.scored_mask.astype("int8"),
            "covered": artifacts.covered_mask.astype("int8"),
        }
    )
    oof_df.to_parquet(exp_dir / "oof_predictions.parquet", index=False)

    pred_df = pd.DataFrame(
        {
            "date_id": test_df["date_id"].reset_index(drop=True),
            "prediction": artifacts.test_predictions.astype("float32"),
        }
    )
    pred_df.to_parquet(exp_dir / "test_predictions.parquet", index=False)

    if not artifacts.feature_importance_folds.empty:
        artifacts.feature_importance_folds.to_csv(exp_dir / "feature_importance_folds.csv", index=False)
    if not artifacts.feature_importance_mean.empty:
        artifacts.feature_importance_mean.to_csv(exp_dir / "feature_importance.csv", index=False)

    default_sub_path = exp_dir / "submission.csv"
    pred_df.to_csv(default_sub_path, index=False)
    print(f"[ok] submission snapshot written to {default_sub_path}")

    if args.out_sub:
        out_path = Path(args.out_sub).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False)
        print(f"[ok] submission exported to {out_path}")

    save_metrics(exp_dir / "fold_details.json", {"folds": fold_details})

    overall = artifacts.metrics["overall"]
    print(
        "[overall] R2={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f} Sharpe={sharpe:.4f}".format(
            r2=overall["r2"],
            rmse=overall["rmse"],
            mae=overall["mae"],
            sharpe=overall["exposure"]["overall_sharpe"],
        )
    )


if __name__ == "__main__":
    main()
