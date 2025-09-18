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
from typing import Dict, Iterable, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

# sklearn RMSE wrapper (future-proof)
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:  # sklearn < 1.4
    from sklearn.metrics import mean_squared_error

    def rmse(y_true, y_pred):  # type: ignore[override]
        return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model_utils import robust_clip_zscore_block, sanitize_predictions
from src.utils_config import cfg_hash


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


def filter_near_constant_by_folds(
    X: pd.DataFrame, n_splits: int, var_eps: float = 1e-12
) -> List[str]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    bad_union: set[str] = set()
    for fold_idx, (tr_idx, _) in enumerate(splitter.split(X), 1):
        stds = X.iloc[tr_idx].std(axis=0).fillna(0.0)
        bad = stds.index[(stds <= var_eps)].tolist()
        if bad:
            print(f"[fold{fold_idx}] drop near-constant in TRAIN: {len(bad)}")
        bad_union.update(bad)
    return sorted(bad_union)


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


def serialize_models(
    exp_dir: Path,
    boosters: Iterable[tuple[lgb.Booster, int]],
    ridge_models: Iterable[Ridge],
    fold_details: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    updated: List[Dict[str, object]] = []
    for idx, ((booster, best_iter), ridge_model, detail) in enumerate(
        zip(boosters, ridge_models, fold_details), 1
    ):
        detail = dict(detail)
        best_iteration = int(detail.get("best_iteration") or best_iter or 0)

        lgb_name = f"model_lgb_fold{idx}.txt"
        ridge_name = f"model_ridge_fold{idx}.joblib"

        booster.save_model(str(exp_dir / lgb_name), num_iteration=best_iteration)
        joblib.dump(ridge_model, exp_dir / ridge_name)

        detail.update(
            {
                "fold": idx,
                "lgb_model": lgb_name,
                "ridge_model": ridge_name,
                "best_iteration": best_iteration,
            }
        )
        updated.append(detail)

    return updated


def save_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robust LGB + Ridge ensemble with experiment logging."
    )
    parser.add_argument("--processed_root", type=str, default="data/processed")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/experiments")
    parser.add_argument("--experiment_name", type=str, default="lgb_ridge")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--z_clip", type=float, default=8.0, help="Clip range for *_rz{w} zscore features"
    )
    parser.add_argument(
        "--pred_clip_sigma",
        type=float,
        default=5.0,
        help="Clip predictions to mean ± k*std of y",
    )
    parser.add_argument(
        "--drop_zscore", action="store_true", help="Drop all *_rz{w} features entirely"
    )
    parser.add_argument(
        "--keep_zscore",
        action="store_true",
        help="Force keeping zscore features even if --drop_zscore is set",
    )
    parser.add_argument("--w1", type=float, default=0.7, help="Weight for LightGBM")
    parser.add_argument("--w2", type=float, default=0.3, help="Weight for Ridge")
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument(
        "--out_sub",
        type=str,
        default=None,
        help="Optional path for submission CSV (defaults to experiment directory).",
    )
    args = parser.parse_args()

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

    drop_zscore = bool(args.drop_zscore and not args.keep_zscore)

    # --- Load features ---
    train_df = pd.read_parquet(ver_dir / "train_l1.parquet")
    test_df = pd.read_parquet(ver_dir / "test_l1.parquet")

    target = "forward_returns"
    drop_cols = {"date_id", "is_scored", target}
    feat_cols = [c for c in train_df.columns if c not in drop_cols]

    if drop_zscore:
        feat_cols = [
            c
            for c in feat_cols
            if not (c.endswith("_rz5") or c.endswith("_rz10") or c.endswith("_rz20"))
        ]

    X = train_df[feat_cols].astype("float32").reset_index(drop=True)
    y = train_df[target].astype("float32").reset_index(drop=True)
    X_test = test_df[feat_cols].astype("float32")
    scored_mask = (
        train_df["is_scored"].reset_index(drop=True) == 1
        if "is_scored" in train_df
        else np.ones(len(train_df), dtype=bool)
    )

    if not drop_zscore:
        X = robust_clip_zscore_block(X, args.z_clip)
        X_test = robust_clip_zscore_block(X_test, args.z_clip)

    assert np.isfinite(X.to_numpy()).all(), "Train features contain NaN/Inf"
    assert np.isfinite(X_test.to_numpy()).all(), "Test features contain NaN/Inf"
    assert np.isfinite(y.to_numpy()).all(), "Target contains NaN/Inf"
    print("y stats:", y.describe().to_dict())

    # --- Fold-wise near-constant filtering (union) ---
    bad_union = filter_near_constant_by_folds(X, n_splits=args.n_splits, var_eps=1e-12)
    if bad_union:
        keep = [c for c in feat_cols if c not in set(bad_union)]
        print(f"[filter] drop union near-constant: {len(bad_union)} → keep {len(keep)}")
        feat_cols = keep
        X = X[feat_cols]
        X_test = X_test[feat_cols]

    # --- Models ---
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
    ridge_tpl = make_pipeline(
        StandardScaler(with_mean=True),
        Ridge(alpha=args.ridge_alpha, random_state=args.seed),
    )

    oof = np.full(len(y), np.nan, dtype="float32")
    preds = np.zeros(len(X_test), dtype="float32")
    covered = np.zeros(len(y), dtype=bool)

    clip_low = float(y.mean() - args.pred_clip_sigma * y.std())
    clip_high = float(y.mean() + args.pred_clip_sigma * y.std())
    print(f"[info] prediction clip range: [{clip_low:.6g}, {clip_high:.6g}]")

    splitter = TimeSeriesSplit(n_splits=args.n_splits)
    lgb_boosters: List[tuple[lgb.Booster, int]] = []
    ridge_models: List[Ridge] = []
    fold_details: List[Dict[str, object]] = []
    fi_parts: List[pd.DataFrame] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(X), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        covered[va_idx] = True

        assert np.isfinite(X_tr.to_numpy()).all() and np.isfinite(X_va.to_numpy()).all()
        assert np.isfinite(y_tr.to_numpy()).all() and np.isfinite(y_va.to_numpy()).all()

        model_lgb = lgb.LGBMRegressor(**lgb_params)
        model_lgb.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )

        model_ridge = clone(ridge_tpl)
        model_ridge.fit(X_tr, y_tr)

        p1_va = model_lgb.predict(X_va)
        p2_va = model_ridge.predict(X_va)
        p1_te = model_lgb.predict(X_test)
        p2_te = model_ridge.predict(X_test)

        fold_oof = sanitize_predictions(args.w1 * p1_va + args.w2 * p2_va, clip_low, clip_high)
        fold_test = sanitize_predictions(args.w1 * p1_te + args.w2 * p2_te, clip_low, clip_high)

        oof[va_idx] = fold_oof
        preds += fold_test.astype("float32") / args.n_splits

        best_iteration = int(getattr(model_lgb, "best_iteration_", 0) or lgb_params["n_estimators"])

        fold_metrics = {
            "r2": float(r2_score(y_va, fold_oof)),
            "rmse": float(rmse(y_va, fold_oof)),
            "mae": float(mean_absolute_error(y_va, fold_oof)),
        }
        print(
            f"[fold{fold_idx}] metrics: R2={fold_metrics['r2']:.6f} RMSE={fold_metrics['rmse']:.6f} MAE={fold_metrics['mae']:.6f}"
        )

        fold_details.append(
            {
                "fold": fold_idx,
                "train_size": int(len(tr_idx)),
                "valid_size": int(len(va_idx)),
                "best_iteration": best_iteration,
                "metrics": fold_metrics,
            }
        )

        booster = model_lgb.booster_
        if booster is None:
            raise RuntimeError("LightGBM booster missing after fit")
        lgb_boosters.append((booster, best_iteration))
        ridge_models.append(model_ridge)

        fi_df = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance_gain": booster.feature_importance(importance_type="gain"),
                "importance_split": booster.feature_importance(importance_type="split"),
                "fold": fold_idx,
            }
        )
        fi_parts.append(fi_df)

    mask = covered & scored_mask
    if not mask.any():
        raise RuntimeError("No covered & scored samples to evaluate!")

    if np.isnan(oof[mask]).any():
        bad_idx = np.where(np.isnan(oof) & mask)[0]
        print(
            f"[warn] OOF NaN remains at indices (masked): {bad_idx[:10]} ... total {len(bad_idx)}; fill with mean."
        )
        oof[bad_idx] = y.mean()

    overall_metrics = {
        "covered": int(mask.sum()),
        "total": int(len(y)),
        "coverage_ratio": float(mask.sum() / len(y)),
        "r2": float(r2_score(y[mask], oof[mask])),
        "rmse": float(rmse(y[mask], oof[mask])),
        "mae": float(mean_absolute_error(y[mask], oof[mask])),
    }

    print(
        "[overall] R2={r2:.6f} RMSE={rmse:.6f} MAE={mae:.6f}".format(
            r2=overall_metrics["r2"],
            rmse=overall_metrics["rmse"],
            mae=overall_metrics["mae"],
        )
    )

    preds = sanitize_predictions(preds, clip_low, clip_high).astype("float32")

    hash_payload = {
        "processed_dir": ver_dir.name,
        "seed": args.seed,
        "n_splits": args.n_splits,
        "weights": {"w1": args.w1, "w2": args.w2},
        "drop_zscore": drop_zscore,
        "z_clip": args.z_clip,
        "pred_clip_sigma": args.pred_clip_sigma,
        "ridge_alpha": args.ridge_alpha,
        "lgb_params": lgb_params,
    }
    exp_dir = prepare_experiment_dir(log_dir, args.experiment_name, hash_payload)
    print(f"[info] saving experiment artifacts to {exp_dir}")

    fold_details = serialize_models(exp_dir, lgb_boosters, ridge_models, fold_details)

    metrics_payload = {
        "overall": overall_metrics,
        "folds": [
            {"fold": fd["fold"], **(fd.get("metrics", {}))}
            for fd in fold_details
        ],
    }
    save_metrics(exp_dir / "metrics.json", metrics_payload)

    target_stats = {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
    }

    experiment_config = {
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "target": target,
        "n_splits": args.n_splits,
        "weights": {"lgb": args.w1, "ridge": args.w2},
        "drop_zscore": drop_zscore,
        "z_clip": args.z_clip,
        "pred_clip_sigma": args.pred_clip_sigma,
        "clip_range": [clip_low, clip_high],
        "target_stats": target_stats,
        "processed": {
            "root": str(processed_root),
            "version": ver_dir.name,
            "path": str(ver_dir),
        },
        "features": {
            "columns": feat_cols,
            "dropped_in_cv": bad_union,
        },
        "model": {
            "lgb_params": lgb_params,
            "ridge": {"alpha": args.ridge_alpha, "with_mean": True, "with_std": True},
            "folds": [
                {
                    "fold": fd["fold"],
                    "train_size": fd["train_size"],
                    "valid_size": fd["valid_size"],
                    "best_iteration": fd["best_iteration"],
                    "lgb_model": fd["lgb_model"],
                    "ridge_model": fd["ridge_model"],
                }
                for fd in fold_details
            ],
        },
    }

    save_metrics(exp_dir / "config.json", experiment_config)

    manifest_path = ver_dir / "manifest.json"
    if manifest_path.exists():
        shutil.copy(manifest_path, exp_dir / "feature_manifest.json")

    oof_df = pd.DataFrame(
        {
            "date_id": train_df["date_id"].reset_index(drop=True),
            "target": y.astype("float32"),
            "oof_prediction": oof.astype("float32"),
            "is_scored": train_df.get("is_scored", pd.Series(1, index=train_df.index))
            .reset_index(drop=True)
            .astype("int8"),
            "covered": covered.astype("int8"),
        }
    )
    oof_df.to_parquet(exp_dir / "oof_predictions.parquet", index=False)

    pred_df = pd.DataFrame(
        {
            "date_id": test_df["date_id"].reset_index(drop=True),
            "prediction": preds,
        }
    )
    pred_df.to_parquet(exp_dir / "test_predictions.parquet", index=False)

    if fi_parts:
        fi_df = pd.concat(fi_parts, ignore_index=True)
        fi_df.to_csv(exp_dir / "feature_importance_folds.csv", index=False)
        fi_mean = (
            fi_df.groupby("feature")[
                ["importance_gain", "importance_split"]
            ]
            .mean()
            .reset_index()
            .sort_values("importance_gain", ascending=False)
        )
        fi_mean.to_csv(exp_dir / "feature_importance.csv", index=False)

    default_sub_path = exp_dir / "submission.csv"
    pred_df.to_csv(default_sub_path, index=False)
    print(f"[ok] submission snapshot written to {default_sub_path}")

    if args.out_sub:
        out_path = Path(args.out_sub).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(out_path, index=False)
        print(f"[ok] submission exported to {out_path}")


if __name__ == "__main__":
    main()

