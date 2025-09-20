"""Training pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:  # sklearn < 1.4
    from sklearn.metrics import mean_squared_error

    def rmse(y_true, y_pred):  # type: ignore[override]
        return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.exposure import (
    calibrate_linear_exposure,
    exposure_edge_share,
    exposure_turnover,
    linear_exposure_transform,
    sharpe_ratio,
)
from src.models.preprocessing import robust_clip_zscore_block, sanitize_predictions


@dataclass
class TrainingParams:
    lgb_params: Dict[str, float]
    cat_params: Dict[str, Any]
    hgb_params: Dict[str, Any]
    weight_cat: float
    weight_hgb: float
    hgb_feature_limit: int
    ridge_alpha: float
    logreg_c: float
    logreg_max_iter: int
    signal_blend: float
    signal_tanh_gain: float
    class_blend: float
    weight_lgb: float
    weight_ridge: float
    candidate_scales: np.ndarray
    exposure_offset: float
    exposure_clip_low: float
    exposure_clip_high: float
    edge_cap: float
    turnover_cap: float
    vol_ratio_cap: float
    exposure_decay: float
    exposure_max_iter: int
    exposure_scale_floor: float
    lgb_clf_params: Dict[str, Any]
    vol_lgb_params: Dict[str, Any]
    risk_aversion: float


@dataclass
class TrainingDataBundle:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    feature_columns: List[str]
    scored_mask: np.ndarray
    clip_low: float
    clip_high: float


@dataclass
class FoldResult:
    fold: int
    train_size: int
    valid_size: int
    metrics: Dict[str, float]
    exposure_metrics: Dict[str, float]
    booster: lgb.Booster
    ridge: Pipeline
    logit: Pipeline
    cat: CatBoostRegressor
    hgb: HistGradientBoostingRegressor
    classifier: lgb.LGBMClassifier
    vol_model: lgb.LGBMRegressor
    best_iteration: int
    oof_pred: np.ndarray
    oof_exposure: np.ndarray
    test_pred: np.ndarray
    feature_importance: pd.DataFrame


@dataclass
class TrainingArtifacts:
    bundle: TrainingDataBundle
    folds: List[FoldResult]
    dropped_features: List[str]
    covered_mask: np.ndarray
    oof_predictions: np.ndarray
    oof_exposures: np.ndarray
    test_predictions: np.ndarray
    metrics: Dict[str, object]
    exposure_summary: Dict[str, float]
    feature_importance_folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_importance_mean: pd.DataFrame = field(default_factory=pd.DataFrame)


def prepare_training_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    drop_cols: Iterable[str],
    drop_zscore: bool,
    z_clip: float,
    pred_clip_sigma: float,
) -> TrainingDataBundle:
    feat_cols = [col for col in train_df.columns if col not in set(drop_cols)]

    X = train_df[feat_cols].astype("float32").reset_index(drop=True)
    y = train_df[target].astype("float32").reset_index(drop=True)
    X_test = test_df[feat_cols].astype("float32")

    if not drop_zscore:
        X = robust_clip_zscore_block(X, z_clip)
        X_test = robust_clip_zscore_block(X_test, z_clip)

    scored_mask = (
        train_df.get("is_scored", pd.Series(1, index=train_df.index)).reset_index(drop=True).to_numpy(dtype=bool)
    )

    clip_low = float(y.mean() - pred_clip_sigma * y.std())
    clip_high = float(y.mean() + pred_clip_sigma * y.std())

    return TrainingDataBundle(
        X=X,
        y=y,
        X_test=X_test,
        feature_columns=feat_cols,
        scored_mask=scored_mask,
        clip_low=clip_low,
        clip_high=clip_high,
    )


def drop_near_constant_features(
    bundle: TrainingDataBundle,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    var_eps: float = 1e-12,
) -> Tuple[TrainingDataBundle, List[str]]:
    dropped: set[str] = set()
    for train_idx, _ in splits:
        stds = bundle.X.iloc[train_idx].std(axis=0).fillna(0.0)
        dropped.update(stds.index[stds <= var_eps].tolist())

    if not dropped:
        return bundle, []

    keep_cols = [col for col in bundle.feature_columns if col not in dropped]
    X = bundle.X[keep_cols].copy()
    X_test = bundle.X_test[keep_cols].copy()

    updated_bundle = TrainingDataBundle(
        X=X,
        y=bundle.y,
        X_test=X_test,
        feature_columns=keep_cols,
        scored_mask=bundle.scored_mask,
        clip_low=bundle.clip_low,
        clip_high=bundle.clip_high,
    )
    return updated_bundle, sorted(dropped)


def _make_models(
    params: TrainingParams,
) -> Tuple[
    lgb.LGBMRegressor,
    Pipeline,
    Pipeline,
    CatBoostRegressor,
    HistGradientBoostingRegressor,
    lgb.LGBMClassifier,
    lgb.LGBMRegressor,
]:
    lgb_model = lgb.LGBMRegressor(**params.lgb_params)
    ridge_model = make_pipeline(
        StandardScaler(with_mean=True),
        Ridge(alpha=params.ridge_alpha, random_state=params.lgb_params.get("random_state", 42)),
    )
    logit_model = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(
            C=params.logreg_c,
            solver="saga",
            max_iter=params.logreg_max_iter,
            random_state=params.lgb_params.get("random_state", 42),
        ),
    )
    cat_model = CatBoostRegressor(**params.cat_params)
    hgb_model = HistGradientBoostingRegressor(**params.hgb_params)
    lgb_clf = lgb.LGBMClassifier(**params.lgb_clf_params)
    vol_model = lgb.LGBMRegressor(**params.vol_lgb_params)
    return lgb_model, ridge_model, logit_model, cat_model, hgb_model, lgb_clf, vol_model


def _compute_exposure_signal(
    reg_tr: np.ndarray,
    reg_va: np.ndarray,
    reg_te: np.ndarray,
    prob_tr: np.ndarray,
    prob_va: np.ndarray,
    prob_te: np.ndarray,
    train_mask: np.ndarray,
    params: TrainingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-6

    reg_mean = float(np.mean(reg_tr[train_mask]))
    reg_std = float(np.std(reg_tr[train_mask], ddof=0))
    if reg_std <= eps:
        reg_std = eps

    reg_signal_tr = (reg_tr - reg_mean) / reg_std
    reg_signal_va = (reg_va - reg_mean) / reg_std
    reg_signal_te = (reg_te - reg_mean) / reg_std

    weight = float(np.clip(params.signal_blend, 0.0, 1.0))
    signal_tr = reg_signal_tr
    signal_va = reg_signal_va
    signal_te = reg_signal_te

    if weight > 0.0:
        prob_mean = float(np.mean(prob_tr[train_mask]))
        prob_std = float(np.std(prob_tr[train_mask], ddof=0))
        if prob_std <= eps:
            prob_std = eps

        prob_signal_tr = (prob_tr - prob_mean) / prob_std
        prob_signal_va = (prob_va - prob_mean) / prob_std
        prob_signal_te = (prob_te - prob_mean) / prob_std

        signal_tr = weight * prob_signal_tr + (1.0 - weight) * reg_signal_tr
        signal_va = weight * prob_signal_va + (1.0 - weight) * reg_signal_va
        signal_te = weight * prob_signal_te + (1.0 - weight) * reg_signal_te

    gain = max(params.signal_tanh_gain, eps)
    if params.signal_tanh_gain > 0.0:
        signal_tr = np.tanh(signal_tr / gain)
        signal_va = np.tanh(signal_va / gain)
        signal_te = np.tanh(signal_te / gain)

    return signal_tr, signal_va, signal_te


def _calibrate_fold_exposure(
    signal_tr: np.ndarray,
    signal_va: np.ndarray,
    signal_te: np.ndarray,
    vol_adj_tr: np.ndarray,
    vol_adj_va: np.ndarray,
    vol_adj_te: np.ndarray,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    params: TrainingParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    base_cfg = calibrate_linear_exposure(
        signal_tr[train_mask],
        y_tr[train_mask],
        params.candidate_scales,
        offset=params.exposure_offset,
        low=params.exposure_clip_low,
        high=params.exposure_clip_high,
        max_edge_share=params.edge_cap,
        max_turnover=params.turnover_cap,
    )

    min_scale = float(params.candidate_scales.min())
    floor_scale = max(float(params.exposure_scale_floor), 1e-4)
    current_scale = max(min_scale, float(base_cfg["scale"]))
    offset = float(base_cfg["offset"])
    center = float(base_cfg["center"])

    attempts = 0
    exposures_tr = exposures_va = exposures_te = None
    metrics_exp: Dict[str, float] | None = None

    for attempt in range(1, params.exposure_max_iter + 1):
        exposures_tr = linear_exposure_transform(
            signal_tr,
            center=center,
            scale=current_scale,
            offset=offset,
            low=params.exposure_clip_low,
            high=params.exposure_clip_high,
        )
        exposures_va = linear_exposure_transform(
            signal_va,
            center=center,
            scale=current_scale,
            offset=offset,
            low=params.exposure_clip_low,
            high=params.exposure_clip_high,
        )
        exposures_te = linear_exposure_transform(
            signal_te,
            center=center,
            scale=current_scale,
            offset=offset,
            low=params.exposure_clip_low,
            high=params.exposure_clip_high,
        )

        exposures_tr = np.clip(exposures_tr * vol_adj_tr, params.exposure_clip_low, params.exposure_clip_high)
        exposures_va = np.clip(exposures_va * vol_adj_va, params.exposure_clip_low, params.exposure_clip_high)
        exposures_te = np.clip(exposures_te * vol_adj_te, params.exposure_clip_low, params.exposure_clip_high)

        train_returns = exposures_tr[train_mask] * y_tr[train_mask]
        val_returns = exposures_va[val_mask] * y_va[val_mask]

        train_sigma = float(np.std(train_returns, ddof=0))
        val_sigma = float(np.std(val_returns, ddof=0))
        vol_ratio = 0.0 if train_sigma <= 1e-12 else val_sigma / train_sigma

        metrics_exp = {
            "scale": current_scale,
            "center": center,
            "offset": offset,
            "train_sharpe": sharpe_ratio(train_returns),
            "train_sigma": train_sigma,
            "val_sharpe": sharpe_ratio(val_returns),
            "val_sigma": val_sigma,
            "edge_share": exposure_edge_share(
                exposures_va[val_mask],
                low=params.exposure_clip_low,
                high=params.exposure_clip_high,
            ),
            "turnover": exposure_turnover(exposures_va[val_mask]),
            "vol_ratio": vol_ratio,
            "attempts": float(attempt),
        }
        attempts = attempt
        constraints_ok = (
            metrics_exp["edge_share"] <= params.edge_cap
            and metrics_exp["turnover"] <= params.turnover_cap
            and metrics_exp["vol_ratio"] <= params.vol_ratio_cap
        )
        if constraints_ok:
            break
        next_scale = current_scale * params.exposure_decay
        if next_scale >= current_scale - 1e-12:
            break
        current_scale = max(floor_scale, next_scale)
        if current_scale <= floor_scale + 1e-9:
            continue

    if exposures_tr is None or exposures_va is None or exposures_te is None or metrics_exp is None:
        raise RuntimeError("Exposure calibration failed to produce outputs")

        if (
            metrics_exp["vol_ratio"] > params.vol_ratio_cap
            and metrics_exp["vol_ratio"] > 0.0
        ):
            scale_factor = max(params.vol_ratio_cap / metrics_exp["vol_ratio"], 0.0)
        exposures_tr = np.clip(
            exposures_tr * scale_factor,
            params.exposure_clip_low,
            params.exposure_clip_high,
        )
        exposures_va = np.clip(
            exposures_va * scale_factor,
            params.exposure_clip_low,
            params.exposure_clip_high,
        )
        exposures_te = np.clip(
            exposures_te * scale_factor,
            params.exposure_clip_low,
            params.exposure_clip_high,
        )

        train_returns = exposures_tr[train_mask] * y_tr[train_mask]
        val_returns = exposures_va[val_mask] * y_va[val_mask]
        train_sigma = float(np.std(train_returns, ddof=0))
        val_sigma = float(np.std(val_returns, ddof=0))
        vol_ratio = 0.0 if train_sigma <= 1e-12 else val_sigma / train_sigma

        current_scale *= scale_factor
        metrics_exp.update(
            {
                "train_sharpe": sharpe_ratio(train_returns),
                "train_sigma": train_sigma,
                "val_sharpe": sharpe_ratio(val_returns),
                "val_sigma": val_sigma,
                "edge_share": exposure_edge_share(
                    exposures_va[val_mask],
                    low=params.exposure_clip_low,
                    high=params.exposure_clip_high,
                ),
                "turnover": exposure_turnover(exposures_va[val_mask]),
                "vol_ratio": vol_ratio,
                "scale": float(metrics_exp["scale"]) * scale_factor,
            }
        )

    metrics_exp["scale"] = float(current_scale)
    metrics_exp.setdefault("attempts", float(attempts))
    return exposures_tr, exposures_va, exposures_te, metrics_exp


def run_training(
    bundle: TrainingDataBundle,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    params: TrainingParams,
    *,
    dropped_features: List[str] | None = None,
) -> TrainingArtifacts:
    num_samples = len(bundle.y)
    num_test = len(bundle.X_test)
    num_folds = len(splits)

    oof = np.full(num_samples, np.nan, dtype="float32")
    oof_exposure = np.full(num_samples, np.nan, dtype="float32")
    preds = np.zeros(num_test, dtype="float32")
    covered = np.zeros(num_samples, dtype=bool)
    fold_results: List[FoldResult] = []
    exposure_log: List[Dict[str, float]] = []
    fi_parts: List[pd.DataFrame] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, y_tr = bundle.X.iloc[tr_idx], bundle.y.iloc[tr_idx]
        X_va, y_va = bundle.X.iloc[va_idx], bundle.y.iloc[va_idx]

        (
            lgb_model,
            ridge_model,
            logit_model,
            cat_model,
            hgb_model,
            lgb_clf,
            vol_model,
        ) = _make_models(params)
        lgb_model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        )
        ridge_model = clone(ridge_model)
        ridge_model.fit(X_tr, y_tr)

        cat_model.fit(X_tr, y_tr, verbose=False)

        limit = max(1, params.hgb_feature_limit)
        tabnet_cols = bundle.feature_columns[:limit]
        X_tr_tab = X_tr[tabnet_cols].to_numpy(dtype=np.float32)
        X_va_tab = X_va[tabnet_cols].to_numpy(dtype=np.float32)
        X_te_tab = bundle.X_test[tabnet_cols].to_numpy(dtype=np.float32)

        hgb_model.fit(X_tr_tab, y_tr)

        logit_model = clone(logit_model)
        y_tr_bin = (y_tr > 0).astype(int)
        logit_model.fit(X_tr, y_tr_bin)

        lgb_clf = clone(lgb_clf)
        lgb_clf.fit(X_tr, y_tr_bin)

        vol_model = clone(vol_model)
        vol_model.fit(X_tr, np.abs(y_tr.to_numpy()))

        p1_tr = lgb_model.predict(X_tr)
        p1_va = lgb_model.predict(X_va)
        p1_te = lgb_model.predict(bundle.X_test)

        p2_tr = ridge_model.predict(X_tr)
        p2_va = ridge_model.predict(X_va)
        p2_te = ridge_model.predict(bundle.X_test)

        p3_tr = cat_model.predict(X_tr)
        p3_va = cat_model.predict(X_va)
        p3_te = cat_model.predict(bundle.X_test)

        p4_tr = hgb_model.predict(X_tr_tab)
        p4_va = hgb_model.predict(X_va_tab)
        p4_te = hgb_model.predict(X_te_tab)

        weight_cat = float(params.weight_cat)
        weight_hgb = float(params.weight_hgb)
        reg_tr = sanitize_predictions(
            params.weight_lgb * p1_tr + params.weight_ridge * p2_tr + weight_cat * p3_tr + weight_hgb * p4_tr,
            bundle.clip_low,
            bundle.clip_high,
        )
        reg_va = sanitize_predictions(
            params.weight_lgb * p1_va + params.weight_ridge * p2_va + weight_cat * p3_va + weight_hgb * p4_va,
            bundle.clip_low,
            bundle.clip_high,
        )
        reg_te = sanitize_predictions(
            params.weight_lgb * p1_te + params.weight_ridge * p2_te + weight_cat * p3_te + weight_hgb * p4_te,
            bundle.clip_low,
            bundle.clip_high,
        )

        prob_logit_tr = logit_model.predict_proba(X_tr)[:, 1]
        prob_logit_va = logit_model.predict_proba(X_va)[:, 1]
        prob_logit_te = logit_model.predict_proba(bundle.X_test)[:, 1]

        prob_lgb_tr = lgb_clf.predict_proba(X_tr)[:, 1]
        prob_lgb_va = lgb_clf.predict_proba(X_va)[:, 1]
        prob_lgb_te = lgb_clf.predict_proba(bundle.X_test)[:, 1]

        class_blend = np.clip(params.class_blend, 0.0, 1.0)
        prob_tr = class_blend * prob_lgb_tr + (1.0 - class_blend) * prob_logit_tr
        prob_va = class_blend * prob_lgb_va + (1.0 - class_blend) * prob_logit_va
        prob_te = class_blend * prob_lgb_te + (1.0 - class_blend) * prob_logit_te

        vol_pred_tr = vol_model.predict(X_tr)
        vol_pred_va = vol_model.predict(X_va)
        vol_pred_te = vol_model.predict(bundle.X_test)

        risk_aversion = max(params.risk_aversion, 0.0)
        vol_adj_tr = 1.0 / (1.0 + risk_aversion * np.clip(vol_pred_tr, 0.0, None))
        vol_adj_va = 1.0 / (1.0 + risk_aversion * np.clip(vol_pred_va, 0.0, None))
        vol_adj_te = 1.0 / (1.0 + risk_aversion * np.clip(vol_pred_te, 0.0, None))

        oof[va_idx] = reg_va.astype("float32")
        covered[va_idx] = True
        preds += reg_te.astype("float32") / num_folds

        train_mask = bundle.scored_mask[tr_idx]
        val_mask = bundle.scored_mask[va_idx]
        if not np.any(train_mask):
            train_mask = np.ones_like(train_mask, dtype=bool)
        if not np.any(val_mask):
            val_mask = np.ones_like(val_mask, dtype=bool)

        signal_tr, signal_va, signal_te = _compute_exposure_signal(
            reg_tr,
            reg_va,
            reg_te,
            prob_tr,
            prob_va,
            prob_te,
            train_mask,
            params,
        )

        exposures_tr, exposures_va, exposures_te, exposure_metrics = _calibrate_fold_exposure(
            signal_tr,
            signal_va,
            signal_te,
            vol_adj_tr,
            vol_adj_va,
            vol_adj_te,
            y_tr.to_numpy(),
            y_va.to_numpy(),
            train_mask,
            val_mask,
            params,
        )

        oof_exposure[va_idx] = exposures_va.astype("float32")
        exposure_metrics.update(
            {
                "train_size_scored": float(train_mask.sum()),
                "valid_size_scored": float(val_mask.sum()),
            }
        )
        exposure_log.append(exposure_metrics)

        best_iteration = int(getattr(lgb_model, "best_iteration_", 0) or params.lgb_params.get("n_estimators", 0))
        fold_metrics = {
            "r2": float(r2_score(y_va, reg_va)),
            "rmse": float(rmse(y_va, reg_va)),
            "mae": float(mean_absolute_error(y_va, reg_va)),
            "sharpe": float(exposure_metrics["val_sharpe"]),
            "edge_share": float(exposure_metrics["edge_share"]),
            "turnover": float(exposure_metrics["turnover"]),
            "vol_ratio": float(exposure_metrics["vol_ratio"]),
        }

        booster = lgb_model.booster_
        if booster is None:
            raise RuntimeError("LightGBM booster missing after fit")

        fi_df = pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "importance_gain": booster.feature_importance(importance_type="gain"),
                "importance_split": booster.feature_importance(importance_type="split"),
                "fold": fold_idx,
            }
        )
        fi_parts.append(fi_df)

        fold_results.append(
            FoldResult(
                fold=fold_idx,
                train_size=int(len(tr_idx)),
                valid_size=int(len(va_idx)),
                metrics=fold_metrics,
                exposure_metrics=exposure_metrics,
                booster=booster,
                ridge=ridge_model,
                logit=logit_model,
                cat=cat_model,
                hgb=hgb_model,
                classifier=lgb_clf,
                vol_model=vol_model,
                best_iteration=best_iteration,
                oof_pred=reg_va.astype("float32"),
                oof_exposure=exposures_va.astype("float32"),
                test_pred=reg_te.astype("float32"),
                feature_importance=fi_df,
            )
        )

    mask = covered & bundle.scored_mask
    if not mask.any():
        raise RuntimeError("No covered & scored samples to evaluate")

    if np.isnan(oof[mask]).any():
        nan_idx = np.where(np.isnan(oof) & mask)[0]
        oof[nan_idx] = bundle.y.mean()

    exposure_mask = mask & ~np.isnan(oof_exposure)
    if not exposure_mask.any():
        raise RuntimeError("No exposure data available on covered samples")

    exposure_summary = _summarise_exposure(
        oof_exposure,
        bundle.y.to_numpy(),
        exposure_mask,
        exposure_log,
        params,
    )

    overall_metrics = {
        "covered": int(mask.sum()),
        "total": int(len(bundle.y)),
        "coverage_ratio": float(mask.sum() / len(bundle.y)),
        "r2": float(r2_score(bundle.y[mask], oof[mask])),
        "rmse": float(rmse(bundle.y[mask], oof[mask])),
        "mae": float(mean_absolute_error(bundle.y[mask], oof[mask])),
        "exposure": exposure_summary,
    }

    metrics = {
        "overall": overall_metrics,
        "folds": [
            {
                "fold": fold.fold,
                **fold.metrics,
                "exposure": fold.exposure_metrics,
            }
            for fold in fold_results
        ],
    }

    fi_folds = pd.concat(fi_parts, ignore_index=True) if fi_parts else pd.DataFrame()
    fi_mean = (
        fi_folds.groupby("feature")[
            ["importance_gain", "importance_split"]
        ]
        .mean()
        .reset_index()
        .sort_values("importance_gain", ascending=False)
        if not fi_folds.empty
        else pd.DataFrame()
    )

    return TrainingArtifacts(
        bundle=bundle,
        folds=fold_results,
        dropped_features=dropped_features or [],
        covered_mask=covered,
        oof_predictions=oof,
        oof_exposures=oof_exposure,
        test_predictions=preds,
        metrics=metrics,
        exposure_summary=exposure_summary,
        feature_importance_folds=fi_folds,
        feature_importance_mean=fi_mean,
    )


def _summarise_exposure(
    exposures: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    fold_metrics: List[Dict[str, float]],
    params: TrainingParams,
) -> Dict[str, float]:
    exposures_eval = exposures[mask]
    returns_eval = exposures_eval * target[mask]

    fold_sharpes = np.array([metrics["val_sharpe"] for metrics in fold_metrics], dtype=float)
    fold_edges = np.array([metrics["edge_share"] for metrics in fold_metrics], dtype=float)
    fold_turnovers = np.array([metrics["turnover"] for metrics in fold_metrics], dtype=float)
    fold_vol_ratios = np.array([metrics["vol_ratio"] for metrics in fold_metrics], dtype=float)

    return {
        "overall_sharpe": sharpe_ratio(returns_eval),
        "overall_edge_share": exposure_edge_share(
            exposures_eval,
            low=params.exposure_clip_low,
            high=params.exposure_clip_high,
        ),
        "overall_turnover": exposure_turnover(exposures_eval),
        "fold_sharpe_mean": float(fold_sharpes.mean()) if fold_sharpes.size else 0.0,
        "fold_sharpe_std": float(fold_sharpes.std(ddof=0)) if fold_sharpes.size else 0.0,
        "fold_sharpe_min": float(fold_sharpes.min()) if fold_sharpes.size else 0.0,
        "fold_edge_mean": float(fold_edges.mean()) if fold_edges.size else 0.0,
        "fold_edge_max": float(fold_edges.max()) if fold_edges.size else 0.0,
        "fold_turnover_mean": float(fold_turnovers.mean()) if fold_turnovers.size else 0.0,
        "fold_turnover_max": float(fold_turnovers.max()) if fold_turnovers.size else 0.0,
        "fold_vol_ratio_max": float(fold_vol_ratios.max()) if fold_vol_ratios.size else 0.0,
    }
