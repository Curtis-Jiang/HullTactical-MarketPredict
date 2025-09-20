"""Feature engineering pipeline utilities."""

from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import cfg_hash

DEFAULT_CFG: Dict[str, Any] = {
    "target": "forward_returns",
    "drop_cols": ["date_id", "is_scored", "forward_returns", "row_id", "id", "index"],
    "safe_feature_policy": {"mode": "intersection_numeric"},
    "impute": {
        "add_isna_flags": True,
        "forward_fill": True,
        "rolling_mean_fallback": {"enable": True, "window": 20},
        "global_median_fallback": True,
    },
    "winsorize": {"enable": True, "q": 0.01},
    "rolling": {"windows": [5, 10, 20], "stats": ["mean", "std", "zscore"]},
    "lag": {"enable": True, "lags": [1, 5, 10], "diff": True, "pct_change": False},
    "binary_rolling": {
        "enable": True,
        "prefixes": ["D"],
        "windows": [5, 10, 20],
    },
    "prefix_aggregates": {
        "enable": True,
        "prefixes": ["D", "E", "I", "M", "P", "S", "V"],
    },
    "pairwise_diff_ratio": {
        "enable": True,
        "max_combinations": 30,
        "epsilon": 1e-6,
    },
}


def read_yaml_if_any(path: Path) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception:
        return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")
    return df


def infer_binary_columns(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    if not prefixes:
        return []
    prefixes = tuple(prefixes)
    return [column for column in columns if column.startswith(prefixes)]


def select_safe_columns(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    policy = cfg.get("safe_feature_policy", {"mode": "intersection_numeric"})
    mode = policy.get("mode", "intersection_numeric")
    if mode == "intersection_numeric":
        train_cols = set(train.select_dtypes(include=[np.number]).columns)
        test_cols = set(test.select_dtypes(include=[np.number]).columns)
        safe = sorted(train_cols & test_cols)
    else:
        safe = [col for col in train.columns if col in test.columns]
    drop_cols = set(cfg.get("drop_cols", []))
    return [col for col in safe if col not in drop_cols]


def time_aware_impute_block(
    X: pd.DataFrame,
    T: pd.DataFrame,
    *,
    safe_cols: Sequence[str],
    train_for_stats: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    add_flags = cfg.get("impute", {}).get("add_isna_flags", True)
    X_block = X[safe_cols].copy()
    T_block = T[safe_cols].copy()

    flags_tr = pd.DataFrame(index=X_block.index)
    flags_te = pd.DataFrame(index=T_block.index)

    if add_flags:
        for col in safe_cols:
            flags_tr[f"{col}_isna"] = X_block[col].isna().astype(np.int8)
            flags_te[f"{col}_isna"] = T_block[col].isna().astype(np.int8)

    if cfg.get("impute", {}).get("forward_fill", True):
        X_block = X_block.ffill()
        T_block = T_block.ffill()

    roll_cfg = cfg.get("impute", {}).get("rolling_mean_fallback", {"enable": True, "window": 20})
    if roll_cfg.get("enable", True):
        window = int(roll_cfg.get("window", 20))
        X_block = X_block.fillna(
            X_block.rolling(window=window, min_periods=1).mean()
        )
        T_block = T_block.fillna(
            T_block.rolling(window=window, min_periods=1).mean()
        )

    if cfg.get("impute", {}).get("global_median_fallback", True):
        medians = train_for_stats[safe_cols].median()
        X_block = X_block.fillna(medians)
        T_block = T_block.fillna(medians)

    X_block = downcast_float32(X_block)
    T_block = downcast_float32(T_block)

    if add_flags:
        flags_tr = flags_tr.astype("int8")
        flags_te = flags_te.astype("int8")
    else:
        flags_tr = None
        flags_te = None

    return X_block, T_block, flags_tr, flags_te


def winsorize_block(
    X: pd.DataFrame,
    T: pd.DataFrame,
    safe_cols: Sequence[str],
    *,
    q: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    thresholds: Dict[str, Tuple[float, float]] = {}
    X_out = X.copy()
    T_out = T.copy()
    for col in safe_cols:
        lo, hi = X[col].quantile([q, 1 - q])
        thresholds[col] = (float(lo), float(hi))
        X_out[col] = X[col].clip(lo, hi)
        T_out[col] = T[col].clip(lo, hi)
    return X_out.astype("float32"), T_out.astype("float32"), thresholds


def rolling_features_block(
    src_train: pd.DataFrame,
    src_test: pd.DataFrame,
    *,
    safe_cols: Sequence[str],
    windows: Sequence[int],
    stats: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outs_tr: List[pd.DataFrame] = []
    outs_te: List[pd.DataFrame] = []

    for window in windows:
        roll_tr = src_train[safe_cols].rolling(window=window, min_periods=1)
        roll_te = src_test[safe_cols].rolling(window=window, min_periods=1)

        if "mean" in stats:
            outs_tr.append(roll_tr.mean().add_suffix(f"_rm{window}"))
            outs_te.append(roll_te.mean().add_suffix(f"_rm{window}"))
        if "std" in stats:
            outs_tr.append(roll_tr.std().add_suffix(f"_rs{window}"))
            outs_te.append(roll_te.std().add_suffix(f"_rs{window}"))
        if "zscore" in stats:
            mean_tr = roll_tr.mean()
            std_tr = roll_tr.std().replace(0.0, np.nan)
            z_tr = (src_train[safe_cols] - mean_tr) / std_tr
            z_tr = z_tr.add_suffix(f"_rz{window}")
            outs_tr.append(z_tr)

            mean_te = roll_te.mean()
            std_te = roll_te.std().replace(0.0, np.nan)
            z_te = (src_test[safe_cols] - mean_te) / std_te
            outs_te.append(z_te.add_suffix(f"_rz{window}"))

    X_roll = pd.concat(outs_tr, axis=1) if outs_tr else pd.DataFrame(index=src_train.index)
    T_roll = pd.concat(outs_te, axis=1) if outs_te else pd.DataFrame(index=src_test.index)

    if not X_roll.empty:
        X_roll = X_roll.fillna(0.0).astype("float32")
    if not T_roll.empty:
        T_roll = T_roll.fillna(0.0).astype("float32")

    return X_roll, T_roll


def lag_features_block(
    train_block: pd.DataFrame,
    test_block: pd.DataFrame,
    train_dates: Sequence[int],
    test_dates: Sequence[int],
    safe_cols: Sequence[str],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    lags_raw = cfg.get("lags", []) if cfg.get("enable", True) else []
    try:
        lags = sorted({int(lag) for lag in lags_raw if int(lag) > 0})
    except Exception:
        lags = []
    if not lags:
        return pd.DataFrame(index=train_block.index), pd.DataFrame(index=test_block.index), {
            "enable": False,
            "lags": [],
            "diff": False,
            "pct_change": False,
            "n_features": 0,
        }

    idx_train = pd.Index(train_dates, name="date_id")
    idx_test = pd.Index(test_dates, name="date_id")

    base = pd.DataFrame(train_block[safe_cols].to_numpy(), columns=safe_cols, index=idx_train)
    base = base.sort_index()

    feats: List[pd.DataFrame] = []
    for lag in lags:
        shifted = base.shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in safe_cols]
        feats.append(shifted)

    if cfg.get("diff", False):
        for lag in lags:
            diff_df = base - base.shift(lag)
            diff_df.columns = [f"{col}_chg{lag}" for col in safe_cols]
            feats.append(diff_df)

    if cfg.get("pct_change", False):
        for lag in lags:
            pct_df = base.pct_change(periods=lag, fill_method=None).replace([np.inf, -np.inf], np.nan)
            pct_df.columns = [f"{col}_pct{lag}" for col in safe_cols]
            feats.append(pct_df)

    combined = pd.concat(feats, axis=1) if feats else pd.DataFrame(index=base.index)
    combined = combined.sort_index().astype("float32").fillna(0.0)

    tr_out = combined.reindex(idx_train).fillna(0.0).astype("float32")
    te_out = combined.reindex(idx_test).fillna(0.0).astype("float32")

    tr_out.index = train_block.index
    te_out.index = test_block.index

    meta = {
        "enable": True,
        "lags": lags,
        "diff": bool(cfg.get("diff", False)),
        "pct_change": bool(cfg.get("pct_change", False)),
        "n_features": tr_out.shape[1],
    }
    return tr_out.copy(), te_out.copy(), meta


def binary_rolling_features_block(
    train_block: pd.DataFrame,
    test_block: pd.DataFrame,
    train_dates: Sequence[int],
    test_dates: Sequence[int],
    safe_cols: Sequence[str],
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not cfg.get("enable", True):
        return (
            pd.DataFrame(index=train_block.index),
            pd.DataFrame(index=test_block.index),
            {"enable": False, "windows": [], "prefixes": [], "n_binary_cols": 0, "n_features": 0},
        )

    prefixes = cfg.get("prefixes", ["D"])
    windows = cfg.get("windows", [5, 10, 20])
    try:
        windows = [int(window) for window in windows if int(window) > 0]
    except Exception:
        windows = []

    binary_cols = infer_binary_columns(safe_cols, prefixes)
    if not binary_cols or not windows:
        return (
            pd.DataFrame(index=train_block.index),
            pd.DataFrame(index=test_block.index),
            {
                "enable": bool(cfg.get("enable", True)),
                "windows": windows,
                "prefixes": prefixes,
                "n_binary_cols": len(binary_cols),
                "n_features": 0,
            },
        )

    idx_train = pd.Index(train_dates, name="date_id")
    idx_test = pd.Index(test_dates, name="date_id")

    base = pd.DataFrame(train_block[binary_cols].to_numpy(), columns=binary_cols, index=idx_train)
    base = base.sort_index()

    feats_tr: List[pd.DataFrame] = []
    feats_te: List[pd.DataFrame] = []
    for window in windows:
        roll_tr = base.rolling(window=window, min_periods=1).sum()
        roll_te = base.reindex(idx_test).rolling(window=window, min_periods=1).sum()
        feats_tr.append(roll_tr.add_suffix(f"_binroll{window}"))
        feats_te.append(roll_te.add_suffix(f"_binroll{window}"))

    tr_out = pd.concat(feats_tr, axis=1).astype("float32").reindex(idx_train).fillna(0.0)
    te_out = pd.concat(feats_te, axis=1).astype("float32").reindex(idx_test).fillna(0.0)

    tr_out.index = train_block.index
    te_out.index = test_block.index

    meta = {
        "enable": True,
        "prefixes": prefixes,
        "windows": windows,
        "n_binary_cols": len(binary_cols),
        "n_features": tr_out.shape[1],
    }
    return tr_out.copy(), te_out.copy(), meta


def prefix_aggregate_block(
    train_block: pd.DataFrame,
    test_block: pd.DataFrame,
    prefixes: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if not prefixes:
        return (
            pd.DataFrame(index=train_block.index),
            pd.DataFrame(index=test_block.index),
            {"enable": False, "prefixes": []},
        )

    agg_train: Dict[str, pd.Series] = {}
    agg_test: Dict[str, pd.Series] = {}

    for prefix in prefixes:
        cols = [col for col in train_block.columns if col.startswith(prefix)]
        if not cols:
            continue
        train_slice = train_block[cols]
        test_slice = test_block[cols]

        agg_train[f"{prefix}_mean"] = train_slice.mean(axis=1)
        agg_train[f"{prefix}_std"] = train_slice.std(axis=1, ddof=0)
        agg_train[f"{prefix}_min"] = train_slice.min(axis=1)
        agg_train[f"{prefix}_max"] = train_slice.max(axis=1)

        agg_test[f"{prefix}_mean"] = test_slice.mean(axis=1)
        agg_test[f"{prefix}_std"] = test_slice.std(axis=1, ddof=0)
        agg_test[f"{prefix}_min"] = test_slice.min(axis=1)
        agg_test[f"{prefix}_max"] = test_slice.max(axis=1)

    if not agg_train:
        return (
            pd.DataFrame(index=train_block.index),
            pd.DataFrame(index=test_block.index),
            {"enable": False, "prefixes": prefixes},
        )

    tr_df = pd.DataFrame(agg_train, index=train_block.index).astype("float32")
    te_df = pd.DataFrame(agg_test, index=test_block.index).astype("float32")

    return tr_df, te_df, {"enable": True, "prefixes": list(agg_train.keys())}


def pairwise_features_block(
    train_block: pd.DataFrame,
    test_block: pd.DataFrame,
    *,
    columns: Sequence[str],
    max_combinations: int,
    epsilon: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    cols = [col for col in columns if col in train_block.columns]
    if len(cols) < 2:
        return (
            pd.DataFrame(index=train_block.index),
            pd.DataFrame(index=test_block.index),
            {"enable": False, "columns": []},
        )

    combos = list(combinations(cols, 2))[:max(1, max_combinations)]
    train_out: Dict[str, pd.Series] = {}
    test_out: Dict[str, pd.Series] = {}

    eps = float(epsilon)
    for a, b in combos:
        diff_name = f"{a}_minus_{b}"
        ratio_name = f"{a}_div_{b}"
        train_out[diff_name] = train_block[a] - train_block[b]
        test_out[diff_name] = test_block[a] - test_block[b]

        train_den = np.where(np.abs(train_block[b]) < eps, np.sign(train_block[b]) * eps, train_block[b])
        test_den = np.where(np.abs(test_block[b]) < eps, np.sign(test_block[b]) * eps, test_block[b])
        train_out[ratio_name] = train_block[a] / train_den
        test_out[ratio_name] = test_block[a] / test_den

    train_df = pd.DataFrame(train_out, index=train_block.index)
    test_df = pd.DataFrame(test_out, index=test_block.index)
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    return train_df, test_df, {"enable": True, "columns": list(train_out.keys())}


def build_level1(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if "date_id" in train.columns:
        train = train.sort_values("date_id").reset_index(drop=True)
    if "date_id" in test.columns:
        test = test.sort_values("date_id").reset_index(drop=True)

    macro_lag_map = {
        "forward_returns": "lagged_forward_returns",
        "risk_free_rate": "lagged_risk_free_rate",
        "market_forward_excess_returns": "lagged_market_forward_excess_returns",
    }
    for src_col, lag_col in macro_lag_map.items():
        if src_col in train.columns and lag_col not in train.columns:
            shifted = train[src_col].shift(1)
            train[lag_col] = shifted.fillna(0.0)

    target = cfg["target"]
    safe_cols = select_safe_columns(train, test, cfg)

    X_base = train[["date_id"]].copy() if "date_id" in train.columns else pd.DataFrame(index=train.index)
    T_base = test[["date_id"]].copy() if "date_id" in test.columns else pd.DataFrame(index=test.index)
    if "is_scored" in train.columns:
        X_base["is_scored"] = train["is_scored"].values
    if target in train.columns:
        X_base[target] = train[target].values

    X_block, T_block, X_flags, T_flags = time_aware_impute_block(
        X=train,
        T=test,
        safe_cols=safe_cols,
        train_for_stats=train,
        cfg=cfg,
    )

    thresholds: Dict[str, Tuple[float, float]] = {}
    wz_cfg = cfg.get("winsorize", {"enable": True, "q": 0.01})
    if wz_cfg.get("enable", True):
        q = float(wz_cfg.get("q", 0.01))
        X_block, T_block, thresholds = winsorize_block(X_block, T_block, safe_cols, q=q)

    roll_cfg = cfg.get("rolling", {})
    X_roll, T_roll = rolling_features_block(
        src_train=X_block,
        src_test=T_block,
        safe_cols=safe_cols,
        windows=roll_cfg.get("windows", [5, 10, 20]),
        stats=roll_cfg.get("stats", ["mean", "std", "zscore"]),
    )

    lag_cfg = cfg.get("lag", {})
    X_lag = pd.DataFrame(index=X_block.index)
    T_lag = pd.DataFrame(index=T_block.index)
    lag_meta = {"enable": False, "lags": [], "n_features": 0}
    if "date_id" in train.columns and "date_id" in test.columns and lag_cfg.get("enable", True):
        X_lag, T_lag, lag_meta = lag_features_block(
            train_block=X_block,
            test_block=T_block,
            train_dates=train["date_id"].to_numpy(),
            test_dates=test["date_id"].to_numpy(),
            safe_cols=safe_cols,
            cfg=lag_cfg,
        )

    bin_cfg = cfg.get("binary_rolling", {})
    X_bin = pd.DataFrame(index=X_block.index)
    T_bin = pd.DataFrame(index=T_block.index)
    bin_meta = {"enable": False, "windows": [], "n_features": 0}
    if "date_id" in train.columns and "date_id" in test.columns and bin_cfg.get("enable", True):
        X_bin, T_bin, bin_meta = binary_rolling_features_block(
            train_block=X_block,
            test_block=T_block,
            train_dates=train["date_id"].to_numpy(),
            test_dates=test["date_id"].to_numpy(),
            safe_cols=safe_cols,
            cfg=bin_cfg,
        )

    agg_cfg = cfg.get("prefix_aggregates", {})
    X_agg = pd.DataFrame(index=X_block.index)
    T_agg = pd.DataFrame(index=T_block.index)
    agg_meta = {"enable": False, "prefixes": []}
    if agg_cfg.get("enable", True):
        prefixes = agg_cfg.get("prefixes", [])
        X_agg, T_agg, agg_meta = prefix_aggregate_block(
            X_block,
            T_block,
            prefixes,
        )

    pair_cfg = cfg.get("pairwise_diff_ratio", {})
    X_pair = pd.DataFrame(index=X_block.index)
    T_pair = pd.DataFrame(index=T_block.index)
    pair_meta = {"enable": False, "columns": []}
    if pair_cfg.get("enable", True):
        macro_candidates = [
            "lagged_forward_returns",
            "lagged_risk_free_rate",
            "lagged_market_forward_excess_returns",
            "risk_free_rate",
            "market_forward_excess_returns",
        ]
        macro_cols = [col for col in macro_candidates if col in X_block.columns]
        agg_mean_cols = [col for col in X_agg.columns if col.endswith("_mean")]
        base_train = pd.DataFrame(index=X_block.index)
        base_test = pd.DataFrame(index=T_block.index)
        if macro_cols:
            base_train[macro_cols] = X_block[macro_cols]
            base_test[macro_cols] = T_block[macro_cols]
        if agg_mean_cols:
            base_train = pd.concat([base_train, X_agg[agg_mean_cols]], axis=1)
            base_test = pd.concat([base_test, T_agg[agg_mean_cols]], axis=1)

        columns = base_train.columns.tolist()
        if len(columns) >= 2:
            X_pair, T_pair, pair_meta = pairwise_features_block(
                base_train,
                base_test,
                columns=columns,
                max_combinations=int(pair_cfg.get("max_combinations", 30)),
                epsilon=float(pair_cfg.get("epsilon", 1e-6)),
            )

    parts_tr = [X_base, X_block]
    parts_te = [T_base, T_block]
    if X_flags is not None:
        parts_tr.append(X_flags)
        parts_te.append(T_flags)
    if not X_roll.empty:
        parts_tr.append(X_roll)
        parts_te.append(T_roll)
    if not X_lag.empty:
        parts_tr.append(X_lag)
        parts_te.append(T_lag)
    if not X_bin.empty:
        parts_tr.append(X_bin)
        parts_te.append(T_bin)
    if not X_agg.empty:
        parts_tr.append(X_agg)
        parts_te.append(T_agg)
    if not X_pair.empty:
        parts_tr.append(X_pair)
        parts_te.append(T_pair)

    X_out = pd.concat(parts_tr, axis=1).copy()
    T_out = pd.concat(parts_te, axis=1).copy()

    meta = {
        "safe_cols": safe_cols,
        "n_features_total_train": X_out.shape[1],
        "n_features_total_test": T_out.shape[1],
        "winsorize": wz_cfg,
        "rolling": roll_cfg,
        "thresholds": thresholds,
        "lag": lag_meta,
        "binary_rolling": bin_meta,
        "prefix_aggregates": agg_meta,
        "pairwise_diff_ratio": pair_meta,
    }
    return X_out, T_out, meta


def build_version_manifest(train_path: Path, test_path: Path, cfg: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    version = cfg_hash(
        {
            "files": {"train": sha256_file(train_path), "test": sha256_file(test_path)},
            "level1_cfg": cfg,
        }
    )
    manifest = {"cfg": cfg, "version": version, "meta": meta}
    return version, manifest


__all__ = [
    "DEFAULT_CFG",
    "build_level1",
    "build_version_manifest",
    "downcast_float32",
    "ensure_dir",
    "read_yaml_if_any",
    "select_safe_columns",
    "sha256_file",
]
