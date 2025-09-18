#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust training for HTMP:
- Fix sklearn FutureWarning via rmse() wrapper.
- Robust z-score handling: clip all *_rz{w} features to [-z_clip, z_clip].
- TimeSeriesSplit OOF evaluation on "covered & scored" samples only.
- Fold-wise near-constant feature filtering (union across folds).
- LGB + Ridge ensemble with NaN/Inf sanitization and prediction clipping.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
# sklearn RMSE wrapper (future-proof)
try:
    from sklearn.metrics import root_mean_squared_error as rmse
except ImportError:  # sklearn < 1.4
    from sklearn.metrics import mean_squared_error
    def rmse(y_true, y_pred):  # noqa: N802
        return mean_squared_error(y_true, y_pred, squared=False)

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone


def set_seed(seed: int = 42):
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_latest_processed(processed_root: Path) -> Path:
    cands = sorted(processed_root.glob("htmp_v*"))
    if not cands:
        raise FileNotFoundError(f"No processed dir found in {processed_root}. Run build_features.py first.")
    return cands[-1]


def filter_near_constant_by_folds(X: pd.DataFrame, n_splits: int, var_eps: float = 1e-12):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    bad_union = set()
    for k, (tr_idx, _) in enumerate(tscv.split(X), 1):
        stds = X.iloc[tr_idx].std(axis=0).fillna(0.0)
        bad = stds.index[(stds <= var_eps)].tolist()
        if bad:
            print(f"[fold{k}] drop near-constant in TRAIN: {len(bad)}")
        bad_union.update(bad)
    return bad_union


def robust_clip_zscore_block(df: pd.DataFrame, z_clip: float) -> pd.DataFrame:
    """Clip *_rz{w} zscore features to [-z_clip, z_clip] in-place-safe style (returns a new DataFrame view)."""
    z_cols = [c for c in df.columns if c.endswith("_rz5") or c.endswith("_rz10") or c.endswith("_rz20")]
    if not z_cols:
        return df
    arr = df.loc[:, z_cols].to_numpy(copy=False)
    np.clip(arr, -float(z_clip), float(z_clip), out=arr)
    # df shares same memory with arr via to_numpy(copy=False); no extra assignment needed
    return df


def main():
    ap = argparse.ArgumentParser(description="Robust training with stable OOF and z-score clipping.")
    ap.add_argument("--processed_root", type=str, default="data/processed")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--z_clip", type=float, default=8.0, help="Clip range for *_rz{w} zscore features")
    ap.add_argument("--pred_clip_sigma", type=float, default=5.0, help="Clip predictions to mean ± k*std of y")
    ap.add_argument("--drop_zscore", action="store_true", help="Drop all *_rz{w} features entirely")
    ap.add_argument("--keep_zscore", action="store_true", help="Keep zscore features (default) with clipping")
    ap.add_argument("--w1", type=float, default=0.7, help="Weight for LGB")
    ap.add_argument("--w2", type=float, default=0.3, help="Weight for Ridge")
    ap.add_argument("--out_sub", type=str, default="submission.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.metrics._regression")

    processed_root = Path(args.processed_root)
    ver_dir = load_latest_processed(processed_root)
    print(f"[info] using processed dir: {ver_dir}")

    # --- Load features ---
    tr = pd.read_parquet(ver_dir / "train_l1.parquet")
    te = pd.read_parquet(ver_dir / "test_l1.parquet")

    target = "forward_returns"
    drop_cols = {"date_id", "is_scored", target}
    feat_cols = [c for c in tr.columns if c not in drop_cols]

    # zscore handling: drop or clip
    if args.drop_zscore and not args.keep_zscore:
        feat_cols = [c for c in feat_cols if not (c.endswith("_rz5") or c.endswith("_rz10") or c.endswith("_rz20"))]

    X = tr[feat_cols].astype("float32").reset_index(drop=True)
    y = tr[target].astype("float32").reset_index(drop=True)
    X_test = te[feat_cols].astype("float32")
    scored = (tr["is_scored"] == 1) if "is_scored" in tr.columns else np.ones(len(tr), dtype=bool)

    # Robust zscore clip (if we keep zscore features)
    if not args.drop_zscore:
        X = robust_clip_zscore_block(X, args.z_clip)
        X_test = robust_clip_zscore_block(X_test, args.z_clip)

    # Sanity checks
    assert np.isfinite(X.to_numpy()).all(), "Train features contain NaN/Inf"
    assert np.isfinite(X_test.to_numpy()).all(), "Test features contain NaN/Inf"
    assert np.isfinite(y.to_numpy()).all(), "Target contains NaN/Inf"
    print("y stats:", y.describe().to_dict())

    # --- Fold-wise near-constant filtering (union) ---
    bad_union = filter_near_constant_by_folds(X, n_splits=args.n_splits, var_eps=1e-12)
    if bad_union:
        keep = [c for c in feat_cols if c not in bad_union]
        print(f"[filter] drop union near-constant: {len(bad_union)} → keep {len(keep)}")
        feat_cols = keep
        X = X[feat_cols]
        X_test = X_test[feat_cols]

    # --- Models ---
    lgb_params = dict(
        n_estimators=3000, learning_rate=0.01, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.0, reg_lambda=0.0,
        min_gain_to_split=0.0, min_data_in_leaf=20,
        random_state=args.seed,
    )
    ridge_tpl = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0, random_state=args.seed))

    oof = np.full(len(y), np.nan, dtype="float32")
    preds = np.zeros(len(X_test), dtype="float32")
    covered = np.zeros(len(y), dtype=bool)

    # Prediction clip range: mean ± k*std
    clip_low = float(y.mean() - args.pred_clip_sigma * y.std())
    clip_high = float(y.mean() + args.pred_clip_sigma * y.std())
    print(f"[info] prediction clip range: [{clip_low:.6g}, {clip_high:.6g}]")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    for k, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        covered[va_idx] = True

        # Safety checks per fold
        assert np.isfinite(X_tr.to_numpy()).all() and np.isfinite(X_va.to_numpy()).all()
        assert np.isfinite(y_tr.to_numpy()).all() and np.isfinite(y_va.to_numpy()).all()

        m1 = lgb.LGBMRegressor(**lgb_params)
        m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
               callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])

        m2 = clone(ridge_tpl); m2.fit(X_tr, y_tr)

        p1v = m1.predict(X_va); p2v = m2.predict(X_va)
        p1t = m1.predict(X_test); p2t = m2.predict(X_test)

        # sanitize
        def sanitize(a):
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            return np.clip(a, clip_low, clip_high)

        p1v, p2v, p1t, p2t = map(sanitize, (p1v, p2v, p1t, p2t))

        # fold stats
        def stats(arr): return (float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)), float(np.std(arr)))
        print(f"[fold{k}] y_va std={y_va.std():.6g} | p1v(min/mean/max/std)={stats(p1v)} | p2v={stats(p2v)}")

        w1, w2 = args.w1, args.w2
        oof[va_idx] = w1 * p1v + w2 * p2v
        preds += (w1 * p1t + w2 * p2t) / args.n_splits

    # Evaluate on covered & scored only
    mask = covered & scored
    if not mask.any():
        raise RuntimeError("No covered & scored samples to evaluate!")
    if np.isnan(oof[mask]).any():
        idx_bad = np.where(np.isnan(oof) & mask)[0]
        print(f"[warn] OOF NaN remains at indices (masked): {idx_bad[:10]} ... total {len(idx_bad)}; fill with mean.")
        oof[idx_bad] = y.mean()

    print(f"Covered samples: {mask.sum()} / {len(y)}")
    print("OOF R2 (scored & covered):", r2_score(y[mask], oof[mask]))
    print("OOF RMSE (scored & covered):", rmse(y[mask], oof[mask]))
    print("OOF MAE  (scored & covered):", mean_absolute_error(y[mask], oof[mask]))

    # Submission
    sub = pd.DataFrame({"date_id": te["date_id"], "prediction": preds})
    out_path = Path(args.out_sub)
    sub.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} | rows={len(sub)}")


if __name__ == "__main__":
    main()