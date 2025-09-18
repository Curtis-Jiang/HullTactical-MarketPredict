#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Level-1 features for HTMP.
- 仅用“预测时可获得”的安全特征：train∩test 的数值列，排除 {date_id,is_scored,forward_returns,...}
- 时间感知填补：ffill → rolling mean → 训练集列中位数（块级操作）
- Winsorize 截尾（块级 clip）
- 纯历史滚动统计：mean/std/zscore，分别在 train/test 内部独立 rolling（不跨边界）
- 输出到 data/processed/htmp_v{hash}/{train_l1.parquet,test_l1.parquet,manifest.json}

注意：
- 全过程避免逐列 df[c] = ...，使用一次性 concat/clip，杜绝 DataFrame 碎片化告警。
"""

import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

# -------------------------------
# 默认配置（无 yml 也能跑）
# -------------------------------
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
}

# -------------------------------
# 工具函数
# -------------------------------
def read_yaml_if_any(path: Path) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # 没装 pyyaml 或解析失败，就用空覆盖
        return {}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def cfg_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:8]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def downcast_float32(df: pd.DataFrame) -> pd.DataFrame:
    # 只对浮点列降位
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype("float32")
    return df

# -------------------------------
# 核心逻辑
# -------------------------------
def select_safe_columns(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    drop = set(cfg["drop_cols"])
    num_train = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
    safe_cols = [c for c in num_train if c in test.columns and c not in drop]
    return safe_cols

def time_aware_impute_block(
    X: pd.DataFrame, T: pd.DataFrame, safe_cols: List[str], train_for_stats: pd.DataFrame, cfg: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回：X_block、T_block、X_flags、T_flags（flags 可能为 None）
    全部为块级操作：不做逐列赋值。
    """
    imp = cfg.get("impute", {})
    add_flags = bool(imp.get("add_isna_flags", True))
    use_ffill = bool(imp.get("forward_fill", True))
    roll_cfg = imp.get("rolling_mean_fallback", {"enable": True, "window": 20})
    use_roll = bool(roll_cfg.get("enable", True))
    w = int(roll_cfg.get("window", 20))
    use_global_median = bool(imp.get("global_median_fallback", True))

    # 取块副本
    X_block = X[safe_cols].copy()
    T_block = T[safe_cols].copy()

    # 缺失指示器（一次性）
    X_flags = X_block.isna().astype("int8").add_suffix("_isna") if add_flags else None
    T_flags = T_block.isna().astype("int8").add_suffix("_isna") if add_flags else None

    # 前向填充（各自内部）
    if use_ffill:
        X_block = X_block.ffill()
        T_block = T_block.ffill()

    # 滚动均值回补（各自内部）
    if use_roll:
        X_mean = X_block.rolling(w, min_periods=1).mean()
        T_mean = T_block.rolling(w, min_periods=1).mean()
        X_block = X_block.fillna(X_mean)
        T_block = T_block.fillna(T_mean)

    # 兜底：用 train 的列中位数（避免看 test 的“未来”）
    if use_global_median:
        med = train_for_stats[safe_cols].median()
        X_block = X_block.fillna(med)
        T_block = T_block.fillna(med)

    # 降位
    X_block = downcast_float32(X_block)
    T_block = downcast_float32(T_block)

    # 一次性 copy，确保去碎片化
    X_block = X_block.copy()
    T_block = T_block.copy()
    if X_flags is not None:
        X_flags = X_flags.copy()
        T_flags = T_flags.copy()

    return X_block, T_block, X_flags, T_flags

def winsorize_block(X: pd.DataFrame, T: pd.DataFrame, safe_cols: List[str], q: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    对 X/T 的 safe_cols 执行按列分位数截尾（使用 X 的分位数），块级 clip。
    返回：裁剪后的 X、T 以及阈值字典（用于 manifest）
    """
    # 用当前 X 的分位数计算阈值
    lo = X[safe_cols].quantile(q)
    hi = X[safe_cols].quantile(1 - q)
    # Ensure thresholds are float32 to match downcasted dataframes
    lo32 = lo.astype("float32")
    hi32 = hi.astype("float32")

    # pandas>=1.5 支持 axis=1 按列对齐 clip；老版本回退逐列（很快）
    try:
        X.loc[:, safe_cols] = (
            X.loc[:, safe_cols]
            .clip(lower=lo32, upper=hi32, axis=1)
            .astype("float32")
        )
        T.loc[:, safe_cols] = (
            T.loc[:, safe_cols]
            .clip(lower=lo32, upper=hi32, axis=1)
            .astype("float32")
        )
    except TypeError:
        for c in safe_cols:
            X[c] = X[c].clip(lo32[c], hi32[c]).astype("float32")
            T[c] = T[c].clip(lo32[c], hi32[c]).astype("float32")

    # 记录阈值到字典
    thr = {f"{c}": [float(lo[c]), float(hi[c])] for c in safe_cols}
    return X, T, thr

def rolling_features_block(src_train: pd.DataFrame, src_test: pd.DataFrame, safe_cols: List[str], windows: List[int], stats: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    在 train/test 内部分别 rolling（不跨边界），生成 mean/std/zscore。
    返回：X_roll、T_roll（可能为空 DataFrame，但不会报错）
    """
    outs_tr = []
    outs_te = []
    need_mean = "mean" in stats or "zscore" in stats
    need_std  = "std" in stats or "zscore" in stats
    for w in windows:
        mean_tr = src_train[safe_cols].rolling(w, min_periods=w).mean() if need_mean else None
        std_tr  = src_train[safe_cols].rolling(w, min_periods=w).std()  if need_std  else None
        mean_te = src_test[safe_cols].rolling(w, min_periods=w).mean()  if need_mean else None
        std_te  = src_test[safe_cols].rolling(w, min_periods=w).std()   if need_std  else None

        # 拼接 mean/std
        if "mean" in stats:
            outs_tr.append(mean_tr.add_suffix(f"_rm{w}"))
            outs_te.append(mean_te.add_suffix(f"_rm{w}"))
        if "std" in stats:
            outs_tr.append(std_tr.add_suffix(f"_rs{w}"))
            outs_te.append(std_te.add_suffix(f"_rs{w}"))

        # zscore
        if "zscore" in stats:
            z_tr = (src_train[safe_cols] - mean_tr) / std_tr.replace(0, np.nan)
            z_te = (src_test[safe_cols]  - mean_te) / std_te.replace(0, np.nan)
            outs_tr.append(z_tr.add_suffix(f"_rz{w}"))
            outs_te.append(z_te.add_suffix(f"_rz{w}"))

    X_roll = pd.concat(outs_tr, axis=1) if outs_tr else pd.DataFrame(index=src_train.index)
    T_roll = pd.concat(outs_te, axis=1) if outs_te else pd.DataFrame(index=src_test.index)

    # 缺失置 0（前 w-1 天没有足够窗口）
    if not X_roll.empty:
        X_roll = X_roll.fillna(0.0).astype("float32").copy()
    if not T_roll.empty:
        T_roll = T_roll.fillna(0.0).astype("float32").copy()

    return X_roll, T_roll

def build_level1(train: pd.DataFrame, test: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    # 0) 排序
    if "date_id" in train.columns:
        train = train.sort_values("date_id").reset_index(drop=True)
    if "date_id" in test.columns:
        test = test.sort_values("date_id").reset_index(drop=True)

    target = cfg["target"]
    safe_cols = select_safe_columns(train, test, cfg)

    # 基础框架（只携带必要列，便于最后拼接）
    X_base = train[["date_id"]].copy() if "date_id" in train.columns else pd.DataFrame(index=train.index)
    T_base = test[["date_id"]].copy()  if "date_id" in test.columns  else pd.DataFrame(index=test.index)
    if "is_scored" in train.columns:
        X_base["is_scored"] = train["is_scored"].values
    if target in train.columns:
        X_base[target] = train[target].values

    # 1) 时间感知填补（块级）+ 缺失指示器
    X_block, T_block, X_flags, T_flags = time_aware_impute_block(
        X=train, T=test, safe_cols=safe_cols, train_for_stats=train, cfg=cfg
    )

    # 2) Winsorize（块级）
    wz = cfg.get("winsorize", {"enable": True, "q": 0.01})
    thr_dict = {}
    if wz.get("enable", True):
        q = float(wz.get("q", 0.01))
        X_block, T_block, thr_dict = winsorize_block(X_block, T_block, safe_cols, q=q)

    # 3) 滚动统计（块级）
    windows = cfg.get("rolling", {}).get("windows", [5, 10, 20])
    stats   = cfg.get("rolling", {}).get("stats", ["mean", "std", "zscore"])
    X_roll, T_roll = rolling_features_block(
        src_train=X_block, src_test=T_block, safe_cols=safe_cols, windows=windows, stats=stats
    )

    # 4) 汇总一次性 concat（避免碎片化）
    parts_tr = [X_base, X_block]
    parts_te = [T_base, T_block]
    if X_flags is not None:
        parts_tr.append(X_flags)
        parts_te.append(T_flags)
    if not X_roll.empty:
        parts_tr.append(X_roll)
        parts_te.append(T_roll)

    X_out = pd.concat(parts_tr, axis=1)
    T_out = pd.concat(parts_te, axis=1)

    # 最终再 copy 一次，确保去碎片化
    X_out = X_out.copy()
    T_out = T_out.copy()

    # 元信息
    meta = {
        "safe_cols": safe_cols,
        "n_features_total_train": X_out.shape[1],
        "n_features_total_test": T_out.shape[1],
        "winsorize": wz,
        "rolling": {"windows": windows, "stats": stats},
        "thresholds": thr_dict,  # winsorize 的阈值，可用于复现
    }
    return X_out, T_out, meta

# -------------------------------
# 主程序
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build Level-1 features (block ops, no fragmentation).")
    parser.add_argument("--train", type=str, default="data/raw/train.csv")
    parser.add_argument("--test",  type=str, default="data/raw/test.csv")
    parser.add_argument("--outdir", type=str, default="data/processed")
    parser.add_argument("--config", type=str, default=None, help="Optional: configs/features.yml")
    args = parser.parse_args()

    RAW_TRAIN = Path(args.train)
    RAW_TEST  = Path(args.test)
    OUT_ROOT  = Path(args.outdir)

    # 读配置（默认 + yml 覆盖）
    cfg = DEFAULT_CFG.copy()
    yml_cfg = read_yaml_if_any(Path(args.config)) if args.config else {}
    # 浅合并（简单够用）
    for k, v in yml_cfg.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    # 读数据
    train = pd.read_csv(RAW_TRAIN)
    test  = pd.read_csv(RAW_TEST)

    # 版本号：原始文件指纹 + Level-1 cfg
    ver = cfg_hash({
        "files": {"train": sha256_file(RAW_TRAIN), "test": sha256_file(RAW_TEST)},
        "level1_cfg": cfg,
    })
    out_dir = OUT_ROOT / f"htmp_v{ver}"
    ensure_dir(out_dir)

    tr_pq = out_dir / "train_l1.parquet"
    te_pq = out_dir / "test_l1.parquet"
    manifest = out_dir / "manifest.json"

    if tr_pq.exists() and te_pq.exists() and manifest.exists():
        print(f"[cache] already built at {out_dir}")
        return

    # 构建
    X, T, meta = build_level1(train, test, cfg)

    # 写 parquet（优先 pyarrow）
    engine = "pyarrow"
    try:
        X.to_parquet(tr_pq, index=False, engine=engine)
        T.to_parquet(te_pq, index=False, engine=engine)
    except Exception:
        # 回退
        X.to_parquet(tr_pq, index=False)
        T.to_parquet(te_pq, index=False)

    # manifest
    manifest.write_text(json.dumps({"cfg": cfg, "version": ver, "meta": meta}, indent=2), encoding="utf-8")

    print(f"[ok] saved Level-1 features to: {out_dir}")
    print(f"  train_l1.parquet: {X.shape}  test_l1.parquet: {T.shape}")
    print(f"  safe_cols: {len(meta['safe_cols'])}  windows: {meta['rolling']['windows']}  stats: {meta['rolling']['stats']}")

if __name__ == "__main__":
    main()