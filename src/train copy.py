import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# === 1) 读特征 ===
ver_dir = max(Path("data/processed").glob("htmp_v*"))
tr = pd.read_parquet(ver_dir/"train_l1.parquet")
te = pd.read_parquet(ver_dir/"test_l1.parquet")

target = "forward_returns"
drop_cols = {"date_id","is_scored",target}
feat_cols = [c for c in tr.columns if c not in drop_cols]

# 先禁用 zscore 特征（等稳定后再逐步放开）
feat_cols = [c for c in feat_cols if not (c.endswith("_rz5") or c.endswith("_rz10") or c.endswith("_rz20"))]

X = tr[feat_cols].astype("float32").reset_index(drop=True)
y = tr[target].astype("float32").reset_index(drop=True)
X_test = te[feat_cols].astype("float32")
scored = (tr["is_scored"]==1) if "is_scored" in tr.columns else np.ones(len(tr), dtype=bool)

# 基本健康检查
assert np.isfinite(X.to_numpy()).all(), "Train features contain NaN/Inf"
assert np.isfinite(X_test.to_numpy()).all(), "Test features contain NaN/Inf"
assert np.isfinite(y.to_numpy()).all(), "Target contains NaN/Inf"

print("y stats:", y.describe().to_dict())

# === 2) 统一过滤“折内近乎常数”的列（每折一致）===
tscv = TimeSeriesSplit(n_splits=5)
bad_union = set()
var_eps = 1e-12
for k, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    stds = X.iloc[tr_idx].std(axis=0).fillna(0.0)
    bad = stds.index[(stds <= var_eps)].tolist()
    if bad:
        print(f"[fold{k}] drop near-constant in TRAIN: {len(bad)}")
    bad_union.update(bad)
if bad_union:
    keep = [c for c in feat_cols if c not in bad_union]
    print(f"[filter] drop union near-constant: {len(bad_union)} → keep {len(keep)}")
    feat_cols = keep
    X = X[feat_cols]; X_test = X_test[feat_cols]

# === 3) 模型、容错、评估 ===
lgb_params = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.0, reg_lambda=0.0,
    min_gain_to_split=0.0, min_data_in_leaf=20,
    random_state=42
)
ridge_tpl = make_pipeline(StandardScaler(with_mean=True), Ridge(alpha=1.0, random_state=42))

oof = np.full(len(y), np.nan, dtype="float32")
preds = np.zeros(len(X_test), dtype="float32")
covered = np.zeros(len(y), dtype=bool)   # 记录哪些索引“当过验证集”

# 夹紧阈值：宽容但合理（±5σ）
clip_low  = float(y.mean() - 5*y.std())
clip_high = float(y.mean() + 5*y.std())
print(f"prediction clip range: [{clip_low:.6g}, {clip_high:.6g}]")

for k, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    covered[va_idx] = True

    m1 = lgb.LGBMRegressor(**lgb_params)
    m1.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse",
           callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])

    m2 = clone(ridge_tpl); m2.fit(X_tr, y_tr)

    p1v = m1.predict(X_va); p2v = m2.predict(X_va)
    p1t = m1.predict(X_test); p2t = m2.predict(X_test)

    # 防 NaN/Inf + 夹紧
    def sanitize(a):
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(a, clip_low, clip_high)
    p1v, p2v, p1t, p2t = map(sanitize, (p1v, p2v, p1t, p2t))

    # 折内监控
    def stats(arr): return (float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)), float(np.std(arr)))
    print(f"[fold{k}] y_va std={y_va.std():.6g} | p1v(min/mean/max/std)={stats(p1v)} | p2v={stats(p2v)}")

    w1, w2 = 0.7, 0.3
    oof[va_idx] = w1*p1v + w2*p2v
    preds      += (w1*p1t + w2*p2t) / tscv.n_splits

# 评估口径：只在“被覆盖 & scored”的样本上算 R²（正确的 OOF 定义）
mask = covered & scored
if not mask.any():
    raise RuntimeError("No covered scored samples to evaluate!")
if np.isnan(oof[mask]).any():
    idx_bad = np.where(np.isnan(oof) & mask)[0]
    print(f"[warn] OOF NaN remains at indices (masked): {idx_bad[:10]} ... total {len(idx_bad)}; fill with mean.")
    oof[idx_bad] = y.mean()

print(f"Covered samples: {mask.sum()} / {len(y)}")
print("OOF R2 (scored & covered):", r2_score(y[mask], oof[mask]))
print("OOF RMSE (scored & covered):", mean_squared_error(y[mask], oof[mask], squared=False))
print("OOF MAE  (scored & covered):", mean_absolute_error(y[mask], oof[mask]))

# 导出提交
sub = pd.DataFrame({"date_id": te["date_id"], "prediction": preds})
sub.to_csv("submission.csv", index=False)
print("[ok] wrote submission.csv")