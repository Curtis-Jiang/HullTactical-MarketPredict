import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

d = Path("data/processed/htmp_v04bc6642")
tr = pd.read_parquet(d/"train_l1.parquet")
te = pd.read_parquet(d/"test_l1.parquet")
target = "forward_returns"

drop_cols = {"date_id","is_scored",target}
feat_cols = [c for c in tr.columns if c not in drop_cols]
X, y = tr[feat_cols].astype("float32"), tr[target].astype("float32")
scored = (tr["is_scored"]==1) if "is_scored" in tr.columns else np.ones(len(tr), bool)

print("y stats (all):", y.describe().to_dict())
print("y stats (scored):", y[scored].describe().to_dict())

# 1) “均值预测”的基线（一定得到 R2=0）
mean_pred = np.full(scored.sum(), y[scored].mean(), dtype="float32")
print("Sanity R2 (predict mean) =",
      r2_score(y[scored], mean_pred))

# 2) 预测值/真实值尺度检查
#   如果你之前的 OOF 接近常数 0 或明显偏离 1e-4 数量级，基本确定有问题
def peek_scale(a, name):
    arr = np.asarray(a, dtype="float64")
    print(f"{name}: mean={arr.mean():.6g}, std={arr.std():.6g}, min={arr.min():.6g}, max={arr.max():.6g}")
peek_scale(y, "y(all)")
peek_scale(y[scored], "y(scored)")

# 3) 折内特征“几乎常数”排查（每折只看训练子集）
tscv = TimeSeriesSplit(n_splits=5)
bad_cols = set()
for k,(tr_idx, va_idx) in enumerate(tscv.split(X),1):
    nunq = X.iloc[tr_idx].nunique(dropna=False)
    bad = nunq[nunq<=1].index.tolist()
    if bad:
        print(f"[fold{k}] near-constant features in TRAIN:", len(bad))
        bad_cols.update(bad)
print("near-constant-in-train (union):", len(bad_cols))

# 4) 用 Ridge-only 做一个 OOF，看看是否也极差（排除 LGBM 特有问题）
ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=42))
oof_r = np.full(len(y), np.nan, dtype="float32")
for tr_idx, va_idx in tscv.split(X):
    m = ridge.__class__(**ridge.get_params())  # clone
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    oof_r[va_idx] = m.predict(X.iloc[va_idx])
print("Ridge OOF R2 (scored-only):", r2_score(y[scored], oof_r[scored]))

# 5) （可选）验证列对齐
assert X.shape[1] == te[feat_cols].shape[1], "train/test feat cols mismatch"