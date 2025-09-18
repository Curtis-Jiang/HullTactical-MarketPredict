# === HTMP Baseline v2: train/test 特征对齐且排除泄露 ===
import numpy as np, pandas as pd
from pathlib import Path

DATA = Path('./data/raw')
train = pd.read_csv(DATA/'train.csv')
test  = pd.read_csv(DATA/'test.csv')


# 0) 排序（时间序列很重要）
if 'date_id' in train.columns: train = train.sort_values('date_id').reset_index(drop=True)
if 'date_id' in test.columns:  test  = test.sort_values('date_id').reset_index(drop=True)

# 1) 目标列自适应（优先 forward_returns；你可按实际列名调整）
possible_y = ['forward_returns','target','y']
target = next((c for c in possible_y if c in train.columns), None)
if target is None:
    raise ValueError(f"找不到目标列。train 列为：{list(train.columns)[:12]} ... 请把真实目标名加入 possible_y。")

# 2) 构建数值特征（先从 train 取，再与 test 求交集），排除非特征
drop_cols = {target, 'date_id', 'is_scored', 'row_id', 'id', 'index'}
train_num = [c for c in train.columns
             if c not in drop_cols and pd.api.types.is_numeric_dtype(train[c])]
# 与 test 求交集，避免 KeyError
num_cols = [c for c in train_num if c in test.columns]

# 调试输出：哪些“只在 train 存在”的数值列被丢弃（比如 risk_free_rate 等）
missing_in_test = sorted(list(set(train_num) - set(num_cols)))
if missing_in_test:
    print("以下列仅出现在 train、在 test 中缺失（因此被丢弃）:")
    print(missing_in_test[:30], f"... 共 {len(missing_in_test)} 列")

X = train[num_cols].copy()
y = train[target].astype('float32').copy()
X_test = test[num_cols].copy()

# 3) 缺失值处理（中位数填充）
for df in (X, X_test):
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

# 4) 时间序列CV + 两模型简单集成
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import lightgbm as lgb

tscv = TimeSeriesSplit(n_splits=5)
ridge = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=1.0, random_state=42))
lgb_params = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.2, reg_lambda=0.6,
    min_child_samples=50, random_state=42, n_jobs=-1
)

oof = np.zeros(len(X), dtype='float32')
preds = np.zeros(len(X_test), dtype='float32')

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    m1 = lgb.LGBMRegressor(**lgb_params)
    m1.fit(X_tr, y_tr, eval_set=[(X_val,y_val)], eval_metric='rmse',
           callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
    p1_val = m1.predict(X_val, num_iteration=m1.best_iteration_)
    p1_tst = m1.predict(X_test, num_iteration=m1.best_iteration_)

    m2 = ridge
    m2.fit(X_tr, y_tr)
    p2_val = m2.predict(X_val)
    p2_tst = m2.predict(X_test)

    w1, w2 = 0.7, 0.3
    oof[val_idx] = w1*p1_val + w2*p2_val
    preds      += (w1*p1_tst + w2*p2_tst)/tscv.n_splits

    print(f"Fold {fold} R2: {r2_score(y_val, oof[val_idx]):.5f}")

print(f"OOF R2: {r2_score(y, oof):.5f}")

# 5) 生成提交：常见为 'date_id' + 'prediction'
PRED_NAME = 'prediction'  # 若Kaggle页面提示列名不同，就改成要求的名字
if 'date_id' in test.columns:
    sub = pd.DataFrame({'date_id': test['date_id'].values, PRED_NAME: preds.astype('float32')})
else:
    sub = pd.DataFrame({PRED_NAME: preds.astype('float32')})

sub.to_csv('submission.csv', index=False)
print("Saved submission.csv")
print(sub.head())