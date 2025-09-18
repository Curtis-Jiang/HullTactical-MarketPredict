import pandas as pd, numpy as np
from pathlib import Path

DATA = Path('../data/raw')  # 改成你的路径
train = pd.read_csv(DATA/'train.csv')
test  = pd.read_csv(DATA/'test.csv')

print("Shapes:", train.shape, test.shape)
print("\nTrain head:\n", train.head(3))
print("\nDtypes:\n", train.dtypes.value_counts())

# 1) 基本列集
all_cols  = train.columns.tolist()
num_cols  = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])]
obj_cols  = [c for c in train.columns if train[c].dtype=='object']
print(f"\nNumeric cols: {len(num_cols)}, Object cols: {len(obj_cols)}")

# 2) 目标/辅助列猜测
possible_y = ['forward_returns','target','y']
target = next((c for c in possible_y if c in train.columns), None)
print("Target guess:", target)

# 3) 缺失率
na_train = train.isna().mean().sort_values(ascending=False)
na_test  = test.isna().mean().sort_values(ascending=False)
na_df = pd.DataFrame({'na_train': na_train, 'na_test': na_test}).fillna(1.0)
na_df.to_csv('eda_missing_rates.csv')
print("\nTop missing (train):\n", na_train.head(10))

# 4) 常量列 / 近零方差
def near_zero_var(s, thr=1e-9):
    if s.notna().sum()==0: return True
    v = s.var()
    return (pd.isna(v)) or (v < thr)

const_cols = [c for c in num_cols if train[c].nunique(dropna=True)<=1]
nzv_cols   = [c for c in num_cols if near_zero_var(train[c])]
print("\nConstant cols:", const_cols[:10])
print("Near-zero-variance cols:", nzv_cols[:10])

# 5) date_id 连续性 & is_scored 分布
if 'date_id' in train.columns:
    train = train.sort_values('date_id').reset_index(drop=True)
    test  = test.sort_values('date_id').reset_index(drop=True)
    gaps = np.diff(train['date_id'].unique())
    print("\nDate_id gaps (train) – unique deltas:", np.unique(gaps)[:10])

if 'is_scored' in train.columns:
    print("\nTrain is_scored value counts:\n", train['is_scored'].value_counts())

# 6) train vs test 漂移（均值/标准差对比）
inter_cols = [c for c in num_cols if c in test.columns and c not in {target}]
drift_rows = []
for c in inter_cols:
    a, b = train[c], test[c]
    drift_rows.append({
        'col': c,
        'mean_train': np.nanmean(a), 'std_train': np.nanstd(a),
        'mean_test':  np.nanmean(b), 'std_test':  np.nanstd(b),
        'delta_mean': np.nanmean(a) - np.nanmean(b),
        'ratio_std':  (np.nanstd(a)+1e-9)/(np.nanstd(b)+1e-9),
    })
drift = pd.DataFrame(drift_rows).sort_values('delta_mean', key=lambda s: s.abs(), ascending=False)
drift.to_csv('eda_train_test_drift.csv', index=False)
print("\nTop drift:\n", drift.head(10)[['col','delta_mean','ratio_std']])

# 7) 只在 train 存在/只在 test 存在 的列
only_train = sorted(list(set(train.columns)-set(test.columns)))
only_test  = sorted(list(set(test.columns)-set(train.columns)))
print("\nOnly in train:", only_train[:20])
print("Only in test:", only_test[:20])

# 8) 与目标的线性相关（快速感知）
if target is not None:
    corr = train[inter_cols+[target]].corr()[target].drop(target).sort_values(ascending=False)
    print("\nTop +corr:\n", corr.head(10))
    print("\nTop -corr:\n", corr.tail(10))