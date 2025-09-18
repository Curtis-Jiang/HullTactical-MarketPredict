import pandas as pd
from pathlib import Path
d = Path("data/processed/htmp_v04bc6642")
tr = pd.read_parquet(d/"train_l1.parquet")
te = pd.read_parquet(d/"test_l1.parquet")

target = "forward_returns"
drop_cols = {"date_id","is_scored",target}

feat_cols_tr = [c for c in tr.columns if c not in drop_cols]
feat_cols_te = [c for c in te.columns if c not in drop_cols]

# 1) 列对齐检查
assert feat_cols_tr == feat_cols_te, f"mismatch: {set(feat_cols_tr)^set(feat_cols_te)}"

# 2) 缺失/Inf 检查
assert not tr[feat_cols_tr].isna().any().any(), "NaN in train features"
assert not te[feat_cols_te].isna().any().any(), "NaN in test features"
import numpy as np
assert np.isfinite(tr[feat_cols_tr].to_numpy()).all(), "Inf in train"
assert np.isfinite(te[feat_cols_te].to_numpy()).all(), "Inf in test"

# 3) dtype 检查（应主要是 float32 / int8）
print(tr[feat_cols_tr].dtypes.value_counts().head())
print("OK ✓")