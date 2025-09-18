# HullTactical-MarketPredict

基于 Kaggle Hull Tactical Market Prediction 数据集的时间序列建模项目，涵盖特征构建、模型训练与推理全流程。

## 快速开始

1. **准备环境**
   ```bash
   pip install -r requirements.txt
   ```

2. **构建特征**（默认配置为 `configs/features.yml`）
   ```bash
   python src/build_features.py --config configs/features.yml
   ```

3. **训练模型并记录实验**
   ```bash
   python src/train.py --experiment_name lgb_ridge_baseline --out_sub data/submissions/submission.csv
   ```
   - 训练脚本会自动在 `logs/experiments/<timestamp>_lgb_ridge_<hash>/` 下保存配置、指标、模型文件、OOF 预测以及 `submission.csv`。

4. **基于保存的实验进行推理**
   ```bash
   python src/predict.py --experiment logs/experiments/<timestamp>_lgb_ridge_<hash> --out_sub data/submissions/submission_infer.csv
   ```
   - 若未指定 `--out_sub`，预测结果会写入 `<experiment>/submission_infer.csv`。

## 目录速览

- `data/processed/htmp_v*/`：`build_features.py` 生成的特征缓存及 manifest。
- `logs/experiments/`：`train.py` 自动创建的实验归档（配置、模型、指标、特征重要性、预测结果）。
- `src/train.py`：带时间序列交叉验证的 LightGBM + Ridge 集成训练入口。
- `src/predict.py`：加载指定实验资产，生成比赛提交文件。
- `docs/执行手册.md`：详述执行流程、目录规划与后续迭代建议。

更多背景信息与实施细节请参阅 `docs/执行手册.md`。
