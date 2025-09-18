# AGENTS 指南：Hull Tactical Market Prediction 项目

> **使命**：以自动化 Agent 协作形式，围绕 Kaggle [Hull Tactical Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview) 竞赛构建顶级解决方案，目标冲击排行榜前三。

## 1. 项目状态概览
- **数据**：`data/raw/` 已放置官方 `train.csv`、`test.csv` 等文件。数据为金融时间序列、逐日预测标普 500 超额收益。
- **代码结构**：
  - `src/`：包含 `data.py`（原始加载）、`train.py`（训练入口，基础框架已搭好）等模块。
  - `configs/`：存放 `features.yml`、`cv.yml`、`model.*.yml` 等配置模板。
  - `docs/`：`执行手册.md`、`eda_report.md` 等文档（需持续更新）。
  - `logs/`：训练日志输出目录，现有实验效果不佳，需迭代。
  - `notebooks/`：`00_project_todo.ipynb` 提供执行路线与 TODO 追踪。
- **现状**：已有初步数据加载、特征处理、训练流程，但性能较弱。需系统扩展特征、模型与自动化能力。

## 2. Agent 工作原则
1. **时间序列约束**：任何特征/插补/切分必须保证时间顺序，不得泄露未来信息。
2. **可复现性**：所有实验通过配置驱动 (`configs/`)，并记录到 `logs/`/`experiments/`（若尚未建立，需补全）。
3. **自动化优先**：所有关键流程（特征生成、训练、推理、提交）需具备脚本或 Make 目标，一键执行。
4. **文档同步**：对流程/配置重大更新，及时维护 `docs/执行手册.md` 与 `docs/eda_report.md`。
5. **协同记忆**：每次迭代在 `AGENTS.md` 或 `docs/` 注册说明，避免重复劳动。

## 3. 数据处理框架
- **目标**：构建稳健的数据流水线，从原始 CSV → 清洗 → 特征矩阵。
- **必做任务**：
  - 完成 `src/data.py` 中的缺失值处理、异常检测、滞后特征生成（与测试集 `lagged_*` 对齐）。
  - 在 `configs/features.yml` 定义窗口参数与特征选择，可按组（D/E/I/M/P/S/V）配置。
  - 构建 `FeatureBuilder`（若未实现）或增强现有实现，支持：滞后、滚动统计、z-score、winsorize、缺失指示器、二元特征计数。
  - 添加 `data/interim/`、`data/processed/` 产出，并在配置/日志中登记版本。
- **可用工具**：`check_for_feature.py` 等辅助脚本，应统一入口命令，避免散乱脚本。

## 4. 特征工程路线图
1. **基础特征**：
   - 滞后 (`lag_1/5/10/20`)、差分、滚动均值/标准差/偏度。
   - 二元特征的运行长度、累计计数、窗口内出现次数。
   - 缺失指示器、按日期分组的均值/标准化。
2. **高级特征**：
   - 市场 regime 识别（基于 `risk_free_rate`、`market_forward_excess_returns` 滞后统计）。
   - PCA/因子分解、交互项、聚类标签。
   - 时序嵌入：利用 Transformer/LSTM 模型自身抽取的隐向量回填至树模型。
3. **特征筛选**：进行 `mutual_info`、`SHAP`、`permutation importance` 评估；低贡献 + 高缺失的列应剔除。

## 5. 模型与实验策略
- **基线**：LightGBM、XGBoost、CatBoost、正则线性模型（Ridge/Lasso/ElasticNet）。
- **高级模型组合**：
  - 时间序列 Transformer（Informer、Temporal Fusion Transformer）。
  - Attention-LSTM / 双向 LSTM、LSTM-GNN（基于因子图或特征图建图）。
  - TabNet、FT-Transformer 等表格神经网络。
  - Stacking/Blending：将树模型、线性模型、神经网络输出进行加权或次级学习。
  - 混合集成（Bagging + Boosting + 深度模型），记录组合权重与验证成绩。
- **实验管理**：
  - 建议引入 `experiments/` 目录和 `mlflow` 或 CSV 日志记录（提交编号、分数、配置哈希）。
  - 每个实验配置存储在 `configs/experiment/<name>.yml`，命名规范：`YYYYMMDD_<model>_<notes>`。
  - `logs/` 中保存训练日志、验证指标、特征重要性；确保路径在 README/文档中说明。

## 6. 训练与验证
- **切分**：使用 `configs/cv.yml` 控制滚动或扩张窗口；确保验证集覆盖接近测试的日期。
- **指标**：
  - Kaggle 官方指标为均方误差 (MSE) 于特定加权方案（详见比赛说明），训练时需实现相同指标函数。
  - 同时追踪 MAE、R²、信息比率等金融指标。
- **训练脚本** (`src/train.py`):
  - 读取配置 → 加载特征 → 数据切分 → 模型训练 → 评估 → 输出模型 (`model.pkl`) 与指标 (`metrics.json`)。
  - 支持命令行参数：`--model-config`、`--features-config`、`--cv-config`、`--experiment-name`。
  - 集成早停、超参搜索（`optuna`）、交叉验证平均。

## 7. 推理与提交
- `src/predict.py`（如未完成需补写）：加载最佳模型与特征流水线 → 处理 `test.csv` → 生成 `submission.csv`。
- 对接 Kaggle CLI：`scripts/submit.sh` 或 Make 目标，调用 `kaggle competitions submit`；提交前验证列顺序与数值范围。
- 保留提交日志（日期、模型、线下分数、线上分数、配置文件）以追踪最佳方案。

## 8. 自动化与一键流程
- 构建 `Makefile` 或 `scripts/`：
  - `make data`：校验 & 生成缺失特征。
  - `make features`：运行特征流水线。
  - `make train MODEL=lgb`：按指定配置训练。
  - `make predict RUN=xxx`：生成 submission。
  - `make submit RUN=xxx MESSAGE="..."`：自动提交并记录日志。
- 若资源允许，准备多 GPU/CPU 并行策略（尤其是 Transformer/LSTM/GNN）。
- 对关键脚本编写单元测试 (`tests/`)；CI（可选 GitHub Actions）运行 `pytest` + `ruff` + `black`。

## 9. 文档与知识库
- `docs/执行手册.md`：需与本指南同步，详细说明环境、指令与流程。
- `docs/eda_report.md`：持续补充数据分析、缺失处理、特征发现。
- `AGENTS.md`：维护高层策略、当前状态、待办任务。更新请标注日期与作者。

## 10. 行动清单（建议起始点）
1. 巩固数据流水线：完善缺失插补、滞后特征、特征缓存逻辑。
2. 搭建统一训练/推理脚本，整理实验日志，保证可复现。
3. 快速迭代树模型 + 正则线性模型，建立稳健基线。
4. 引入时间序列 Transformer / Attention-LSTM，比较性能并探索堆叠方案。
5. 设计混合集成框架（权重搜索或二级模型），整合集体最优模型。
6. 自动化提交流程，建立分数追踪表，冲击排行榜前三。

> **提醒**：金融时间序列噪声大、漂移快。保持严谨的验证策略与记录，逐步增强模型稳定性。任何大幅改动务必同步文档和配置，确保后续 Agent 能迅速接手。
