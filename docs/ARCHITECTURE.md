# 系统架构总览

本项目的代码结构围绕“数据特征构建 → 训练流水线 → 推理与暴露控制”三条主线展开。经过本次重构，核心模块被拆分为清晰的子包，方便在不同阶段快速理解与迭代。

## 目录结构

- `src/config/`：配置加载与合并的工具（`loader.py` 提供 `load_config` / `cfg_hash`）。
- `src/cv/`：时间序列交叉验证拆分器（`splitters.py` 内的 `PurgedExpandingTimeSeriesSplit`）。
- `src/features/`：特征流水线（`pipeline.py` 封装 Level-1 构建、清洗、滞后、跨组聚合以及版本生成）。
- `src/models/`：模型侧辅助函数
  - `preprocessing.py`：预测裁剪、z-score 截断。
  - `exposure.py`：敞口映射、Sharpe 计算与风险指标。
- `src/pipelines/`：端到端流水线
  - `training.py`：多折训练、模型集成、敞口校准与指标采集。
- CLI 脚本
  - `src/build_features.py` 仅负责解析参数并调用 `features.pipeline`。
  - `src/train.py` 负责加载配置、调度 `pipelines.training`、持久化实验产物。

## 数据流程

1. **特征构建**
   - `build_features.py` 读取原始 CSV，加载 YAML 配置后调用 `build_level1`。
   - `build_level1` 内部执行时间感知缺失值填补、winsorize、滚动统计、滞后/离散累计，输出 `train_l1.parquet` / `test_l1.parquet` 以及 `manifest.json`。
2. **训练阶段**
   - `train.py`
     1. 加载最新特征版本，读取 `configs/cv.*.yml`，实例化 `PurgedExpandingTimeSeriesSplit`。
     2. 借助 `prepare_training_data`（`pipelines.training`）生成特征矩阵、裁剪阈值与评分掩码，并剔除折间低方差列。
     3. `run_training` 负责：
        - 同步训练 LightGBM 回归、CatBoost 回归、LightGBM 分类、Ridge 回归与 Logistic 回归；回归预测与分类概率按 `signal_blend` / `class_blend` 融合，并在 `signal_tanh_gain` 控制下以 `tanh` 平滑信号。
        - 利用绝对收益模型对未来波动做出估计，并按 `risk_aversion` 转化为敞口调节因子；与信号一同交由 `_calibrate_fold_exposure`，通过迭代衰减与 `exposure_scale_floor` 约束满足边界占比、换手率、波动比要求。
        - 聚合 OOF 预测、敞口、测试集平均预测、特征重要度与暴露信息，返回 `TrainingArtifacts`。
     4. 将模型快照、指标、OOF/Test 结果及配置写入 `logs/experiments/<timestamp>_...`。
3. **推理与提交**
   - 训练目录中存储 `model_lgb_fold*.txt`、`model_ridge_fold*.joblib`、`model_logit_fold*.joblib`，供 `predict.py` 及后续自动化脚本复用。

## 关键设计决策

- **模块化配置**：`src/config.loader` 统一处理 YAML 合并，避免散落的手工解析。
- **拆分训练逻辑**：`pipelines/training.py` 将原先冗长的脚本拆成数据准备、单折训练、敞口校准和指标汇总四个阶段，便于单元测试和未来扩展（如替换模型或引入 stacking）。
- **敞口工具下沉**：所有 Sharpe / 风险指标迁移到 `src/models/exposure.py`，训练阶段只需关注信号生成与约束参数（`signal_blend`、`class_blend`、`signal_tanh_gain`、`risk_aversion`、`exposure_scale_floor` 等）。
- **特征流水线可复用**：CLI 仅承担输入输出职责，特征构建逻辑被抽象为函数，便于 notebook 或批量任务直接调用。

以上结构支持后续快速替换模型、扩展特征或在 CI 中复用训练逻辑，同时减少跨模块耦合。

## 协作规则
- **单一事实源 (SSoT)**：任何结构调整务必同步 `docs/ARCHITECTURE.md` 与 `CHANGELOG.md`。
- **不可跳过的检查**：新增模块需配套单测或最小化验证（`pytest` 基线）、并确保 `python -m compileall src` 通过。
- **暴露约束**：敞口相关逻辑只能通过 `src/models/exposure.py` 维护，防止多处复制粘贴导致风险参数不一致。
