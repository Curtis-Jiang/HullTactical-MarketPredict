# Changelog

## 2025-09-19
- 重构训练脚本，将折叠训练与敞口校准逻辑转移到 `src/pipelines/training.py`，并新增 Logistic 回归信号用于敞口映射。
- 拆分模型辅助函数：新增 `src/models/exposure.py` 与 `src/models/preprocessing.py`，`src/model_utils.py` 作为兼容层。
- 新建配置与交叉验证子包（`src/config/`, `src/cv/`），统一管理 YAML 读取与时间序列拆分。
- 抽离特征构建流水线到 `src/features/pipeline.py`，CLI `build_features.py` 仅负责解析参数与写出结果。
- 更新实验生成流程：`train.py` 现专注于调度、落盘与日志记录，并序列化 LGB/Ridge/Logit 三类模型。
- 新增架构文档 `docs/ARCHITECTURE.md`，同步项目结构说明。

## 2025-09-20
- 调整训练流水线信号生成：支持 `signal_blend` 融合回归/分类信号，并引入 `signal_tanh_gain` 进行平滑压缩。
- 对 `_calibrate_fold_exposure` 增强约束控制，新增 `exposure_scale_floor` 与更细的缩放衰减逻辑，显著降低边界敞口占比及换手率。
- 更新 CLI `src/train.py` 参数文档，暴露新的风险控制开关；实验结果记录在 `logs/experiments/20250919_175851_refactor_blend_v2_2d4842f2/`。
- 同步 `docs/ARCHITECTURE.md`，说明信号融合与约束流程，以保持架构知识库一致。
- 新增 `prefix_aggregates` 行分组统计，扩充 Level-1 特征；引入 LightGBM 分类与波动回归模型配合 `risk_aversion` 做敞口调节。
- 引入 CatBoost 回归器并接入现有集成，新增 `w_cat`、`cat_params` 相关配置，训练脚本同步序列化 CatBoost 模型产物。
