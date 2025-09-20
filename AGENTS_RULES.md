# AGENTS 协作守则

1. **配置即真理**：所有流水线参数必须写入 `configs/`，并通过 `src/config.loader` 读取；禁止在脚本中硬编码。
2. **结构同步**：当代码结构变化时，务必同时更新 `docs/ARCHITECTURE.md` 与 `CHANGELOG.md`。
3. **暴露控制**：策略暴露、Sharpe、换手率等度量统一放在 `src/models/exposure.py`，训练阶段只能调用该模块。
4. **训练封装**：新增或修改训练逻辑时，应在 `src/pipelines/training.py` 内实现，`src/train.py` 仅负责调度与输出。
5. **特征流水线**：若新增特征步骤，请在 `src/features/pipeline.py` 中扩展，并保证 `build_features.py` 的 CLI 行为不变。
6. **验证清单**：提交前至少运行 `python -m compileall src` 与 `pytest`，确保基础校验通过。
