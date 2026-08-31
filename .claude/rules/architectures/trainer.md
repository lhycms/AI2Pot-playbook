# 基于 AI2Pot 的训练流程规范

本文档定义了基于 AI2Pot 开发机器学习模型时应遵循的训练流程规范。

以下规则应视为长期设计约束。除非用户明确要求进行架构层面的调整，否则不应为了单个模型的实现便利而随意修改训练框架。

# 1. 使用 PyTorch Lightning 管理训练流程

AI2Pot 的训练流程应统一基于 **PyTorch Lightning** 实现。

模型训练、验证、优化器配置、学习率调度、梯度裁剪以及日志管理等流程，应优先通过 `LightningModule` 和 `Trainer` 完成。

实现新模型的训练流程时，应优先参考：

```text
AI2Pot/ai2pot/models/potential_train.py
```

除非存在明确的技术原因，否则不要为单个模型重新实现独立的训练循环。

# 2. 复用统一的 Callback

训练过程中涉及的通用功能应优先通过 PyTorch Lightning Callback 实现，并复用 AI2Pot 已有的 Callback 基础设施。

参考：

```text
AI2Pot/ai2pot/models/potential_train_utils.py
```

例如以下功能不应重复散落在不同模型的训练代码中：

```text
描述符归一化
训练状态记录
梯度监控
模型保存
日志输出
训练过程控制
```

当多个模型需要相同的训练功能时，应优先扩展或复用公共 Callback，而不是在各模型中分别实现。

# 3. 保持模型与训练流程解耦

Model 应主要负责前向计算和模型本身的数学逻辑。

训练流程相关逻辑，例如：

```text
loss 组织
optimizer
scheduler
logging
checkpoint
callback
```

应尽量由 `LightningModule`、`Trainer` 或 `Callback` 管理。

不要为了简化训练代码，将通用训练控制逻辑直接写入模型内部。
