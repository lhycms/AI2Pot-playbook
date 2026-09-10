# Port Model to AI2Pot

本 Skill 用于将已有机器学习模型的参考实现迁移到 AI2Pot 架构中。

适用于以下场景：

```text
AI2Pot-playbook/
AI2Pot/
Upstream-Model/
Target-Implementation/
```

其中：

* `AI2Pot-playbook`：定义 AI2Pot 的架构规则和开发规范；
* `AI2Pot`：提供可以复用的基础设施和参考实现；
* `Upstream-Model`：提供待迁移模型的原始算法和参考代码；
* `Target-Implementation`：实际需要完成开发的目标仓库。

---

# 1. 基本原则

迁移模型时，应遵循以下优先级：

```text
AI2Pot-playbook
      ↓
AI2Pot architecture
      ↓
Upstream algorithm
      ↓
Target implementation
```

应保留 upstream 模型的：

* 数学定义；
* 模型结构；
* 物理含义；
* 必要的数值处理；
* 核心算法行为。

但不要求保留 upstream 的：

* Dataset 设计；
* Graph 构建方式；
* Trainer；
* 数据接口；
* 项目目录结构；
* 部署方式；
* 与 AI2Pot 重复的基础设施。

当 upstream 的软件架构与 AI2Pot 冲突时：

> 保留算法行为，将软件实现适配到 AI2Pot 架构。

---

# 2. 开始实现前先分析

不要立即修改代码。

首先阅读：

```text
AI2Pot-playbook/.claude/rules/
```

尤其关注：

```text
architecture/model.md
architecture/dataset.md
architecture/neighbor-list.md
architecture/trainer.md
```

然后分析 upstream repository，至少明确：

```text
模型真正的核心 forward 是什么？
模型需要哪些输入？
Neighbor List / Graph 如何构建？
模型有哪些内部 representation？
输出哪些物理量？
Loss 如何定义？
训练流程有哪些特殊处理？
是否存在自定义算子或特殊数值操作？
```

区分：

```text
算法必要部分
vs.
upstream 软件工程实现
```

不要机械复制 upstream repository。

---

# 3. 优先复用 AI2Pot 基础设施

在实现任何基础组件之前，先检查 AI2Pot 是否已经提供对应功能。

优先复用：

```text
Dataset
Neighbor List
GraphDataConverter
C++ / CUDA operators
training infrastructure
LitModel
Callbacks
ASE interface
LAMMPS interface
utility functions
```

如果 AI2Pot 已经存在功能相同或接近的基础设施，应优先复用或扩展，而不是重新实现一套模型专用版本。

---

# 4. 适配模型输入

标准 AI2Pot Model 应优先接收：

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

不要因为 upstream 使用：

```text
Graph
edge_index
positions
neighbor_pairs
PyG Data
custom batch
```

就修改 AI2Pot 的公共模型接口。

如果模型需要 Graph 或其他特殊表示，应在 Model 内部完成：

```text
7-Tensor Interface
        ↓
Model-specific Converter
        ↓
Model-specific Representation
        ↓
Model
```

例如：

```python
def forward(
    self,
    binum_tensor,
    bilist_tensor,
    bnumneigh_tensor,
    bfirstneigh_tensor,
    brcs_tensor,
    btypes_tensor,
    bnghost_tensor,
):
    graph_data = GraphDataConverter.convert_nblist_to_graph(
        binum_tensor=binum_tensor,
        bilist_tensor=bilist_tensor,
        bnumneigh_tensor=bnumneigh_tensor,
        bfirstneigh_tensor=bfirstneigh_tensor,
        brcs_tensor=brcs_tensor,
        btypes_tensor=btypes_tensor,
        bnghost_tensor=bnghost_tensor,
    )

    ...
```

---

# 5. 实现 Model

Model 主要负责：

```text
读取 AI2Pot 输入
      ↓
构建内部 representation
      ↓
执行模型计算
      ↓
输出预测结果
```

尽量保持 upstream 的数学结构清晰可辨。

不要在 Model 中加入：

```text
training loop
checkpoint
logging
通用 optimizer 配置
Dataset preprocessing
```

这些职责应由 AI2Pot 的其他模块承担。

---

# 6. 实现 Dataset

如果现有 AI2Pot Dataset 已经能够提供模型需要的数据，应直接复用。

只有确实存在新的数据格式或监督信息时，才新增 Dataset。

新的 Dataset 应遵循 `architecture/dataset.md`。

原则上：

```text
Dataset
   ↓
AI2Pot Core Atomistic Interface
   ↓
Model
```

不要让 Dataset 构建模型专用的 Graph 或 Descriptor。

---

# 7. 实现 LitModel 和训练流程

训练流程统一遵循 AI2Pot 的 PyTorch Lightning 范式。

优先参考：

```text
AI2Pot/ai2pot/models/potential_train.py
```

LitModel 主要负责：

```text
调用 Model
计算 Loss
training_step
validation_step
optimizer
scheduler
logging
```

不要直接复制 upstream 的完整 training loop。

如果 upstream 存在必要的特殊训练策略，应将其合理映射到 AI2Pot 的 Lightning 训练体系中。

---

# 8. 保持实现最小化

第一次迁移时优先完成一个：

> **最小、正确、可训练、可验证的 AI2Pot 实现。**

不要在第一阶段同时进行：

```text
大规模重构
激进性能优化
CUDA kernel 重写
额外功能扩展
与原模型无关的代码清理
```

先保证算法正确，再逐步优化。

---

# 9. 验证

实现完成后必须进行验证。

优先验证：

```text
输入 shape 和 dtype
Neighbor List / Graph 转换
单结构 forward
batch forward
energy / force 等输出 shape
gradient
training step
数值稳定性
```

如果 upstream implementation 可以运行，应尽可能构建相同输入，对比：

```text
中间 representation
中间 feature
最终预测结果
```

重点验证 **数值等价性或合理的一致性**，而不仅仅是代码能够运行。

如果结果存在差异，应首先检查：

```text
Neighbor List convention
edge direction
relative vector direction
type mapping
cutoff
PBC
padding
normalization
unit
aggregation convention
```

不要为了让测试通过而随意修改物理定义。

---

# 10. 完成后的检查

完成迁移后确认：

* 是否遵循 AI2Pot 七张量模型接口；
* 是否复用了 AI2Pot 已有基础设施；
* 是否避免将模型特定逻辑放入公共 Dataset；
* Model 与 LitModel 是否职责清晰；
* Neighbor List 索引是否符合 AI2Pot 规范；
* 是否保留 upstream 的核心算法行为；
* 是否存在可以删除的重复实现；
* 是否具有基本的 numerical tests；
* 是否能够完成基本训练流程。

如果为了该模型修改了 AI2Pot 公共架构，应重新判断该修改是否确实具有通用价值。

---

# 11. 推荐工作流程

```text
阅读 AI2Pot-playbook
        ↓
分析 upstream repository
        ↓
提取模型核心算法
        ↓
检查 AI2Pot 可复用组件
        ↓
设计 AI2Pot-compatible architecture
        ↓
实现 Model
        ↓
实现必要的 Dataset / Converter
        ↓
实现 LitModel
        ↓
运行单元测试
        ↓
与 upstream 数值对比
        ↓
修复差异
        ↓
最后再考虑性能优化
```

最终目标不是复制 upstream repository。

最终目标是：

> **在保持原模型算法行为的基础上，得到一个符合 AI2Pot 架构、能够复用 AI2Pot 基础设施，并可以继续维护和扩展的实现。**
