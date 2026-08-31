# AI2Pot-Playbook

本仓库是 AI2Pot 的机器可读工程知识库，用于指导 Coding Agent 维护、扩展和基于 AI2Pot 开发新的原子尺度机器学习模型。

本仓库本身不是 AI2Pot 的实现代码。AI2Pot 的具体实现位于独立的 `AI2Pot` 仓库中。

---

# 1. 基本职责

在典型工作区中：

```text
workspace/
├── AI2Pot-Playbook/
├── AI2Pot/
├── Upstream-Model/
└── Target-Implementation/
```

各目录的职责为：

```text
AI2Pot-Playbook
    → 定义架构规则、开发规范和 Agent Skills

AI2Pot
    → 提供 AI2Pot 当前实现和可复用基础设施

Upstream-Model
    → 提供待迁移模型的原始算法和参考实现

Target-Implementation
    → 当前实际开发的目标仓库
```

开发时应明确区分：

> AI2Pot-Playbook 规定“应该如何实现”。

> AI2Pot 展示“AI2Pot 当前如何实现”。

> Upstream-Model 提供“模型算法应该是什么”。

---

# 2. 开发优先级

进行 AI2Pot 相关开发时，应遵循以下优先级：

```text
AI2Pot-Playbook Rules
        ↓
AI2Pot 架构与公共接口
        ↓
AI2Pot 现有实现
        ↓
Upstream Model 的算法行为
        ↓
Target Implementation
```

当 upstream repository 的软件结构与 AI2Pot 架构冲突时：

> 保留 upstream 模型的算法和数学行为，但将其软件架构适配到 AI2Pot 的开发规范。

不要机械复制 upstream repository 的 Dataset、Trainer、Graph 构建方式或项目结构。

---

# 3. 开始开发前必须阅读 Rules

进行具体开发任务之前，应先检查：

```text
.claude/rules/
```

尤其是：

```text
architecture/
├── dataset.md
├── model.md
├── neighbor-list.md
└── trainer.md
```

Rules 定义 AI2Pot 中长期保持稳定的架构约束。

不要为了某个模型的实现便利而绕过或修改这些约束。

如果确实需要改变公共架构，应先明确说明原因，并判断该修改是否具有跨模型的通用价值。

---

# 4. 优先使用 Skills

对于已经存在对应 Skill 的任务，应优先按照：

```text
.claude/skills/
```

中的流程执行。

例如：

```text
port-model-to-ai2pot
remote-develop
```

Rules 主要回答：

> AI2Pot 中什么必须保持一致？

Skills 主要回答：

> 某类开发任务应该按照什么流程完成？

---

# 5. AI2Pot 核心模型接口

标准 AI2Pot 原子模型应优先使用以下 7 个张量作为核心输入：

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

这些张量构成 AI2Pot Core Atomistic Interface。

不要因为某个模型使用 Graph、Descriptor 或其他内部表示，就修改公共输入接口。

模型特定的数据表示应优先在模型内部构建：

```text
AI2Pot 7-Tensor Interface
            ↓
Model-specific Converter
            ↓
Model-specific Representation
            ↓
Model
```

详细规范参考：

```text
.claude/rules/architecture/model.md
```

---

# 6. 优先复用 AI2Pot 基础设施

开发新模型之前，应先检查 AI2Pot 是否已经提供对应功能。

优先复用：

```text
Dataset
Neighbor List
GraphDataConverter
C++ / CUDA operators
LitModel
Trainer
Callbacks
ASE interface
LAMMPS interface
utility functions
```

不要在 Target Implementation 中重新实现 AI2Pot 已经具备的通用基础设施。

如果某项新功能明显具有跨模型的通用价值，应考虑将其抽象为 AI2Pot 的公共基础设施。

---

# 7. Model 与训练流程分离

Model 应主要负责：

```text
输入解析
内部表示转换
模型计算
物理量预测
```

LitModel / Trainer 应主要负责：

```text
loss
training_step
validation_step
optimizer
scheduler
logging
callback
checkpoint
```

除非技术上确有必要，不要把通用训练控制逻辑放进 Model。

---

# 8. 开发策略

新增或迁移模型时，应优先采用：

```text
分析 upstream 模型
        ↓
提取核心数学和算法
        ↓
检查 AI2Pot 可复用基础设施
        ↓
设计 AI2Pot-compatible implementation
        ↓
实现最小正确版本
        ↓
添加单元测试
        ↓
进行数值验证
        ↓
最后再进行性能优化
```

第一次实现时优先追求：

> 最小、正确、可训练、可验证。

不要在算法尚未验证之前同时进行大规模重构或激进性能优化。

---

# 9. 数值正确性优先

对于模型迁移，不应以“代码能够运行”作为完成标准。

如果 upstream implementation 可以运行，应尽可能进行数值对比。

优先检查：

```text
Neighbor List
Graph / Representation
中间 feature
energy
forces
gradient
loss
```

出现数值差异时，应优先检查：

```text
dtype
unit
type mapping
Neighbor List convention
edge direction
relative vector direction
PBC
padding
cutoff
normalization
aggregation
```

不要通过随意放宽 tolerance 掩盖明显的数值问题。

---

# 10. Remote Development

如果本地缺少 CUDA、编译环境或训练条件，而远程服务器无法安装 Claude Code，应使用：

```text
.claude/skills/remote-develop/
```

标准原则为：

```text
Local Repository
    = Source of Truth

Remote Server
    = Isolated Execution Backend
```

Agent 在本地理解和修改代码，通过 SSH / rsync 将代码同步到远程服务器完成：

```text
compile
test
CUDA validation
training
benchmark
LAMMPS test
```

远程开发时，应为当前项目创建独立的 Python / Conda 环境，不要修改服务器已有项目环境。

---

# 11. 不要做的事情

除非用户明确要求，否则不要：

* 为单个模型重新设计 AI2Pot 核心接口；
* 将模型特定 Graph 构建逻辑放入通用 Dataset；
* 机械复制 upstream repository 的软件架构；
* 重复实现 AI2Pot 已有基础设施；
* 为训练和部署设计不必要的两套模型接口；
* 在数值正确性尚未确认前进行激进优化；
* 直接修改远程服务器并使其与本地源码产生未同步分叉；
* 破坏服务器已有 Python / Conda 环境。

---

# 12. 最终目标

AI2Pot-Playbook 的目标不是让 Agent 简单地“生成代码”。

其目标是让 Agent：

```text
理解 AI2Pot 架构
        ↓
遵守稳定的公共接口
        ↓
复用已有基础设施
        ↓
正确实现新的模型和功能
        ↓
通过测试和数值验证
        ↓
形成可继续维护的 AI2Pot-compatible implementation
```

模型可以变化。

内部表示可以变化。

具体实现可以变化。

但 AI2Pot 的核心接口、架构边界和开发原则应尽可能保持稳定。
