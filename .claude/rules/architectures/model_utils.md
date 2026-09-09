# AI2Pot 模型的 model_utils 规范

本文档定义了基于 AI2Pot 开发模型时，`model_utils` 应遵循的基本规范。

以下规则应视为长期设计约束。除非用户明确要求进行架构层面的调整，否则不应为了单个模型的实现便利而随意修改已有公共接口。

---

# 1. 基本要求

对于可训练、可导出和可部署的 AI2Pot 模型，应提供对应的 `model_utils`。

通常至少包括：

```text
ModelSerializer
Model4Extxyz
ModelCalculator
```

三者分别负责：

```text
Model
 ├── ModelSerializer  → 模型序列化与部署
 ├── Model4Extxyz     → 数据集预测与结果导出
 └── ModelCalculator  → ASE Calculator 接口
```

不同模型之间应尽量保持这三个工具类的职责和使用方式一致。

---

# 2. ModelSerializer

`ModelSerializer` 用于将训练完成的模型转换为部署模型。

应优先继承：

```python
PotentialSerializerBase
```

主要职责包括：

* 从 Lightning checkpoint 加载模型；
* 提取实际 `torch.nn.Module`；
* 切换至推理状态；
* 转换为部署所需的数据类型；
* 导出 TorchScript 模型。

典型流程：

```text
Lightning Checkpoint
        ↓
     LitModel
        ↓
       Model
        ↓
 torch.jit.script
        ↓
 torch.jit.freeze
        ↓
    model.pt
```

序列化后的模型应尽可能保持与训练阶段相同的 AI2Pot 模型输入接口。

---

# 3. Model4Extxyz

`Model4Extxyz` 用于对 extxyz 数据集进行预测，并输出模型预测值与对应 Label。

应优先继承：

```python
Potential4ExtxyzBase
```

主要用于：

* 训练集预测；
* 验证集 / 测试集预测；
* Energy、Force、Virial 等预测结果导出；
* RMSE 等误差分析；
* parity plot 等后处理。

应优先复用：

```text
ExtxyzDataModule
AI2Pot Dataset
AI2Pot 7-Tensor Interface
```

不要为了某个模型重新实现独立的数据读取和 Neighbor List 流程。

---

# 4. ModelCalculator

`ModelCalculator` 用于提供 ASE Calculator 接口。

应优先继承：

```python
PotentialCalculatorBase
```

主要职责包括：

```text
ase.Atoms
    ↓
AI2Pot 数据转换
    ↓
7-Tensor Interface
    ↓
Model
    ↓
Energy / Force / Virial
    ↓
ASE results
```

应尽量复用 AI2Pot 已有的：

```text
MlffInput
Neighbor List
Model interface
type_map
```

不要为单个模型重新实现与 AI2Pot 公共数据接口重复的 ASE 数据处理流程。

---

# 5. 优先继承已有 Base

实现新的 `model_utils` 时，应优先复用 AI2Pot 已有公共基类：

```python
PotentialSerializerBase
Potential4ExtxyzBase
PotentialCalculatorBase
```

只有模型确实存在公共基类无法支持的特殊需求时，才扩展新的逻辑。

如果某项功能可以被多个模型共同使用，应优先加入公共 Base，而不是在多个模型的 `model_utils` 中重复实现。

---

# 6. 推荐参考实现

新增模型时，应优先阅读：

```text
AI2Pot/ai2pot/models/mtp/linear_mtp_utils.py
AI2Pot/ai2pot/models/mtp/nn_mtp_utils.py
AI2Pot/ai2pot/models/nep/nep_utils.py
```

其中 Linear MTP 已分别通过 `LinearMtpSerializer`、`LinearMtp4Extxyz` 和 `LinearMtpCalculator` 实现上述三类接口，可作为新模型的主要参考范式。

---

# 7. 职责边界

`model_utils` 主要负责：

```text
模型加载
模型序列化
数据适配
数据集预测
结果导出
ASE 接口
```

模型的核心数学计算应保留在 `Model` 中。

训练相关逻辑应保留在 `LitModel` / Trainer 中。

因此不应在 `model_utils` 中重复实现：

```text
模型核心 forward
Descriptor / Graph 核心计算
training loop
optimizer
scheduler
通用 loss
```

总体职责应保持：

```text
Model
    → 模型本身的数学和物理计算

LitModel / Trainer
    → 训练流程

model_utils
    → Serialization / Dataset Prediction / ASE Deployment
```
