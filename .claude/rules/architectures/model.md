# AI2Pot 模型接口与实现规范

本文档定义了基于 AI2Pot 开发模型时应遵循的核心架构规范。

以下规则应视为长期设计约束。除非用户明确要求进行架构层面的重新设计，否则不应为了单个模型的实现便利而随意修改核心接口。

---

# 1. 核心模型接口

AI2Pot 的核心抽象是一套统一的原子体系数据接口，可供机器学习势函数、图神经网络、神经网络紧束缚模型、生成模型等不同类型的模型使用。

标准 AI2Pot 模型应使用以下 7 个张量作为核心输入：

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

这些张量共同构成 **AI2Pot 核心原子体系接口（AI2Pot Core Atomistic Interface）**。

不同模型之间应尽可能保持这些张量的名称、语义、维度约定和功能定义一致。

标准数据流应保持为：

```text
Dataset / ASE / LAMMPS
          │
          ▼
   AI2Pot Nblist
          │
          ▼
  7-Tensor Interface
          │
          ▼
        Model
```

Dataset、Trainer、ASE 接口和 LAMMPS 接口负责提供统一的 AI2Pot 数据表示，不应负责构建模型特定的数据表示。

---

# 2. 模型内部表示

不同模型可以采用不同的内部表示，例如：

```text
MTP
7 tensors
   ↓
Moment Tensor Descriptor
   ↓
Model


NEP
7 tensors
   ↓
NEP Descriptor
   ↓
Model


Graph Model
7 tensors
   ↓
GraphDataConverter
   ↓
GraphData
   ↓
Model
```

不要因为某个模型需要特殊的数据表示，就修改 AI2Pot 的公共输入接口。

模型需要 Graph、Descriptor、Orbital Graph 等特殊表示时，应在模型内部或与模型直接相关的 Converter 中完成转换。

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

Graph、Descriptor 等均属于 **派生表示（Derived Representation）**，而不是 AI2Pot 的公共输入格式。

Neighbor List 的具体约定参考同目录下的 `neighbor-list.md`。

---

# 3. Model 与 LitModel 的职责

AI2Pot 中应尽量区分 **模型本身的计算逻辑** 与 **训练逻辑**。

## Model

Model 应主要负责：

```text
输入解析
内部表示转换
模型前向计算
物理量预测
```

Model 不应负责通用的训练流程控制，例如：

```text
optimizer
scheduler
training loop
checkpoint
logging
```

实现新的 Model 时，应优先参考：

```text
AI2Pot/ai2pot/models/mtp/linear_mtp.py
AI2Pot/ai2pot/models/nep/nep.py
AI2Pot/ai2pot/models/mtp/nn_mtp.py
```

## LitModel

训练相关逻辑应通过 PyTorch Lightning 的 `LightningModule` 实现。

LitModel 通常负责：

```text
调用 Model
计算 loss
training_step
validation_step
optimizer
scheduler
logging
```

实现新的 LitModel 时，应优先参考：

```text
AI2Pot/ai2pot/models/potential_train.py
```

除非存在明确的技术原因，否则不要让每个模型重新设计完全不同的 LitModel 训练范式。

训练流程的详细规范参考同目录下的 `trainer.md`。

---

# 4. 训练与部署应保持相同的模型接口

AI2Pot 应尽量保证训练阶段和部署阶段调用相同的 Model 接口。

例如：

```text
Training Dataset
      │
      ▼
7-Tensor Interface
      │
      ▼
    Model
```

以及：

```text
LAMMPS
   │
   ▼
Neighbor List
   │
   ▼
7-Tensor Interface
   │
   ▼
 Model
```

Model 原则上不应关心输入来自 Dataset、ASE、LAMMPS、CPU 或 GPU。

除非技术上确有必要，否则不要为训练和分子动力学推理设计两套不同的 Model 输入接口。

---

# 5. 新模型的推荐实现方式

新增模型时，应优先遵循：

```text
复用 AI2Pot Dataset / Neighbor List
                ↓
        接收标准 7 个张量
                ↓
      构建模型内部表示
                ↓
         实现 Model
                ↓
         实现 LitModel
                ↓
   复用 AI2Pot Trainer / Callback
                ↓
     复用 ASE / LAMMPS 部署接口
```

不要仅为了简化某个模型的实现，就修改 Dataset、Neighbor List 或公共 Model 接口。

---

# 6. 额外模型输入

7 个核心张量描述通用的局域原子环境，但不要求覆盖所有可能的物理信息。

部分模型可能需要额外输入，例如：

```text
k-points
orbitals
charges
spins
diffusion timestep
conditioning variables
external fields
```

在确有必要时，可以在保留 7 个核心输入的基础上增加额外参数：

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
    task_specific_input,
):
    ...
```

不要修改已有核心张量的语义来承载无关信息。

---

# 7. 架构决策原则

新增模型或修改现有模型时，应优先保证：

```text
稳定的 7-Tensor Interface
        +
模型特定逻辑留在模型内部
        +
Model 与 LitModel 职责清晰
        +
训练与部署接口一致
        +
公共基础设施可以被不同模型复用
```

不要为了单个模型的实现便利，将模型特定的假设引入 AI2Pot 的公共基础设施。

**模型可以变化，内部表示可以变化，训练任务可以变化，但 AI2Pot 核心原子体系接口应尽可能保持稳定。**
