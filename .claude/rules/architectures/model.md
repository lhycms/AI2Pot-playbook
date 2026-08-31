# AI2Pot Core Atomistic Interface

## Status

这个文档定义了 AI2Pot 的基础架构的标准。

请将以下规则视为长期的设计约束，而不是为了方便而随意修改的开发约定。

除非用户明确要求进行架构层面的重新设计，否则不要修改核心模型接口。

---

# 1. Core principle

AI2Pot 不仅仅是一套机器学习势函数模型的实现集合。

其核心抽象是一套 **统一的原子体系数据接口**，可以由不同模型（或许是势函数、生成模型甚至哈密顿量模型）共同使用。

无论模型内部采用何种架构，标准的 AI2Pot 原子模型都使用以下7个张量作为输入：

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

这些张量共同构成了 **AI2Pot 核心原子体系接口（AI2Pot Core Atomistic Interface）**

在不同模型的实现中，应尽可能保持这些张量的名称、语义、维度约定及其功能定义一致且稳定。

---

# 2. 七张量接口是模型对外的统一接口规范

AI2Pot 的标准预测流程应该保持如下形式：

```text
Structure / Dataset / ASE / LAMMPS
                │
                ▼
     AI2Pot atomistic preprocessing
                │
                ▼
       Core 7-Tensor Interface
                │
                ▼
              Model
```

模型不应要求 Dataset、Trainer、ASE 接口或 LAMMPS 接口负责构建模型特定的数据表示。

模型外部的基础设施应统一提供 AI2Pot 的通用数据表示。

模型自身负责对该通用表示进行解析，并根据需要将其转换为模型内部所使用的特定表示形式。

---

# 3. 模型内部表示应由模型自身负责

不同类型的模型通常需要采用不同的数学表示形式。

Examples include:

```text
MTP
7 tensors
   ↓
moment-tensor descriptors
   ↓
energy / forces


NEP
7 tensors
   ↓
NEP descriptors
   ↓
neural network


Graph neural network
7 tensors
   ↓
GraphDataConverter
   ↓
GraphData
   ↓
message passing


Neural tight binding
7 tensors
   ↓
orbital / graph representation
   ↓
Hamiltonian model
```

不要因为某个模型要不同的的数据表示形式，就改变 AI2Pot 全局统一的输入风格。

相反，应在模型内部中实现相应的适配器或者转换器，将 AI2Pot 的通用数据表示转换为该模型所需要的特定表示形式。

For example, graph models should generally follow:

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

因此，图表示（Graph Representation）应被视为一种**派生表示（Derived Representation）**，而不是 AI2Pot 的通用输入格式。

---

# 4. Neighbor-list representation is more fundamental than graph representation

Do not assume that every atomistic machine-learning model is fundamentally a graph neural network.

The AI2Pot native representation is intentionally close to atomistic simulation neighbor-list data structures.

Conceptually:

```text
AI2Pot native neighbor representation
                │
      ┌─────────┼──────────┐
      │         │          │
      ▼         ▼          ▼
     MTP       NEP     Graph Adapter
                           │
                           ▼
                      Graph Model
```

A graph is one possible interpretation of the atomistic neighborhood.

It should not become a mandatory intermediate representation for models that do not need one.

---

# 5. Training and deployment should expose the same model interface

One major architectural goal of AI2Pot is to minimize differences between training-time and deployment-time model execution.

For example:

```text
Training Dataset
      │
      ▼
7-Tensor Interface
      │
      ▼
    Model


LAMMPS
   │
   ▼
neighbor-list data
   │
   ▼
7-Tensor Interface
   │
   ▼
 Model
```

The model should ideally not care whether its inputs originate from:

* an AI2Pot dataset,
* ASE,
* a training dataloader,
* LAMMPS,
* CPU execution,
* GPU execution.

Avoid creating separate model APIs for training and molecular-dynamics inference unless technically unavoidable.

---

# 6. New model development

When implementing a new model, prefer the following development pattern:

```text
1. Reuse AI2Pot dataset/preprocessing infrastructure
                       ↓
2. Receive the standard seven tensors
                       ↓
3. Convert them internally if necessary
                       ↓
4. Implement model-specific physics / architecture
                       ↓
5. Reuse AI2Pot training and deployment infrastructure
```

Examples of acceptable internal representations include:

* local descriptors,
* graph representations,
* equivariant graph representations,
* orbital graphs,
* basis-function representations,
* model-specific compressed representations.

Do not redesign the global dataset or model interface merely to simplify one model implementation.

---

# 7. Additional inputs

The seven tensors describe the common local atomistic environment, but they are not assumed to encode every possible physical quantity.

Future models may require additional task-specific information such as:

```text
k-points
orbitals
charges
spins
electronic occupations
diffusion timestep
conditioning variables
external fields
```

When additional inputs are genuinely required, preserve the seven core tensors and extend the model interface deliberately.

Conceptually:

```python
forward(
    binum_tensor,
    bilist_tensor,
    bnumneigh_tensor,
    bfirstneigh_tensor,
    brcs_tensor,
    btypes_tensor,
    bnghost_tensor,
    task_specific_input,
)
```

Do not repurpose the meaning of an existing core tensor to carry unrelated information.

---

# 8. Representation adapters

Converters such as:

```text
GraphDataConverter
```

should be treated as reusable infrastructure.

Future converters may include concepts such as:

```text
GraphDataConverter
EquivariantGraphConverter
OrbitalGraphConverter
DescriptorConverter
```

A converter should transform the stable AI2Pot core representation into a model-specific representation without changing the global data contract.

---

# 9. Performance considerations

Architectural stability and runtime implementation are separate concerns.

For example:

```text
7 tensors
   ↓
GraphDataConverter
   ↓
GraphData
```

may initially be implemented using native PyTorch operations such as:

```text
gather
mask
repeat_interleave
stack
```

If conversion becomes a performance bottleneck, optimize the implementation rather than changing the public model contract.

Possible optimization stages include:

```text
PyTorch implementation
        ↓
torch.compile-friendly implementation
        ↓
fused C++ implementation
        ↓
fused CUDA implementation
```

The interface should remain stable while its implementation becomes faster.

---

# 10. Avoid model-specific infrastructure leakage

Before adding a new tensor, preprocessing step, dataset field, or global abstraction, ask:

> Is this information fundamentally required by atomistic models in general, or only by this particular model?

If it is model-specific, prefer keeping it inside the model or a model-specific converter.

Avoid changes such as:

```text
Dataset
   ↓
DPA-specific preprocessing
   ↓
DPA-specific graph
```

when this can instead be:

```text
Dataset
   ↓
AI2Pot Core Interface
   ↓
DPA model
   ↓
DPA-specific representation
```

---

# 11. Architectural decision rule

When choosing between two implementations, prefer the design that preserves:

```text
stable data contract
        +
model independence
        +
training/deployment consistency
        +
high-performance implementation freedom
```

over a design that makes one model easier to implement but introduces model-specific assumptions into the AI2Pot core.

---

# 12. AI2Pot's intended abstraction

Think of AI2Pot as:

```text
                 AI2Pot

        Atomistic Data Contract
                  │
        Neighbor Infrastructure
                  │
          C++ / CUDA Runtime
                  │
              PyTorch
                  │
      ┌───────────┼───────────┐
      │           │           │
     MTP         NEP         GNN
                              │
                         GraphData
```

Potential future extensions may include:

```text
ML potentials
graph neural networks
neural tight-binding models
atomistic generative models
property-prediction models
```

The models may change.

The internal representations may change.

The optimized kernels may change.

The **AI2Pot Core Atomistic Interface should remain stable**.
