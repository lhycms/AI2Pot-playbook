# AI2Pot Core Atomistic Interface

## Status

This document defines a fundamental architectural invariant of AI2Pot.

Treat the rules below as long-term design constraints rather than conventions that may be changed for convenience.

Do not modify the core model interface unless the user explicitly requests an architectural redesign.

---

# 1. Core principle

AI2Pot is not only a collection of interatomic-potential implementations.

Its core abstraction is a unified atomistic data interface shared by different models and execution environments.

Regardless of the internal model architecture, standard AI2Pot atomistic models receive the following seven tensors as their core input:

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

These tensors constitute the **AI2Pot Core Atomistic Interface**.

Their names, semantics, dimensional conventions, and roles should remain stable across model implementations whenever possible.

---

# 2. The seven-tensor interface is the external model contract

The standard AI2Pot prediction path should conceptually remain:

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

Models should not require the dataset, trainer, ASE interface, or LAMMPS interface to construct model-specific representations.

The infrastructure outside the model should provide the common AI2Pot representation.

The model is responsible for interpreting or transforming that representation.

---

# 3. Internal representations belong inside models

Different model families naturally require different mathematical representations.

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

Do not change the global AI2Pot input format simply because one model requires another representation.

Instead, implement an adapter/converter inside or immediately adjacent to that model.

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

Graph representation is therefore a **derived representation**, not the universal AI2Pot input format.

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
