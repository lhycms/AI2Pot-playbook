# AI2Pot-Playbook

一个面向 [AI2Pot](https://github.com/lhycms/AI2Pot) 开发、维护与扩展的机器可读工程知识库。

AI2Pot-Playbook 提供架构规则、开发技能和实现规范，既可以供开发者阅读，也可以直接供 Coding Agent 使用。

它的主要目标是让 AI2Pot 的开发过程更加稳定和可复现：不再需要每次都重新向 Agent 解释 AI2Pot 的架构和开发约定，而是将这些知识显式地组织在 Playbook 中。

---

## AI2Pot-Playbook 是什么？

AI2Pot-Playbook 描述的是：

> **AI2Pot 应该如何开发。**

而 AI2Pot 仓库本身提供的是：

> **AI2Pot 当前已经实现了什么。**

两者的职责不同：

```text
AI2Pot-Playbook
    ↓
架构规则
开发约定
可复用工作流
Agent Skills

AI2Pot
    ↓
核心实现
参考模型
计算后端
训练与部署基础设施
```

典型使用场景包括：

* 维护和重构 AI2Pot；
* 基于 AI2Pot 实现新的机器学习模型；
* 将已有的原子尺度机器学习模型迁移到 AI2Pot 架构；
* 添加新的 Dataset、Model、Trainer、Converter 或 Operator；
* 指导 Coding Agent 按照 AI2Pot 的架构规范完成开发任务。

---

## 推荐的 Workspace 结构

AI2Pot-Playbook 通常与 AI2Pot 仓库以及待开发的模型仓库一起使用。

推荐的目录结构为：

```text
workspace/
├── AI2Pot-Playbook/
├── AI2Pot/
└── AI2Pot-PaiNN/
```

如果还需要参考某个模型的原始实现，可以进一步加入 upstream repository：

```text
workspace/
├── AI2Pot-Playbook/
├── AI2Pot/
├── PaiNN-reference/
└── AI2Pot-PaiNN/
```

各个仓库承担不同的职责：

```text
AI2Pot-Playbook
    → 规定代码应该如何设计和实现

AI2Pot
    → 提供现有实现和可复用基础设施

PaiNN-reference
    → 提供原始算法和参考实现

AI2Pot-PaiNN
    → 实际需要完成开发的目标仓库
```

---

## 与 Claude Code 配合使用

通常应从 **AI2Pot-Playbook 根目录启动 Claude Code**。

原因是 AI2Pot-Playbook 中的 `.claude/` 目录包含项目所需的 Rules 和 Skills。

例如：

```bash
cd workspace/AI2Pot-Playbook

claude \
    --add-dir ../AI2Pot \
    --add-dir ../AI2Pot-PaiNN
```

如果同时需要参考原始模型仓库：

```bash
cd workspace/AI2Pot-Playbook

claude \
    --add-dir ../AI2Pot \
    --add-dir ../PaiNN-reference \
    --add-dir ../AI2Pot-PaiNN
```

此时 Coding Agent 可以同时访问：

```text
AI2Pot-Playbook
    → 架构规则和开发技能

AI2Pot
    → AI2Pot 的参考实现

PaiNN-reference
    → 上游模型的参考实现

AI2Pot-PaiNN
    → 实际开发目标
```

随后可以直接向 Agent 提出类似任务：

```text
请在 AI2Pot-PaiNN 中实现 PaiNN。

开发过程中遵循 AI2Pot-Playbook 中定义的架构规则和开发规范。

尽可能复用 AI2Pot 中已有的基础设施，
PaiNN-reference 主要用于参考模型结构、算法和数学实现。
```

---

## 仓库结构

AI2Pot-Playbook 主要由 **Rules** 和 **Skills** 两部分组成：

```text
AI2Pot-Playbook/
└── .claude/
    ├── rules/
    │   └── architectures/
    │       ├── dataset.md
    │       ├── model.md
    │       ├── neighbor-list.md
    │       └── trainer.md
    │
    └── skills/
        └── ...
```

### Rules

`rules/` 用于定义 AI2Pot 长期保持稳定的架构约束和开发规范。

例如：

* Dataset 接口；
* AI2Pot 七张量模型接口；
* Neighbor List 索引规范；
* Model 与 LitModel 的职责；
* 训练流程规范。

Rules 主要回答：

> **AI2Pot 中哪些设计应该保持一致？**

### Skills

`skills/` 用于定义完成特定开发任务时可以复用的工作流程。

例如：

```text
implement-new-model
port-model-to-ai2pot
add-dataset
add-operator
debug-cuda-operator
validate-model
```

Skills 主要回答：

> **某一类开发任务应该按照什么步骤完成？**

---

## 核心开发原则

实现新的模型时，应优先遵循：

```text
复用 AI2Pot 基础设施
          ↓
遵循 AI2Pot 公共接口
          ↓
在模型内部构建模型特定的数据表示
          ↓
实现模型自身的数学和物理逻辑
          ↓
复用 AI2Pot 的训练与部署基础设施
          ↓
与参考实现进行验证
```

不要仅为了简化某一个模型的实现，就修改 AI2Pot 的公共基础设施。

当上游模型的代码结构与 AI2Pot 架构发生冲突时，应：

> **保留上游模型的算法行为，同时将其软件架构适配到 AI2Pot 的开发规范。**

---

## 开发理念

AI2Pot-Playbook 将软件架构和开发经验视为可以持续积累和复用的工程知识。

模型可以变化。

内部表示可以变化。

具体实现可以变化。

但 AI2Pot 的公共接口和核心架构原则应尽可能保持稳定。
