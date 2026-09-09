# AI2Pot-cli 中模型的接入规范

本文件定义新增模型接入 AI2Pot-cli 时应遵循的规范。

AI2Pot-cli 是 AI2Pot 的官方命令行工具，提供交互式菜单与 `ai2pot-cli` 子命令两种模式，覆盖数据预处理、训练输入生成、模型训练、后处理与 MD 工具等面向用户的入口。

以下规则应视为长期设计约束。除非用户明确要求进行架构层面的调整，否则不应为了单个模型的接入便利而修改 CLI 的通用骨架。

# 1. 职责边界

AI2Pot-cli 负责编排：

```text
训练输入生成
配置解析与模型类型识别
训练调度（ai2pot-cli train）
后处理与工具入口
```

AI2Pot-cli 不负责：

```text
模型核心计算
Descriptor / Graph 构建
数据读取与 Neighbor List
训练循环内部细节
```

模型侧代码位于 AI2Pot（`ai2pot/models/<model>/...`）或独立模型仓库（例如 `painn` 包），CLI 只负责 import 与调度：

```text
ai2pot-cli
   ├── generate_<model>_input → <model>_train.jsonc
   ├── ai2pot-cli train --input <model>_train.jsonc
   │        ↓
   │    run_train（通用流程）
   │        ↓
   │    LitModel（来自模型侧）+ ExtxyzDataModule
   └── 后处理（parity / descriptors / serialize）
            ↓
   模型侧 model_utils（Model4Extxyz / Serializer）
```

不要将模型实现复制进 AI2Pot-cli，也不要在 CLI 中重复实现 AI2Pot 已有基础设施。

# 2. 新增模型必须完成的三类接入

新增或迁移模型时，除模型本体、LitModel 与 model_utils 之外，还应完成 AI2Pot-cli 的三类接入：

```text
训练文件生成
    ↓
ai2pot-cli train 支持
    ↓
后处理工具
```

| 功能 | CLI 侧实现位置 |
| --- | --- |
| 训练文件生成 | `ai2pot_cli/templates/<model>_train.jsonc`、`ai2pot_cli/menus/potential_train/<model>_train_input.py`、`ai2pot_cli/main.py` 菜单与分发 |
| `ai2pot-cli train` | `ai2pot_cli/train.py` 中的模型类型识别、rcut 校验、LitModel 构造、callback 分支 |
| 后处理工具 | `ai2pot_cli/menus/postprocessing/` 下各工具的模型类型识别与模型侧导入分支 |

面向用户的训练入口应统一走 `ai2pot-cli train --input ...`，不要为目标模型编写并行的独立训练脚本。

# 3. 训练文件生成规范

* 模板文件位于 `ai2pot_cli/templates/`，文件名为 `<model>_train.jsonc`；
* 模板必须携带完整注释，逐字段说明含义、取值范围与默认值，模板本身即模型的参数文档；
* 生成函数位于 `ai2pot_cli/menus/potential_train/<model>_train_input.py`，形式为：

```python
def generate_<model>_input(output_path: str = "<model>_train.jsonc"):
    ...
```

其行为为：将内置模板拷贝到当前目录，并提示下一步命令 `ai2pot-cli train --input <output_path>`。

* 在 `ai2pot_cli/main.py` 的菜单与分发中加入对应条目，编号沿用现有菜单风格；
* 训练配置使用 JSONC（支持 `//` 与 `/* */` 注释），结构统一为：

```text
{
    "Trainer": {...},
    "Model": {...},
    "Dataset": {...},
    "Resume": {...}
}
```

新模型不应引入与既有模型结构不同的配置文件格式。

# 4. 模型类型识别与唯一检测键

CLI 通过特征键识别模型类型：

* `ai2pot_cli/train.py` 从训练配置的 `Model` 段识别；
* 后处理工具从 checkpoint 的 `hyper_parameters` 识别。

每个模型必须拥有互不冲突、可唯一区分的检测键组合。已登记键：

| model_type | 检测键 | 说明 |
| --- | --- | --- |
| painn | `num_interactions` + `hidden_state_size` | 独立仓库接入范例 |
| nnmtp | `mtp_level` + `num_neurons` | 必须先于 mtp 判断 |
| mtp | `mtp_level`（无 `num_neurons`） | |
| nep | `n_radial_basis` | |
| 新模型 | 在此登记 | |

约束：

* 新模型的检测键不得与已有键混淆；检测分支应放在 `elif` 链中不与已有键冲突的位置；
* 若新模型的键与已有键存在包含关系，更具体的组合必须写在更宽泛的检测之前；
* checkpoint 侧检测依赖 LitModel 保存的 `hyper_parameters`，因此 LitModel 的构造参数中必须包含这些特征键，否则后处理工具无法识别已训练模型；
* 新增模型时同步更新本登记表（若 Playbook 仓库不可写，至少在任务报告中说明检测键与登记建议）。

# 5. rcut / 截断一致性校验

`ai2pot-cli train` 启动训练前应校验 `Dataset.rcut` 不小于模型自身的截断半径。已有模型：

```text
mtp / nnmtp → Model.rmax
nep         → Model.rmax_radial
painn       → Model.cutoff
```

新模型若具有自身的径向 / 角向 / 边截断，必须在 `run_train` 中加入对应校验分支。

# 6. 后处理工具与 model_utils 的依赖关系

后处理菜单应复用模型侧 `model_utils`，而不是在 CLI 中重新实现：

```text
Plot E/F/V Parity           → <Model>4Extxyz
Plot Descriptor Projection  → <Model>4Extxyz + calculate_descriptors()
Export TorchScript Model    → <Model>Serializer
```

接入条件：

* 模型完成 `model_utils` 三件套（`ModelSerializer` / `Model4Extxyz` / `ModelCalculator`）之前，不接入后处理菜单；
* Descriptor Projection 依赖模型侧提供 `calculate_descriptors()`，模型不支持时该工具对该模型不可用，不应在 CLI 中另实现 descriptor 计算；
* 模型位于独立仓库时，CLI 从该仓库导入（例如 `from painn.painn_utils import PaiNNSerializer`）；模型位于 AI2Pot 时，从 `ai2pot.models.<model>.<model>_utils` 导入。

# 7. 保持 CLI 通用骨架不变

`run_train` 的通用流程（配置加载、ExtxyzDataModule、Trainer、CSVLogger、ModelCheckpoint、resume 的 metrics 合并等）与菜单框架为所有模型共用。

新增模型只添加自身的分支：

```text
模型类型识别
rcut 校验
LitModel 构造
descriptor-norm callback 选择（如适用）
后处理工具中的识别与导入分支
```

不要为单个模型修改：

```text
配置文件结构
训练入口形式
通用 callback 机制
菜单框架与通用后处理绘图逻辑
```

# 8. 测试

接入完成后，应在 AI2Pot-cli 仓库中验证：

* 训练输入生成可生成合法 `<model>_train.jsonc`；
* `ai2pot-cli train --input` 可用小规模数据集在 CPU 上完成冒烟训练；
* 后处理工具可在冒烟训练的 checkpoint 上运行（参考 `tests/test_postprocessing_smoke.py` 的模式）；
* 若修改了通用骨架，应对既有模型（mtp / nep / nnmtp / painn）进行回归冒烟。

参考实现：AI2Pot-cli 中 painn 是最近一次完整接入，可对照 `mtp` / `nep` / `nnmtp` 理解既有约定。
