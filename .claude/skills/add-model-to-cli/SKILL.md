# Add Model to AI2Pot-cli

本 Skill 用于当 AI2Pot（或独立模型仓库）中新增 / 迁移了一个模型后，为 AI2Pot-cli 接入该模型的三类用户功能：

```text
训练文件生成
ai2pot-cli train 支持
后处理工具
```

前置条件（模型侧已完成，参考 `port-model-to-ai2pot` 与 `rules/architectures/`）：

```text
模型核心 Model 已完成且可训练
LitModel 已完成（训练超参数会成为 checkpoint hyper_parameters）
model_utils 三件套已完成（接入后处理前必须）
```

长期约束参考 `.claude/rules/architectures/cli.md`，本 Skill 描述执行顺序与验证方式。

---

# 1. 开始前

1. 阅读 `.claude/rules/architectures/cli.md` 与 `model_utils.md`；
2. 确定模型实现所在位置与 CLI 中的 import 路径：
   * 模型并入 AI2Pot → `ai2pot.models.<model>.<model>_utils`；
   * 模型位于独立仓库 → `<model_pkg>.<model>_utils`（如 nnmtp 的 `nnmtp.nnmtp_utils`）；
3. 参考已有接入：
   * mtp / nep / nnmtp —— 模型位于 AI2Pot 内的接入范例。
   * painn —— 独立仓库模型的完整接入范例；

CLI 只做编排与调度，不在 CLI 中复制模型代码或 AI2Pot 基础设施。

---

# 2. 训练文件生成

在 AI2Pot-cli 仓库中完成：

1. **模板** `ai2pot_cli/templates/<model>_train.jsonc`
   * 参考 `templates/nnmtp_train.jsonc` 的结构；
   * 配置统一为 Trainer / Model / Dataset / Resume 四大节；
   * 每个字段带注释（含义、范围、默认值），模板即模型参数文档；
   * 文件名必须为 `<model>_train.jsonc`。

2. **生成函数** `ai2pot_cli/menus/potential_train/<model>_train_input.py`
   * 定义 `generate_<model>_input(output_path: str = "<model>_train.jsonc")`；
   * 行为：把内置模板拷贝到当前目录，提示下一步命令 `ai2pot-cli train --input <output_path>`。

3. **菜单接入** `ai2pot_cli/main.py`
   * `MAIN_SECTIONS` 的 "Potential Training Input" 小节加入新条目（编号接续现有条目）；
   * 增加对应的 `elif` 分发分支。

---

# 3. ai2pot-cli train 支持

修改 `ai2pot_cli/train.py` 的 `run_train()`：

1. **模型类型识别**：在检测 `elif` 链中加入新分支；
   * 确认模型特征键唯一且不与既有模型冲突（对照 cli.md 登记表）；
   * 若键存在包含关系，更具体的组合放在更宽泛的判断之前；
2. **rcut 校验**：加入 `Dataset.rcut >= 模型自身截断` 的校验分支（模型无自身截断则跳过）；
3. **LitModel 构造**：从模型侧 import 并实例化 LitModel；
   * 通用超参（type_map / umax_num_neigh_atoms / fit_virial / lr / 权重）遵循 `common_kwargs` 既有模式；
   * 模型特有超参直接传参；
   * 确保构造参数（即检测键）会进入 checkpoint 的 `hyper_parameters`；
4. **callback**：模型有 descriptor 归一化则注册对应的 NormCallback；没有则跳过（painn 先例：直接不注册）。

不要重写 `run_train` 的通用流程（配置加载、ExtxyzDataModule、Trainer、CSVLogger、ModelCheckpoint、resume 处理）。

---

# 4. 后处理

`ai2pot_cli/menus/postprocessing/` 下按模型分发的位置加入新模型分支：

| 菜单工具 | 文件 | 需要模型侧提供 |
| --- | --- | --- |
| Export TorchScript Model | `serialize_model.py` | `<Model>Serializer` |
| Plot E/F/V Parity | `plot_parity.py` | `<Model>4Extxyz` |
| Plot Descriptor Projection | `plot_descriptors.py` | `<Model>4Extxyz` + `calculate_descriptors()` |

每个文件只需在其模型类型识别 / `_build_model` 函数中增加一个分支，复用模型侧 `model_utils` 类。

`plot_trainlog.py` 与模型无关，通常无需改动。

若模型 utils 尚未提供上述能力，先补齐 model_utils（参考 `rules/architectures/model_utils.md`），再接后处理。

---

# 5. 文档同步与验证

1. 同步更新 AI2Pot-cli README 中的交互菜单文本与用法说明（README 内嵌菜单列表，容易遗漏）；
2. 验证清单：
   * 交互菜单出现新条目，`generate_<model>_input()` 能生成 `<model>_train.jsonc`；
   * 检查生成的 jsonc 各节键与 `train.py` 的识别 / 构造逻辑一致；
   * CPU 冒烟：`ai2pot-cli train --input <model>_train.jsonc`（小数据集、1-2 epoch）；
   * 后处理冒烟：对冒烟训练的 checkpoint 运行 parity / descriptors / serialize（参考 `tests/test_postprocessing_smoke.py` 的模式）；
   * 模型在独立仓库时，确认运行 CLI 的 Python 环境已安装该模型包；
   * 改动通用骨架时，对既有模型（mtp / nep / nnmtp）做回归冒烟；
3. 把新模型的检测键登记回 cli.md 登记表（Playbook 仓库可写时）。

---

# 6. 不要做的事情

* 在 CLI 内复制 / 重实现模型代码，或 AI2Pot 的数据读取 / Neighbor List / descriptor 逻辑；
* 为单个模型重写 `run_train` 通用流程、配置文件结构或菜单框架；
* 跳过 model_utils 直接接入后处理菜单；
* 使用与既有模型冲突的检测键，或依赖 config 中存在但不会进入 checkpoint `hyper_parameters` 的键做模型识别；
* 为模型另造与 `ai2pot-cli train --input` 并行的独立训练入口。

---

# 7. 完成标准

```text
用户可在交互菜单生成 <model>_train.jsonc
        ↓
ai2pot-cli train --input <model>_train.jsonc 可完成训练
        ↓
后处理工具（parity / descriptors / serialize）可运行
        ↓
检测键已登记，无与既有模型的识别冲突
        ↓
CLI 通用骨架未被模型专用逻辑污染
```
