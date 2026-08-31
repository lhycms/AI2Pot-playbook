# Remote Develop

本 Skill 用于本地无法完成编译、训练或测试，而远程服务器具备运行环境但无法安装 Claude Code 的开发场景。

典型工作目录：

```text
workspace/
├── AI2Pot-Playbook/
├── AI2Pot/
├── Upstream-Model/
└── Target-Implementation/
```

Claude Code 在本地 `AI2Pot-Playbook` 中运行，通过 SSH 使用远程服务器完成编译、测试、训练和数值验证。

---

# 1. 核心原则

远程开发应遵循：

```text
本地
AI2Pot-Playbook
      +
AI2Pot
      +
Upstream Model
      +
Target Implementation
          │
          │ 修改代码
          ▼
      Local Source
          │
          │ sync
          ▼
Remote Execution Environment
          │
          │ compile / test / train
          ▼
        Logs
          │
          └────────→ Local Agent
```

**本地仓库始终作为源码真值（Source of Truth）。**

远程服务器仅作为独立的执行环境，主要用于：

- 编译；
- 单元测试；
- CUDA 测试；
- 数值验证；
- 模型训练；
- 性能测试；
- LAMMPS 测试。

除非用户明确要求，否则不要长期直接修改远程服务器上的源码。

---

# 2. 创建独立的远程 Python 环境

远程开发时，**默认必须为当前项目创建独立的 Python / Conda 环境**。

不要直接使用：

```text
base
已有 AI2Pot 环境
其他项目的 Conda 环境
系统 Python
```

除非用户明确要求复用已有环境。

开始前应检查：

```text
SSH host
远程 workspace 路径
AI2Pot 的 Python / PyTorch 要求
Upstream Model 的环境要求
CUDA / Driver
C++ / CUDA 编译器
```

优先检查：

```bash
ssh <host> "hostname"
ssh <host> "nvidia-smi"
ssh <host> "nvcc --version"
ssh <host> "conda --version"
```

根据 AI2Pot、Upstream Model 和服务器 CUDA 环境确定兼容的 Python / PyTorch 版本，然后创建独立环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh

conda create -n <project-env> python=<python-version> -y
conda activate <project-env>
```

环境名称应能够反映当前任务，例如：

```text
ai2pot-painn
ai2pot-mattergen
ai2pot-dpa
```

不要默认固定 Python 版本。

环境创建完成后，后续所有：

```text
pip install
cmake
C++ / CUDA 编译
pytest
training
numerical validation
benchmark
LAMMPS test
```

均应显式激活该独立环境。

不要假设非交互 SSH shell 会自动加载 `.bashrc` 或 Conda 环境。

---

# 3. 安装项目依赖

优先根据项目已有配置安装依赖：

```text
pyproject.toml
requirements.txt
environment.yml
setup.py
README
```

如果 AI2Pot 需要以开发模式安装，优先使用：

```bash
pip install -e <remote-workspace>/AI2Pot
```

如果 Target Implementation 也需要安装：

```bash
pip install -e <remote-workspace>/Target-Implementation
```

未经用户明确允许，不要：

```text
修改 base 环境
升级其他项目环境中的 PyTorch
sudo pip install
使用系统级 pip install
破坏已有 Conda 环境
```

当前开发任务应能够在不修改服务器已有 Python 环境的情况下完成。

---

# 4. 本地负责代码修改

Agent 应在本地完成：

```text
阅读 AI2Pot-Playbook
        ↓
分析 AI2Pot
        ↓
分析 Upstream Model
        ↓
修改 Target Implementation
```

架构设计、代码修改和代码审查均应优先在本地完成。

修改完成后，再将需要运行的代码同步到服务器。

---

# 5. 同步代码到服务器

优先使用 `rsync`。

例如：

```bash
rsync -az \
    ../Target-Implementation/ \
    <host>:<remote-workspace>/Target-Implementation/
```

如果同时修改了 AI2Pot：

```bash
rsync -az \
    ../AI2Pot/ \
    <host>:<remote-workspace>/AI2Pot/
```

如果测试需要 Upstream Model，也可以同步其代码。

通常**不需要同步 AI2Pot-Playbook**，因为 Playbook 主要供本地 Agent 使用。

不要默认使用：

```bash
rsync --delete
```

除非明确确认远程目录只是临时镜像。

避免覆盖：

```text
训练数据
checkpoint
编译缓存
测试结果
用户已有文件
```

必要时使用：

```bash
--exclude ".git"
--exclude "build"
--exclude "checkpoints"
--exclude "outputs"
```

---

# 6. 远程命令通过 SSH 执行

服务器不需要安装 Claude Code。

本地 Agent 应通过 SSH 执行远程命令：

```bash
ssh <host> "<command>"
```

复杂命令应显式加载并激活当前项目环境：

```bash
ssh <host> '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate <project-env>

    cd <remote-workspace>/Target-Implementation

    <command>
'
```

例如：

```bash
ssh <host> '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ai2pot-painn

    cd ~/workspace/AI2Pot-PaiNN

    pytest tests/test_painn.py -v
'
```

---

# 7. 优先运行最小测试

远程测试应遵循：

```text
最小测试
   ↓
定位问题
   ↓
本地修改
   ↓
重新同步
   ↓
再次测试
```

推荐顺序：

```text
import test
    ↓
single forward
    ↓
shape / dtype test
    ↓
gradient test
    ↓
unit test
    ↓
small training test
    ↓
numerical comparison
    ↓
full training / benchmark
```

不要在基础错误尚未解决时直接运行长时间训练或大规模 benchmark。

---

# 8. 根据远程结果继续本地开发

运行远程命令后，应读取：

```text
stdout
stderr
exit code
test result
numerical difference
```

标准迭代过程：

```text
Local Edit
    ↓
Sync
    ↓
Remote Test
    ↓
Read Error
    ↓
Local Fix
    ↓
Sync
    ↓
Remote Test
```

直到测试通过。

不要仅根据静态代码分析判断 CUDA、C++ 或数值实现已经正确。

---

# 9. C++ / CUDA 编译

对于 C++ / CUDA 扩展，应在服务器完成实际编译验证。

例如：

```bash
ssh <host> '
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate <project-env>

    cd <remote-workspace>/AI2Pot

    mkdir -p build
    cd build

    cmake ..
    cmake --build . -j
'
```

普通代码修改优先增量编译。

只有怀疑以下问题时再执行 clean build：

```text
CMake 配置
ABI
CUDA architecture
旧编译缓存
编译选项变化
```

不要无理由频繁删除整个 `build` 目录。

---

# 10. 数值验证

移植模型时，应尽可能比较：

```text
AI2Pot Implementation
        vs.
Upstream Implementation
```

优先比较：

```text
输入
Neighbor List / Graph
中间 representation
中间 feature
energy
forces
gradient
loss
```

如果结果不一致，应首先检查：

```text
dtype
unit
type mapping
Neighbor List convention
edge direction
relative vector direction
PBC
padding
normalization
cutoff
aggregation
```

不要通过放宽 tolerance 掩盖明显的数值错误。

---

# 11. 远程文件修改原则

默认：

> **不要直接在远程服务器修改核心源码。**

如果为了快速诊断必须临时修改远程文件：

1. 明确该修改仅用于诊断；
2. 将正式修改重新应用到本地仓库；
3. 重新同步本地版本；
4. 确保最终服务器代码与本地一致。

不要让远程服务器形成未同步回本地的隐藏版本。

---

# 12. 大文件和训练结果

训练数据、checkpoint 和大型输出文件通常应保留在服务器。

不要无条件同步：

```text
datasets
checkpoints
trajectories
build artifacts
large logs
```

只有在本地分析确实需要时，才同步必要文件。

例如：

```bash
rsync -az \
    <host>:<remote-path>/result.json \
    ./results/
```

---

# 13. 安全约束

未经用户明确允许，不要执行：

```text
删除远程项目目录
删除训练数据
删除 checkpoint
修改其他用户文件
修改系统环境
sudo 操作
全局安装软件
修改或删除其他 Conda 环境
```

对于以下破坏性操作应格外谨慎：

```bash
rm -rf
rsync --delete
git reset --hard
git clean -fdx
```

---

# 14. 推荐工作流程

```text
读取 AI2Pot-Playbook
        ↓
分析 AI2Pot / Upstream
        ↓
检查远程 CUDA / 编译环境
        ↓
创建独立 Conda 环境
        ↓
同步代码
        ↓
安装 AI2Pot / Target 依赖
        ↓
在本地修改代码
        ↓
同步 Target 到服务器
        ↓
运行最小远程测试
        ↓
读取错误和数值结果
        ↓
本地修复
        ↓
重新同步
        ↓
再次测试
        ↓
单元测试通过
        ↓
数值验证
        ↓
训练 / 性能测试
```

最终目标是：

> **让 Coding Agent 在本地负责理解、设计和修改代码，并将远程服务器作为隔离的编译与计算后端，从而在服务器无法运行 Claude Code 的情况下，仍然完成完整、可复现的开发闭环。**