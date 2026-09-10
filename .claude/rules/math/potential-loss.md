# 势函数训练中的 Loss 与 RMSE 规范

本文档定义 AI2Pot 势函数模型训练时，Loss（用于反向传播）与 RMSE（用于日志记录）的计算约定。

两者不是同一个量：

```text
Loss
    → 每个 configuration 的加权 MSE，batch 内取平均，用于反向传播

RMSE
    → 在一个 batch 内聚合的误差统计量，不加 loss 权重，用于 Lightning logger
```

# 1. 损失函数

训练时的损失函数（用于反向传播）主要以单个 configuration 的 MSE 为计算主体，并在结构内部按真实原子数 $N$ 归一化：

Energy Loss:
$$L_E = \frac{1}{N_{atom}}(E^{ML} - E^{DFT})^2$$

Force Loss:
$$L_F = \frac{1}{3N_{atom}} \sum_i \sum_{\alpha}{ (F^{ML}_{i,\alpha} - F^{DFT}_{i,\alpha})^2 }$$

Virial Loss:
$$L_V = \frac{1}{9N_{atom}} \sum_{\alpha}\sum_{\beta}(V^{ML}_{\alpha, \beta} - V^{DFT}_{\alpha, \beta})^2$$

单个 configuration 的损失为三项加权和：

$$L_b = w_E L_E + w_F L_F + w_V L_V$$

batch 损失取 batch 内所有结构损失的算术平均，即反向传播实际使用的量：

$$L_{batch} = \frac{1}{B}\sum_{b=1}^{B} L_b$$

约定：

* 归一化分母是结构 $b$ 的真实原子数 $inum$，不含 ghost 原子（参考 `neighbor-list.md`）；
* batch 内各结构等权（除以 $B$），不按原子数加权；
* 力与 virial 的求和只遍历真实原子（`ilist`），并使用真实原子索引读写；
* `fit_virial=False` 时去掉 $L_V$ 项，损失为 $L_E + L_F$；
* 反向传播由 C++ op 的 backward 实现（`find_loss_backward` / `find_ef_loss_backward`），Python 侧不重新实现 loss 的梯度。

## 权重退火

$w_E, w_F, w_V$ 不是常数，而是随学习率从 start 值线性退火到 end 值：

$$w_E = w_{E,start}\cdot r + w_{E,end}\cdot(1-r), \qquad r = \frac{lr}{lr_{start}}$$

因此：

> 训练早期与晚期，同样的物理误差对损失的贡献不同。
> logger 中的 `train/mse` 是加权后的损失，其下降同时包含“拟合变好”和“权重变化”两个因素，不能单独用来判断拟合程度。

# 2. 日志中的 RMSE

logger 记录的 RMSE 是用一个 batch 内的所有结构聚合计算的（C++ launcher），由 batch 内的原始误差直接汇总，而不是对第 1 节的加权 MSE 做简单变换得到的。

对 batch 内 $B$ 个结构：

Energy:
$$\mathrm{e\_rmse} = \sqrt{\frac{1}{B}\sum_b \left(\frac{E^{ML}_b - E^{DFT}_b}{N_b}\right)^2}$$

Force:
$$\mathrm{f\_rmse} = \sqrt{\frac{\sum_b \sum_{i \in b}\sum_{\alpha}(F^{ML}_{i,\alpha} - F^{DFT}_{i,\alpha})^2}{3\sum_b N_b}}$$

Virial:
$$\mathrm{v\_rmse} = \sqrt{\frac{1}{9B}\sum_b \sum_{\alpha\beta}\left(\frac{V^{ML}_{\alpha\beta,b} - V^{DFT}_{\alpha\beta,b}}{N_b}\right)^2}$$

三种量的聚合方式并不相同：

| 量 | 聚合方式 | 结构间权重 |
| --- | --- | --- |
| `e_rmse` | 结构内先除以 $N_b$（per-atom 能量误差），平方后对结构取平均 | 各结构等权 |
| `v_rmse` | 结构内先除以 $N_b$（per-atom virial 误差），平方后对结构取平均 | 各结构等权 |
| `f_rmse` | 不先按结构平均，batch 内所有真实原子的分量误差直接汇总（pooling），再除以总分量数 | 按原子数加权 |

由此可得：

* `e_rmse` 的平方恰为 batch 内各结构 per-atom energy MSE 的算术平均；
* `f_rmse` 的平方为**按原子数加权**的 per-structure force MSE 平均 —— 各结构原子数不等时，与“结构等权平均”不同；
* `v_rmse` 与 $L_V$ 的差别是分子中额外的 $1/N_b$（virial 先转为 per-atom 量），因此量纲是 per-atom virial；
* RMSE 不乘 $w_E, w_F, w_V$，反映的是物理误差，不随退火权重变化；
* RMSE 与 `train/mse` 之间没有固定换算关系（后者加权，且处于模型归一化单位）。

其它约定：

* 能量取结构总能量，不是 per-atom 能量；ML 与 DFT 两侧同样处理；
* 只统计真实原子（`inum` / `ilist`），ghost 原子不参与；
* 计算 dtype 跟随输入张量（float32 → float32，float64 → float64）；
* 不要用“逐结构开根号再平均”替代 batch 聚合，二者不等价（mean of RMSEs ≠ RMSE of pooled errors）。

# 3. conv（conversion）因子与单位换算

`conv_energy` / `conv_length` 中的 `conv` 指 **conversion（换算因子）**，不是统计意义上的归一化（如除以标准差）。

它实现的是 target 的归一化：把数据集（物理单位，如 eV / Å）的 target 换算到模型内部采用的单位体系（代码中经 conv 换算后的量以 `_norm` 为后缀）：

```text
内部单位的值 = 物理单位的值 × conv
物理单位的值 = 内部单位的值 / conv
```

* 训练时，DFT 标签以及模型侧以物理单位存储的量（如 `type_bias`、`rmax`、`zbl_*`）都要先乘 conv 因子换算到核内部单位，再送入 C++ 计算；核输出的预测值除以 conv 因子回到物理单位；
* 力的换算因子由能量与长度组合得到，virial 与能量同量纲：

```text
conv_force  = conv_energy / conv_length
conv_virial = conv_energy
```

* RMSE 关于输入是齐次量，因此在模型 Python 侧除以对应 conv 因子即可还原为物理单位：

```text
e_rmse / conv_energy
f_rmse / conv_force
v_rmse / conv_virial
```

* 当前实现中两个 conv 均为 1.0，即内部单位与数据集单位一致（eV / Å）。

> 新模型若引入自己的单位换算，必须保证最终记录到 logger 的 RMSE 是物理单位下的量。

# 4. 记录与聚合

LitModel 在 training / validation / test step 中记录：

```text
train/mse
    ← bmse_tensor.mean()，即反向传播使用的损失（加权、模型归一化单位）

train/e_rmse
train/f_rmse
train/v_rmse     （fit_virial=True 时）
```

* 记录参数为 `on_step=True, on_epoch=True, sync_dist=True`；
* epoch 级别的数值是各 step 值的算术平均（Lightning 默认 mean 归约，未使用 `weight_avg`），即**各 batch RMSE 的平均**，而不是整个 epoch 数据集上的全局 RMSE；各 batch 原子数不等或最后一个 batch 较小时，会与全局 RMSE 存在偏差；
* batch 维度的聚合在 C++ 内完成，每步只回传一个标量。

# 5. 新增模型时的要求

1. Loss 保持“per-configuration + 按原子数归一化 + batch 内结构等权平均”的形式，并支持权重退火；
2. RMSE 在 batch 维聚合：E / V 为 per-atom 量且结构等权，F 为全体分量 pooling；
3. RMSE 不带 loss 权重，且以物理单位记录；
4. 只对真实原子统计，索引使用真实原子索引（参考 `neighbor-list.md`）；
5. 若模型引入新的 loss 分量（电荷、自旋等），应在模型文档中明确其归一化方式、聚合方式与单位；
6. 修改 loss / rmse 的公共语义属于架构变更，需要先评估跨模型的影响，而不是在单个模型内自行其是。
