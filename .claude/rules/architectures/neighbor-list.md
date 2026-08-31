# AI2Pot 中 Neighbor List 的约定

本文档定义了基于 AI2Pot 开发机器学习模型时应遵循的 Neighbor List 规范。

Neighbor List 同时连接 Dataset 输出与 Model 输入，因此其索引语义必须保持一致。以下规则应视为长期设计约束，除非用户明确要求进行架构调整，否则不应随意修改。

# 1. Neighbor List 的遍历规范

遍历 Neighbor List 时，应始终通过 `ilist` 获取中心原子的真实索引：

```cpp
int center_idx;
int type_central;
int Zi;

int neigh_idx;
int type_outer;
int Zj;

CoordType neigh_vec[3] = {0.0};

for (int ii = 0; ii < inum; ii++) {
    center_idx = ilist[ii];

    type_central = types[center_idx];
    Zi = type_map[type_central];

    for (int jj = 0; jj < numneigh[center_idx]; jj++) {
        neigh_idx = firstneigh[center_idx * umax_num_neigh_atoms + jj];

        type_outer = types[neigh_idx];
        Zj = type_map[type_outer];

        for (int aa = 0; aa < 3; aa++) {
            neigh_vec[aa] = rcs[center_idx * umax_num_neigh_atoms + jj][aa];
        }
    }
}
```

其中：

* `ii`：`ilist` 中的位置，不代表真实原子索引。
* `center_idx = ilist[ii]`：中心原子的真实索引。
* `neigh_idx`：近邻原子的真实索引。
* `numneigh`、`firstneigh`、`rcs` 和 `types` 均应使用真实原子索引访问。

因此，不应默认：

```cpp
ii == center_idx
```

# 2. 原子属性的索引规范

与原子对应的属性，例如：

```text
force
atomic energy
charge
magnetic moment
```

均应按照真实原子索引进行读写，即使用：

```cpp
center_idx
neigh_idx
```

而不是 Neighbor List 遍历索引 `ii` 或近邻序号 `jj`。

例如，对中心原子写入力：

```cpp
forces[center_idx * 3 + aa] += value;
```

对近邻原子写入力：

```cpp
forces[neigh_idx * 3 + aa] += value;
```

除非某个数组被明确设计为按照 `ilist` 顺序存储，否则原子属性默认均与 `types` 的原子索引顺序保持一致。
