# 基于 AI2Pot 的 Dataset 规范

本文档定义了基于 AI2Pot 开发机器学习模型时，Dataset 应遵循的基本规范。

以下规则应视为长期设计约束。除非用户明确要求进行架构层面的调整，否则不应为了单个模型的实现便利而随意修改 Dataset 的核心接口。

# 1. Dataset 基本要求

所有 Dataset 应继承：

```python
torch.utils.data.Dataset
```

在非必要情况下，Dataset 的初始化参数应尽量保持以下形式：

```python
filename: str
rcut: float
umax_num_neigh_atoms: int = 200
pbc_xyz: List[bool] = [True, True, True]
sort: bool = False
torch_float_dtype: torch._C.dtype = torch.float32
has_virial: bool = False
```

其中，模型不需要的参数可以省略。例如，对于不涉及 virial 的任务，可以不提供 `has_virial`。

Dataset 应至少实现：

```python
__len__()
__getitem__()
_get_max_num_atoms()
_get_type_map()
get_type_map()
analyse_pymatgen()
analyse_ase()
```

其中：

* `_get_max_num_atoms()`：获取数据集中单个结构的最大原子数，用于 padding。
* `_get_type_map()`：建立原子序数与 AI2Pot 内部原子类型索引之间的映射。

# 2. Dataset 输出接口

`Dataset.__getitem__()` 应优先返回可以直接作为 AI2Pot `LitModel` 输入的数据。

对于标准原子势模型，前 7 个返回值应与 AI2Pot Core Atomistic Interface 保持一致：

```python
inum
ilist
numneigh
firstneigh
relative_coords
types
nghost
```

对应模型侧的：

```python
binum_tensor
bilist_tensor
bnumneigh_tensor
bfirstneigh_tensor
brcs_tensor
btypes_tensor
bnghost_tensor
```

随后再附加任务所需的监督数据，例如：

```python
energy
forces
virial
```

因此，势函数 Dataset 的典型输出形式为：

```python
[
    inum,
    ilist,
    numneigh,
    firstneigh,
    relative_coords,
    types,
    nghost,
    energy,
    forces,
]
```

如果包含 virial，则可以扩展为：

```python
[
    inum,
    ilist,
    numneigh,
    firstneigh,
    relative_coords,
    types,
    nghost,
    energy,
    forces,
    virial,
]
```

对于其他类型的机器学习任务，可以在保持前 7 个核心输入不变的基础上，添加模型所需的其他标签或任务相关数据。

# 3. 推荐实现

```python
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read
from torch.utils.data import Dataset

from ai2pot.core import Nblist


class ExtxyzDataset(Dataset):
    def __init__(
        self,
        filename: str,
        rcut: float,
        umax_num_neigh_atoms: int = 200,
        pbc_xyz: List[bool] = [True, True, True],
        sort: bool = False,
        torch_float_dtype: torch._C.dtype = torch.float32,
        has_virial: bool = False,
    ):
        self.atoms_list: List[Atoms] = ase_read(filename=filename, index=":")

        self.rcut = rcut
        self.umax_num_neigh_atoms = umax_num_neigh_atoms
        self.pbc_xyz = pbc_xyz
        self.sort = sort
        self.has_virial = has_virial

        if torch_float_dtype == torch.float32:
            self.npy_float_dtype = np.float32
            self.torch_float_dtype = torch.float32
        else:
            self.npy_float_dtype = np.float64
            self.torch_float_dtype = torch.float64

        self.max_num_atoms = self._get_max_num_atoms()
        self.type_map = self._get_type_map()

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, index: int):
        atoms = self.atoms_list[index]

        cell = atoms.get_cell().array.astype(self.npy_float_dtype)
        atom_types = np.array(
            [self.type_map.index(z) for z in atoms.get_atomic_numbers()],
            dtype=np.int32,
        )
        coords = atoms.get_positions().astype(self.npy_float_dtype)

        num_real_atoms = len(atoms)

        ilist = np.zeros(self.max_num_atoms, dtype=np.int32)
        numneigh = np.zeros(self.max_num_atoms, dtype=np.int32)

        firstneigh = np.zeros(
            (self.max_num_atoms, self.umax_num_neigh_atoms),
            dtype=np.int32,
        )

        relative_coords = np.zeros(
            (self.max_num_atoms, self.umax_num_neigh_atoms, 3),
            dtype=self.npy_float_dtype,
        )

        types = np.zeros(self.max_num_atoms, dtype=np.int32)

        forces = np.zeros(
            (self.max_num_atoms, 3),
            dtype=self.npy_float_dtype,
        )

        (
            inum,
            ilist[:num_real_atoms],
            numneigh[:num_real_atoms],
            firstneigh[:num_real_atoms],
            relative_coords[:num_real_atoms],
            types[:num_real_atoms],
            nghost,
        ) = Nblist.find_info4mlff(
            cell,
            atom_types,
            coords,
            self.rcut,
            self.umax_num_neigh_atoms,
            True,
            self.pbc_xyz,
            self.sort,
        )

        forces[:num_real_atoms] = atoms.get_forces()

        data = [
            torch.tensor(inum, dtype=torch.int32),
            torch.tensor(ilist, dtype=torch.int32),
            torch.tensor(numneigh, dtype=torch.int32),
            torch.tensor(firstneigh, dtype=torch.int32),
            torch.tensor(relative_coords, dtype=self.torch_float_dtype),
            torch.tensor(types, dtype=torch.int32),
            torch.tensor(nghost, dtype=torch.int32),
            torch.tensor(
                atoms.get_total_energy(),
                dtype=self.torch_float_dtype,
            ),
            torch.tensor(
                forces,
                dtype=self.torch_float_dtype,
            ),
        ]

        if self.has_virial:
            if "virial" in atoms.info:
                virial = atoms.info["virial"].reshape(9)
            elif "stress" in atoms.info:
                virial = (
                    -atoms.info["stress"] * atoms.get_volume()
                ).reshape(9)
            else:
                raise NotImplementedError(
                    "Extxyz file has no virial or stress."
                )

            data.append(
                torch.tensor(
                    virial,
                    dtype=self.torch_float_dtype,
                )
            )

        return data

    def _get_max_num_atoms(self):
        return max(len(atoms) for atoms in self.atoms_list)

    def _get_type_map(self):
        atomic_numbers = set()

        for atoms in self.atoms_list:
            atomic_numbers.update(atoms.get_atomic_numbers())

        return sorted(atomic_numbers)


    @staticmethod
    def get_type_map(filename: str) -> List[int]:
        atoms_list: List[Atoms] = ase_read(filename=filename, index=":")
        atom_symbol_list: List[int] = []
        for atoms in atoms_list:
            for tmp_symbol in np.unique(atoms.get_atomic_numbers()):
                if tmp_symbol not in atom_symbol_list:
                    atom_symbol_list.append(tmp_symbol)
        type_map: List[int] = sorted(atom_symbol_list)
        return type_map
    
    
    def analyse_pymatgen(self,
                         structure: Structure,
                         device: torch._C.device = torch.device("cpu")):
        cell: np.ndarray = np.array(structure.lattice.matrix).astype(self.npy_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el.Z) for el in structure.species])
        cart_coords: np.ndarray = np.array(structure.cart_coords).astype(self.npy_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=device).view(1,)
        ]


    def analyse_ase(self,
                    atoms: Atoms,
                    device: torch._C.device = torch.device("cpu")):
        cell: np.ndarray = atoms.cell.array.astype(self.npy_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el) for el in atoms.get_atomic_numbers()])
        cart_coords: np.ndarray = atoms.positions.astype(self.npy_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=device).view(1,)
        ]
```

# 4. Dataset 与 Model 的接口一致性

Dataset 的输出顺序、数据类型和张量语义必须与 Model 的输入接口保持一致。

尤其需要保证前 7 个核心输入严格对应：

```text
Dataset                      Model

inum                  →      binum_tensor
ilist                 →      bilist_tensor
numneigh              →      bnumneigh_tensor
firstneigh            →      bfirstneigh_tensor
relative_coords       →      brcs_tensor
types                 →      btypes_tensor
nghost                →      bnghost_tensor
```

不要为了某个特定模型修改 Dataset 的通用原子体系表示。

如果模型需要 Graph、descriptor 或其他特定表示，应优先在 Model 内部完成转换。

关于 Model 的详细接口规范，请参考同目录下的 `model.md`。
