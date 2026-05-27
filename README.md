# Hadron2NP

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

重味强子衰变到暗物质过程中物理观测量的数值计算框架。

## 项目简介

Hadron2NP 用于计算重味强子（B 介子、K 介子、$\Lambda_b$ 重子等）衰变到暗物质过程中的衰变宽度与分支比等观测量。

理论框架基于暗物质低能有效场论（DLEFT），后续将扩展支持其他简化模型。

支持的暗物质类型：

| 符号 | 类型 | 说明 |
|------|------|------|
| `phi` | 标量型（Scalar） | 自旋为 0 的暗物质粒子 |
| `X` | 矢量型（Vector） | 自旋为 1 的暗物质粒子 |
| `chi` | 费米型（Fermion） | 自旋为 1/2 的暗物质粒子 |
| — | 类轴子型（ALP） | 规划中 |

支持的衰变过程类型：

- **两体衰变**：$H_1 \to H_2 + \text{DM}$，$H \to \text{DM}\,\text{DM}$
- **三体衰变**：$H_1 \to H_2 + \text{DM}\,\text{DM}$

其中 $H_1$, $H_2$ 为强子态，DM 为暗物质粒子。已实现的强子跃迁包括 $\Lambda_b \to \Lambda$、$B \to K$、$B \to \pi$、$B_s \to \text{inv}$ 等。

## 安装

### 环境要求

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/)（推荐的 Python 包管理器）

### 从源码安装
在一个空目录 `workspace/` 下，克隆 `form-factor` 和 `hadron2np` 两个仓库。
然后使用 uv 新建一个项目目录。相应的终端命令如下：

```bash
cd workspace
git clone https://gitee.com/flaour-physics-group/form-factor.git
git clone https://gitee.com/flaour-physics-group/hadron2np.git

uv init my-new-project && cd my-new-project
uv add ../form-factor
uv add ../hadron2np
```


### 从 PyPI 安装（即将支持）

```bash
uv pip install hadron2np
```

## 快速上手

### 基本使用
在 `workspace/my-new-project/` 目录下，新建一个 python 脚本，将以下内容写入后运行。
```python
import hadron2np
import numpy as np

print(hadron2np.parameters_dict)
```

### 计算衰变宽度与分支比

```python
import hadron2np

# 创建衰变过程: Lambda_b -> Lambda + phi (标量暗物质)
process = hadron2np.new_decay_process(
    ['Lambdab', 'Lambda', 'phi'],
    basis='DLEFT(S/P)'
)

# 设定 Wilson 系数与暗物质质量 (单位: GeV)
m_dm = [0.1, 0.1]

# 设定 Wilson 系数值，每个Wilson系数在味空间中对应一个四维矩阵
wc_S_dphi = np.zeros((3, 3, 2, 2), dtype=complex)
wc_S_dphi[*process.index] = 0.5 + 3j
wcs = {'L_S_dphi': wc_S_dphi}

# 计算衰变宽度
width = process.width(wcs, m_dm)

# 计算分支比
br = process.branching_ratio(wcs, m_dm)
```

### 三体衰变的微分分布

```python
import hadron2np

# 创建三体衰变过程: Lambda_b -> Lambda + chi chi (费米暗物质)
process = hadron2np.new_decay_process(
    ['Lambdab', 'Lambda', 'chi', 'chi'],
    basis='DLEFT(S/P)'
)

m_dm = [0.1, 0.1]
wc_S_dphi = np.zeros((3, 3, 2, 2), dtype=complex)
wc_S_dphi[*process.index] = 0.5 + 3j
wcs = {'L_S_dphi': wc_S_dphi}

# 计算 dGamma/dq^2
qsq = 1.0  # GeV^2
dwidth = process.dWidth_over_dqsq(wcs, m_dm, qsq)
```

### 粒子命名规则

| 粒子 | 代码名称 |
|------|----------|
| $\Lambda_b$ | `Lambdab` |
| $\Lambda$ | `Lambda` |
| $B^+$ | `B+` |
| $B_s$ | `Bs` |
| $K^+$ | `K+` |
| $\pi^0$ | `pi0` |

暗物质粒子使用 `phi`（标量）、`X`（矢量）、`chi`（费米子）。

## 依赖项

| 包名 | 用途 |
|------|------|
| [flavio](https://flav-io.github.io/) | 味物理唯象计算 |
| [wilson](https://wilson-eft.github.io/wilson/) | Wilson 系数跑动与匹配 |
| [particle](https://github.com/scikit-hep/particle) | PDG 粒子数据 |
| [hepunits](https://github.com/scikit-hep/hepunits) | 高能物理单位换算 |
| [scipy](https://scipy.org/) | 数值积分 |
| [matplotlib](https://matplotlib.org/) | 绘图 |
| [hmff](https://gitee.com/flaour-physics-group) | 强子形状因子 |

## 项目结构

```
hadron2np/
├── src/hadron2np/
│   ├── DecayProcess.py        # 衰变过程核心类
│   ├── classes.py             # 参数管理基础设施
│   ├── DLEFT/                 # Wilson 系数定义与基矢变换
│   ├── FormFactor/            # 强子形状因子实现
│   ├── physics/               # 不同暗物质类型的解析表达式
│   │   ├── dm_scalar/         # 标量暗物质
│   │   ├── dm_vector/         # 矢量暗物质
│   │   └── dm_fermion/        # 费米暗物质
│   └── data/                  # 物理参数与配置文件
├── tests/                     # 测试
└── pyproject.toml
```

## 开源协议

本项目基于 [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html) 发布。

## 引用

如果本项目对您的研究有帮助，请考虑引用相关工作。

## 链接

- 项目主页：<https://gitee.com/flaour-physics-group/hadron2np>
- 问题反馈：<https://gitee.com/flaour-physics-group/hadron2np/issues>
