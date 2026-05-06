

根据提供的代码地图，我可以了解这是一个关于强子衰变与暗物质有效场论（DM_EFT）的计算物理项目。让我来生成README文档。

# hadron2np

[![Python版本](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

**hadron2np** 是一个用于计算强子衰变过程与暗物质有效场论（Dark Matter Effective Field Theory, DM_EFT）的Python计算物理库。该库提供了强子衰变宽度、分支比以及相关可观测量的计算框架，支持自旋为 0、1/2 和 1 的暗物质候选者。

### 物理背景

本项目涵盖以下暗物质类型：

- **Spin-0 DM（标量暗物质）**：赝标量暗物质候选者（如轴子）
- **Spin-1/2 DM（费米子暗物质）**：旋量暗物质
- **Spin-1 DM（矢量暗物质）**：矢量暗物质

### 支持的衰变过程

#### K介子衰变
- $K_L \to \pi^0 \phi$，$K_L \to \pi^+ \pi^- \phi$
- $K^+ \to \pi^+ \phi$
- $K_L \to \bar{\phi}\phi$，$K_L \to \pi^0\bar{\phi}\phi$

#### B介子衰变
- $B \to H^{(*)}\phi$（$H \in \{K, \pi\}$），共8个过程
- $B \to \bar{\psi}\psi$，$B \to H^{(*)}\bar{\psi}\psi$

## 项目结构

```
hadron2np/
├── src/hadron2np/
│   ├── DecayProcess.py      # 衰变过程基类定义
│   ├── Phase_space_factors.py  # 相空间因子计算
│   ├── classes.py           # 核心类定义
│   ├── methods.py            # 辅助方法
│   ├── FormFactor/           # 形状因子实现
│   ├── physics/
│   │   ├── dm_scalar/        # 标量暗物质物理
│   │   ├── dm_fermion/      # 费米子暗物质物理
│   │   ├── dm_vector/       # 矢量暗物质物理
│   │   └── sm_invisible/    # 不可见SM末态
│   └── data/                # 参数与元数据
└── tests/                   # 测试文件
```

## 安装

```bash
pip install -e .
```

或使用 uv：

```bash
uv pip install -e .
```

## 快速开始

### 创建衰变过程

```python
import hadron2np

# 创建两体衰变过程
process = hadron2np.new_decay_process(['B', 'K', 'phi'], basis='DLEFT(L/R)')

# 设置Wilson系数
wcs = {'phi_B_s': 1.0, 'phi_B_p': 0.1}

# 设置暗物质质量
m_dm = [1.0, 1.0]

# 计算衰变宽度
width = process.width(wcs, m_dm)

# 计算分支比
br = process.branching_ratio(wcs, m_dm)
```

### 三体衰变

```python
from hadron2np import DecayProcess

# 创建三体衰变过程
process = DecayProcess.ThreeBodyDecayProcess(
    ['B', 'K', 'phi', 'phi'],
    basis='DLEFT(L/R)'
)
```

## 核心类

### DecayProcessBase

衰变过程基类，提供以下功能：

| 方法 | 描述 |
|------|------|
| `width(wcs, m_dm)` | 计算衰变宽度 |
| `branching_ratio(wcs, m_dm)` | 计算分支比 |
| `get_flavour_index()` | 获取味指数 |
| `set_parameter(name, value)` | 设置参数值 |

### 两体与三体衰变

- **TwoBodyDecayProcess**：两体衰变过程
- **ThreeBodyDecayProcess**：三体衰变过程，支持微分宽度计算 `dWidth_over_dqsq`

## 物理实现

### 形状因子

库中包含多种形状因子实现：

- `bpi_bcl_1103.py`：$B \to \pi$ 形状因子
- `bv_pole_0406232.py`：$B \to K$ 形状因子
- `Lambda_b2Lambda.py`：$\Lambda_b \to \Lambda$ 形状因子

### 相空间因子

```python
from hadron2np import Phase_space_factors as psf

# Kallen函数
kallen_factor = psf.kallen(asq, bsq, csq)

# 两体相空间因子
phase_space_2 = psf.two_body_phase_space_factor(m_IS, m_1, m_2)

# 三体相空间因子
phase_space_3 = psf.three_body_phase_space_factor(
    m_IS, m_1, m_2, m_3, m_12, theta1
)
```

## 配置与参数

参数通过 YAML 文件配置：

- `data/parameters.yaml`：物理参数
- `data/observables_metadata.yaml`：可观测量元数据
- `data/pdg_particle.yaml`：粒子数据

## API 参考

更多 API 细节请参见 `src/hadron2np/README.md`。

## 测试

运行测试：

```bash
pytest tests/
```

或使用 Jupyter Notebook：

```bash
jupyter notebook tests/test_hadron2np.ipynb
```

## 依赖

- numpy
- scipy
- yaml

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件