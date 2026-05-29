# Hadron2NP

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

[中文文档](README_zh.md)

Numerical computation framework for physical observables in heavy-flavor hadron decays to dark matter.

## Overview

Hadron2NP computes decay widths and branching ratios for processes where heavy-flavor hadrons (B mesons, K mesons, $\Lambda_b$ baryons, etc.) decay into dark matter particles.

The theoretical framework is based on Dark Matter Low-Energy Effective Field Theory (DLEFT), with plans to extend support to other simplified models.

Supported dark matter types:

| Symbol | Type | Description |
|--------|------|-------------|
| `phi` | Scalar | Spin-0 dark matter particle |
| `X` | Vector | Spin-1 dark matter particle |
| `chi` | Fermion | Spin-1/2 dark matter particle |
| — | ALP | Planned |

Supported decay process types:

- **Two-body decays**: $H_1 \to H_2 + \text{DM}$, $H \to \text{DM}\,\text{DM}$
- **Three-body decays**: $H_1 \to H_2 + \text{DM}\,\text{DM}$

where $H_1$, $H_2$ are hadron states and DM denotes dark matter particles. Implemented hadronic transitions include $\Lambda_b \to \Lambda$, $B \to K$, $B \to \pi$, $B_s \to \text{inv}$, etc.

## Installation

### Requirements

- Python >= 3.14
- [uv](https://docs.astral.sh/uv/) (recommended Python package manager)

### From source
In an empty directory `workspace/`, clone both the `form-factor` and `hadron2np` repositories, then use uv to create a new project:

```bash
cd workspace
git clone https://github.com/FlavourPhysicsGroup/form-factor.git
git clone https://github.com/FlavourPhysicsGroup/hadron2np.git

uv init my-new-project && cd my-new-project
uv add ../form-factor
uv add ../hadron2np
```

### From PyPI (coming soon)

```bash
uv pip install hadron2np
```

## Quick Start

### Basic usage

In the `workspace/my-new-project/` directory, create a Python script with the following content and run it:

```python
import hadron2np
import numpy as np

print(hadron2np.parameters_dict)
```

### Computing decay widths and branching ratios

```python
import hadron2np

# Create decay process: Lambda_b -> Lambda + phi (scalar dark matter)
process = hadron2np.new_decay_process(
    ['Lambdab', 'Lambda', 'phi'],
    basis='DLEFT(S/P)'
)

# Set dark matter mass (unit: GeV)
m_dm = [0.1, 0.1]

# Set Wilson coefficients; each coefficient corresponds to a 4D matrix in flavor space
wc_S_dphi = np.zeros((3, 3, 2, 2), dtype=complex)
wc_S_dphi[*process.index] = 0.5 + 3j
wcs = {'L_S_dphi': wc_S_dphi}

# Compute decay width
width = process.width(wcs, m_dm)

# Compute branching ratio
br = process.branching_ratio(wcs, m_dm)
```

### Differential distributions for three-body decays

```python
import hadron2np

# Create three-body decay process: Lambda_b -> Lambda + chi chi (fermion dark matter)
process = hadron2np.new_decay_process(
    ['Lambdab', 'Lambda', 'chi', 'chi'],
    basis='DLEFT(S/P)'
)

m_dm = [0.1, 0.1]
wc_S_dphi = np.zeros((3, 3, 2, 2), dtype=complex)
wc_S_dphi[*process.index] = 0.5 + 3j
wcs = {'L_S_dphi': wc_S_dphi}

# Compute dGamma/dq^2
qsq = 1.0  # GeV^2
dwidth = process.dWidth_over_dqsq(wcs, m_dm, qsq)
```

### Particle naming conventions

| Particle | Code name |
|----------|-----------|
| $\Lambda_b$ | `Lambdab` |
| $\Lambda$ | `Lambda` |
| $B^+$ | `B+` |
| $B_s$ | `Bs` |
| $K^+$ | `K+` |
| $\pi^0$ | `pi0` |

Dark matter particles use `phi` (scalar), `X` (vector), `chi` (fermion).

## Dependencies

| Package | Purpose |
|---------|---------|
| [flavio](https://flav-io.github.io/) | Flavor physics phenomenology |
| [wilson](https://wilson-eft.github.io/wilson/) | Wilson coefficient running and matching |
| [particle](https://github.com/scikit-hep/particle) | PDG particle data |
| [hepunits](https://github.com/scikit-hep/hepunits) | HEP unit conversions |
| [scipy](https://scipy.org/) | Numerical integration |
| [matplotlib](https://matplotlib.org/) | Plotting |
| [hmff](https://gitee.com/flaour-physics-group) | Hadronic form factors |

## Project structure

```
hadron2np/
├── src/hadron2np/
│   ├── DecayProcess.py        # Core decay process classes
│   ├── classes.py             # Parameter management infrastructure
│   ├── DLEFT/                 # Wilson coefficient definitions and basis transformations
│   ├── FormFactor/            # Hadronic form factor implementations
│   ├── physics/               # Analytic expressions for different DM types
│   │   ├── dm_scalar/         # Scalar dark matter
│   │   ├── dm_vector/         # Vector dark matter
│   │   └── dm_fermion/        # Fermion dark matter
│   └── data/                  # Physics parameters and config files
├── tests/                     # Tests
└── pyproject.toml
```

## License

This project is released under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

## Citation

If this project is helpful for your research, please consider citing the relevant publications.

## Links

- Project page: <https://github.com/FlavourPhysicsGroup/hadron2np>
- Issue tracker: <https://github.com/FlavourPhysicsGroup/hadron2np/issues>
