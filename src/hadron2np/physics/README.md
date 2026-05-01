## 目录: fdm/physics/
子目录同时也是 python 的模块.


### dm_scalar
#### 文件视角
- __init__.py: 暴露对外接口
- DLEFT_All.py: 依据 python 习惯, 实现接口定义
- DLEFT_XX_XX.py: 依据 Mathematica 习惯, 定义衰变宽度函数

#### python 视角
对外暴露的接口

- Gamma_IS_XX(fcnc_hadron, index, wcs, m_dm, m_sm, decay_constant)
- Gamma_IS_FSX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp)
- dGamma_IS_FSXX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp, qsq)


### dm_vector
### dm_fermion
### sm_invisible: 中微子末态
