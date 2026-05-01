## API接口
1. class: Implementation
   1. init(name: str, quantity: str, function, arguments=None, *container)
   2. 属性: name, description, quantity, cites, arguments
   3. 属性: function = f(wc_obj, par_dict, *args, **kwargs)
   4. get_values(wc_obj, par_dict, *args, **kwargs) -> 返回dict,如果是值:{central: 0, upper:0, lower:0}, 如果是group{key: value}
    
2. class: Quantity
   1. init(name, *tex, *description, *container)
   2. 属性: name, tex, description, implementations
   3. add_implementation(Implementation)
   4. get_implementation(*name)
   5. show_implementations()

3. class: ParameterGroup(Quantity)
   1. init(name, **kwargs)
   2. 属性: name, tex, description, members, implementations
   3. add_implementation(Implementation)
   4. get_implementation(*name)
   5. show_implementations()
   <!-- 6. get_central_all(wc_obj, par_dict, imp_name=None, *kwargs) -> dict -->
   
4. TODO: class: Parameter(Quantity)
   1. init(name, value, *container)

   
5. class: Measurement(Quantity)
   1. init(name, observable, *container)

6. dict: parameters
7. dict: parameter_groups


## Implementation(name, quantity, function, **kwargs)
- quantity
- function: must be ff(wc_obj, par_dict, *args, **kwargs)
- attributions
- get_central()
- get()


## Quantity(name, **kwargs)
- attributions
- implementations
- add_implementation()
- get_implementation()
- show_implementations()

### Parameter(name, value: str, **kwargs)
为了简化抽象, 学习FeynRules的方法, 把参数加一个类型的属性, 分Internal 和 External
基本参数: 定义值是一个常数(加误差), 不依赖于其它参数.
通常只有一种实现, 即直接给定数值.
- implementation
- get_central()
- parse_value()

### CompositeParameter(name, )
间接参数: 由基本参数组合而成, 可以有多种实现方式
- add_implementation(Implementation)
- get_central()
  暂时只有中心值吧

### Observable
可观测量: 有点像是间接参数, 定义为理论的计算解析表达式. 它还要联系上实验观测.


## DecayProcess()
### __init__(初末态粒子, 算符基, 其它参数)

### 属性:
- IS: str, FS: str, dms: list[str], dm_name: str, particles: list[str] : 初末态粒子
- fcnc_hadron: str : 强子矩阵元的初末态粒子, 如 "B+->K+"
- fcnc_quark: str : 夸克流的初末态夸克, 如 "b->s"
- fcnc_class: str : 强子矩阵元的分类, 如 "B->P"

### 方法:
- dWidth_over_dqsq(wcs, m_dm, qsq)->float
- dBr_over_dqsq(wcs, m_dm, qsq)->float
- width(wcs, m_dm)->float
- branching_ratio(wcs, m_dm)->float
- 其它角观测量

### 内部属性:
- analytic_dir: module : 依据 self.dm_name['chi', 'phi', 'X'], 确定解析表达式所在的目录模块
- analytic_func: function : 依据 self.fcnc_hadron, 确定具体的解析表达式函数, 如 analytic_dir.Gamma_B_PXX()
- index: list[int] : 依据 self.fcnc_quark, 确定Wilson系数的味指标
- formfactor: fdm.Implemention : 依据 self.fcnc_hadron, 确定形状因子函数
- m_sm: list[float] : 初末态介子以及流夸克质量
- scale: float : 衰变过程的典型能标, 用于流夸克质量跑动. TODO: 加入 Wilson 系数的跑动.

### 内部方法:
- get_flavour_index()->list[int] : 依据 self.fcnc_quark, 得到 Wilson 系数的味指标. 如 sbab: (1, 2, 0, 1)
- get_decay_energy_scale()->float
- get_nf(scale)->int
- get_sm_masses()->list[float]