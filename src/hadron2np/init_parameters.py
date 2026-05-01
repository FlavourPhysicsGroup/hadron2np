"""初始化FDM包所用到的一些基本参数, 将其读入内存中."""
import yaml
from pathlib import Path
from particle import Particle
from hepunits import units as u
import hadron2np

from .classes import Parameter, ParameterGroup, Implementation
from .FormFactor import bv_pole_0406232, bv_pole_0412079, bpi_bcl_1103, Lambda_b2Lambda

# 全局变量, 最后在fdm.__init__.py文件中交给fdm来直接管理
parameters = {}
parameters_dict = {}
parameter_groups = {}


def init_parameters_from_yaml(filename):
    """从metadata文件中初始化多个参数"""
    with open(filename, 'r') as file_obj:
        infos = yaml.safe_load(file_obj)

    parameter_infos = infos['parameters']
    for k, v in parameter_infos.items():
        q = Parameter(k, container=parameters)
        for info_name, info in v.items():
            q.__dict__[info_name] = info

    parameter_group_infos = infos['parameter_groups']
    for k, v in parameter_group_infos.items():
        try:
            q = ParameterGroup(k, members=v.pop('members'), container=parameter_groups)
        except KeyError:
            raise KeyError(f"初始化参数组 {k} 时，未指定成员参数。")

        for info_name, info in v.items():
            q.__dict__[info_name] = info


def get_parameter(name):
    if name in parameters:
        par = parameters[name]
    else:
        par = Parameter(name, container=parameters)
        par.description = f'{name}'
    return par


# 读入参数描述信息
init_parameters_from_yaml((Path(__file__).parent / 'data/parameters_metadata.yaml').absolute())

# 读入参数数值
with open((Path(__file__).parent / 'data/parameters.yaml').absolute(), 'r') as file:
    infos = yaml.safe_load(file)
    for k, v in infos.items():
        q = get_parameter(k)
        q.set_value(v)


# ############ BEGIN: 为B介子衰变过程的FormFactors添加Implementation #############
def ff_function(function, process):
    # 加一次函数变换, 为了统一 Implementation.get_values() 方法的参数前两个永远是wc, par
    return lambda wc_obj, par_dict, q2: function(process, q2)


_processes = {'b_p': ['B->pi', 'B->K', 'B->eta'],
              'b_v': ['B->rho', 'Bs->K*', 'B->K*', 'B->omega', 'Bs->phi'],
              'Lambdab_Lambda': ['Lambdab->Lambda']}
for _k, _ps in _processes.items():
    for _p in _ps:
        q_name = _p + ' form factors'
        q: Parameter = parameter_groups[q_name]
        if _k == 'b_p':
            imp = Implementation(name='one-pole', quantity=q_name,
                                 function=ff_function(function=bv_pole_0406232.ff, process=_p),
                                 arguments=['q2', ])
            q.add_implementation(imp)
        elif _k == 'b_v':
            imp = Implementation(name='one-pole', quantity=q_name,
                                 function=ff_function(function=bv_pole_0412079.ff, process=_p),
                                 arguments=['q2', ])
            q.add_implementation(imp)
        elif _k == 'Lambdab_Lambda':
            imp = Implementation(name='default', quantity=q_name,
                                 function=lambda wc, par, q2: Lambda_b2Lambda.ffs_two_order(q2),
                                 arguments=['q2', ])
            q.add_implementation(imp)

imp = Implementation('bcl', 'B->pi form factors',
                     lambda wc, par, q2: bpi_bcl_1103.ff(q2), arguments=['q2', ])
parameter_groups['B->pi form factors'].add_implementation(imp)


# ############ END: 为B介子衰变过程的FormFactors添加Implementation #############

# ############ BEGIN: 为介子质量添加 Implementation #############
# 使用particle包, 从PDG中读入
for meson_name, meson_id in hadron2np.config['pdgid'].items():
    meson = Particle.from_pdgid(meson_id)

    q_name = 'm_' + meson_name
    qm: Parameter = Parameter(q_name, container=parameters)
    qm.set_value(meson.mass / u.GeV)

    try:
        q_name = 'Gamma_' + meson_name
        qt: Parameter = Parameter(q_name, container=parameters)
        qt.set_value(meson.width / u.GeV)
    except TypeError:
        # print(meson.name)
        pass
# ############ END: 为介子质量添加 Implementation #############

for par_name, par in parameters.items():
    # 这里有一些参数没有实现，也就是没有数值，所以要处理一下错误
    try:
        imp: Implementation = par.get_implementation()
        values = imp.get_values(None, None)
        if isinstance(values, dict):
            c_value = values['central']
        else:
            c_value = values
        parameters_dict[par_name] = c_value
    except KeyError:
        pass

# print(parameter_groups.keys())
# print(parameters.keys())
# exit()
