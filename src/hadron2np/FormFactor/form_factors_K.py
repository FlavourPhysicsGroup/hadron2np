"""如果使用DLEFT_KDecays.py, 则需要先调用此文件，实现 K->pi 衰变的形状因子。"""
import flavio
from flavio.classes import AuxiliaryQuantity

import hadron2np
from hadron2np.classes import Parameter, Implementation


def ff_function():
    # 加一次函数变换, 为了统一 Implementation.get_values() 方法的参数前两个永远是wc, par
    par_flavio = flavio.default_parameters.get_central_all()
    ff_quantity = AuxiliaryQuantity['K->pi form factor']
    return lambda wc_obj, par_dict, q2: ff_quantity.prediction(par_flavio, None, q2=q2)


q_name = 'K->pi form factors'
q: Parameter = hadron2np.parameter_groups[q_name]
imp = Implementation(name='flavio', quantity=q_name,
                     function=ff_function(), arguments=['q2', ])
q.add_implementation(imp)
