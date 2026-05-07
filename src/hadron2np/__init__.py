from .init_config import config
from .init_parameters import parameters
from .init_parameters import parameter_groups
from .init_parameters import parameters_dict
from .DecayProcess import TwoBodyDecayProcess, ThreeBodyDecayProcess, DecayProcessBase

all_decay_processes = [
    'Lambdab->Lambda+phi', 'Lambdab->Lambda+phiphi', 'Lambdab->Lambda+X', 'Lambdab->Lambda+XX',
    'Lambdab->Lambda+chichi'
]


def todo_decay_processes():
    all_processes = []
    for fcnc_hadron in config['hadron matrix element']:
        IS_name, FS_name = fcnc_hadron.split('->')
        for dm_name in ['phi', 'X', 'chi']:
            if FS_name == '0':
                all_processes.append(f'{IS_name}->{dm_name}{dm_name}')
            else:
                all_processes.append(f'{fcnc_hadron}+{dm_name}')
                all_processes.append(f'{fcnc_hadron}+{dm_name}{dm_name}')
    return all_processes



def new_decay_process(particles: list, basis='DLEFT(L/R)') -> DecayProcessBase:
    """生成 DecayProcess 对象, 根据末态粒子数量生成对应的 TwoBodyDecayProcess 或 ThreeBodyDecayProcess"""
    match len(particles):
        case 3:
            if particles[1] in ['phi', 'X', 'chi']:  # 防止出现 [IS, X1, X2] 这种情况
                particles.insert(1, '0')
            return TwoBodyDecayProcess(particles, basis)
        case 4 if particles[1] == '0':
            return TwoBodyDecayProcess(particles, basis)
        case 4:
            return ThreeBodyDecayProcess(particles, basis)


__all__ = ['new_decay_process', 'parameters', 'parameter_groups', 'parameters_dict', 'config']
