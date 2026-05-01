"""Lambda_b -> Lambda 形状因子"""

from pathlib import Path
from numpy import sqrt

# 主要参考文献: Lattice QCD 1602.01399
# z 参数化方法

name_match = {
    'f0': ['m_pole_f0', 'a0_f0', 'a1_f0', 'a2_f0'],
    'f+': ['m_pole_fp', 'a0_fplus', 'a1_fplus', 'a2_fplus'],
    'fp': ['m_pole_fp', 'a0_fperp', 'a1_fperp', 'a2_fperp'],
    'g0': ['m_pole_g0', 'a0_g0', 'a1_g0', 'a2_g0'],
    'g+': ['m_pole_gp', 'a0_gpp', 'a1_gplus', 'a2_gplus'],
    'gp': ['m_pole_gp', 'a0_gpp', 'a1_gperp', 'a2_gperp'],
    'h+': ['m_pole_fp', 'a0_hplus', 'a1_hplus', 'a2_hplus'],
    'hp': ['m_pole_fp', 'a0_hperp', 'a1_hperp', 'a2_hperp'],
    'h~+': ['m_pole_gp', 'a0_htildepp', 'a1_htildeplus', 'a2_htildeplus'],
    'h~p': ['m_pole_gp', 'a0_htildepp', 'a1_htildeperp', 'a2_htildeperp']
}


def read_data(file_path):
    """用于读取数据文件: 两列(name=value)"""
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            _tmp = line.split()
            data[_tmp[0]] = _tmp[1]
    return data


def z(qsq):
    """Lattice QCD 1602.01399: 公式 (33)"""
    # ##### 质量来自 PDG
    m_B = 5.27934
    m_K = 0.493677
    m_Lambdab = 5.6196
    m_Lambda = 1.115683

    t_plus = (m_B + m_K)**2
    t_0 = (m_Lambdab - m_Lambda)**2
    return (sqrt(t_plus - qsq) - sqrt(t_plus - t_0)) / (sqrt(t_plus - qsq) + sqrt(t_plus - t_0))


def eq_one_order(qsq, m_pole, a_0, a_1):
    """Lattice QCD 1602.01399: 公式 (38)"""
    return (a_0 + a_1 * z(qsq)) / (1 - qsq / m_pole**2)


def eq_two_order(qsq, m_pole, a_0, a_1, a_2):
    """Lattice QCD 1602.01399: 公式 (49)"""
    return (a_0 + a_1 * z(qsq) + a_2 * z(qsq)**2) / (1 - qsq / m_pole**2)


def ffs_one_order(qsq):
    """Lattice QCD 1602.01399: 公式 (38)"""
    a_data_file = Path(__file__).parent / 'Data_1602.01399/LambdabLambda_results.dat'
    m_data_file = Path(__file__).parent / 'Data_1602.01399/m_pole.dat'
    a_data = read_data(a_data_file)
    m_data = read_data(m_data_file)
    # ff_names = ['f0', 'f+', 'fp', 'g0', 'g+', 'gp', 'h+', 'hp', 'h~+', 'h~p']

    ffs = {}
    for ff_name, par_names in name_match.items():
        m_pole = float(m_data[par_names[0]])
        a_0 = float(a_data[par_names[1]])
        a_1 = float(a_data[par_names[2]])
        ffs[ff_name] = eq_one_order(qsq, m_pole, a_0, a_1)
    return ffs


def ffs_two_order(qsq):
    """Lattice QCD 1602.01399: 公式 (49)"""
    a_data_file = Path(__file__).parent / 'Data_1602.01399/LambdabLambda_HO_results.dat'
    m_data_file = Path(__file__).parent / 'Data_1602.01399/m_pole.dat'
    a_data = read_data(a_data_file)
    m_data = read_data(m_data_file)
    # ff_names = ['f0', 'f+', 'fp', 'g0', 'g+', 'gp', 'h+', 'hp', 'h~+', 'h~p']

    ffs = {}
    for ff_name, par_names in name_match.items():
        m_pole = float(m_data[par_names[0]])
        a_0 = float(a_data[par_names[1]])
        a_1 = float(a_data[par_names[2]])
        a_2 = float(a_data[par_names[3]])
        ffs[ff_name] = eq_two_order(qsq, m_pole, a_0, a_1, a_2)
    return ffs
