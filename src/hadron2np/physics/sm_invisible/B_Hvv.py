#!/usr/bin/python
r"""标准模型B介子衰变到强子+中微子的半单举过程。参考1111.6402。
共有4个过程, 分别是 $B->pi(rho)vv$ 和 $B->K(K*)vv$ 的带电过程.
"""

from math import pi, sqrt

from flavio.physics import ckm
from scipy import integrate


def lambda_f(a, b, c):
    return a ** 2 + b ** 2 + c ** 2 - 2 * (a * b + a * c + b * c)


def _dGamma_dz_pi(z, par):
    """dGamma/dz(B+ -> pi+ v v)"""

    m_B = par['m_B+']
    r_tau = par['m_tau'] / m_B
    r_pi = par['m_pi+'] / m_B

    pre_factor = m_B ** 6 * r_tau ** 5 * par['f_B+'] ** 2 * par['f_pi+'] ** 2 * \
                 abs(ckm.xi('u', 'bd')(par)) ** 2 * par['GF'] ** 4 * par['tau_tau'] / (64 * pi ** 2)
    return pre_factor * ((1 - r_tau ** 2) * (1 - r_pi ** 2 / r_tau ** 2) - z)


def BR_pi(wc_obj, par_dict):
    """BR(B+ -> pi+ v v)"""
    m_B = par_dict['m_B+']
    r_tau = par_dict['m_tau'] / m_B
    r_pi = par_dict['m_pi+'] / m_B
    return integrate.quad(_dGamma_dz_pi, 0,
                          (1 - r_tau ** 2) * (1 - r_pi ** 2 / r_tau ** 2),
                          (par_dict,))[0]


def _dGammaT_dz_rho(z, par):
    """dGamma_T/dz(B+ -> pi+ v v)"""
    m_B = par['m_B+']
    r_tau = par['m_tau'] / m_B
    r_rho = par['m_rho+'] / m_B
    lambda_rhoz = lambda_f(1, z, r_rho ** 2)
    # temp = 2 * (1 - r_tau ** 2) * (1 - r_rho ** 2 / r_tau ** 2) + 2 * z
    # print('lambda_rhoz ', lambda_rhoz)
    # print('reson: ', lambda_rhoz - 2 * (1 - r_tau**2) *
    #                          (1 - r_rho**2 / r_tau**2) + 2 * z)

    pre_factor = m_B ** 6 * r_tau ** 5 * par['f_B+'] ** 2 * par['f_rho0'] ** 2 * \
                 abs(ckm.xi('u', 'bd')(par)) ** 2 * par['GF'] ** 4 * par['tau_tau'] / (64 * pi ** 2)
    return pre_factor * z * (lambda_rhoz - 2 * (1 - r_tau ** 2) *
                             (1 - r_rho ** 2 / r_tau ** 2) + 2 * z) / lambda_rhoz


def _dGammaL_dz_rho(z, par):
    """dGamma_T/dz(B+ -> pi+ v v)"""
    m_B = par['m_B+']
    r_tau = par['m_tau'] / m_B
    r_rho = par['m_rho+'] / m_B
    lambda_rhoz = lambda_f(1, z, r_rho ** 2)

    pre_factor = m_B ** 6 * r_tau ** 5 * par['f_B+'] ** 2 * par['f_rho0'] ** 2 * \
                 abs(ckm.xi('u', 'bd')(par)) ** 2 * par['GF'] ** 4 * par['tau_tau'] / (64 * pi ** 2)
    return pre_factor * (1 - z - r_rho ** 2) ** 2 * (
            (1 - r_tau ** 2) * (1 - r_rho ** 2 / r_tau ** 2) - z) / lambda_rhoz


def BR_rho(wc_obj, par_dict):
    """BR(B+ -> rho+ v v)"""
    m_B = par_dict['m_B+']
    r_tau = par_dict['m_tau'] / m_B
    r_rho = par_dict['m_rho+'] / m_B
    return integrate.quad(_dGammaL_dz_rho, 0, (1 - r_tau ** 2) * (1 - r_rho ** 2 / r_tau ** 2), (par_dict,))[0] + \
        integrate.quad(_dGammaT_dz_rho, 0, (1 - r_tau ** 2) * (1 - r_rho ** 2 / r_tau ** 2), (par_dict,))[0]


def _dGamma_dz_K(z, par):
    """dGamma/dz(B+ -> K+ v v)"""
    C_vv = 6.33
    m_B = par['m_B+']
    r_K = par['m_K+'] / m_B
    lambda_Kz = lambda_f(1, z, r_K ** 2)

    pre_factor = m_B ** 5 * abs(ckm.xi('u', 'bd')(par)) ** 2 \
                 * (par['GF'] * par['alpha_e'] * C_vv) ** 2 / (256 * pi ** 5)

    return pre_factor * sqrt(lambda_Kz) ** 3 * par['B->K form factor f+'] ** 2


def BR_K(wc_obj, par_dict):
    """BR(B+ -> K+ v v)"""
    r_K = par['m_K+'] / par['m_B+']
    return integrate.quad(_dGamma_dz_K, 0, (1 - r_K) ** 2, (par_dict, ))[0] * par_dict['tau_B+']


# if __name__ == "__main__":
#     import flavio
#
#     par = flavio.default_parameters.get_central_all()
#     par.update({
#         'f_Bs': 0.2388,
#         'f_B0': 0.1928,
#         'B->pi form factor f0': 0.258,
#         'B->pi form factor f+': 0.258,
#         'B->pi form factor fT': 0.253,
#         'B->K form factor f0': 0.331,
#         'B->K form factor f+': 0.331,
#         'B->K form factor fT': 0.358
#     })
#
#     # print(Observable['BR(B+->pivv)'].prediction.function(None, par))
#     print(Observable['BR(B+->Kvv)'].prediction.function(None, par))
#     # _dGammaT_dz_rho(0, par)
