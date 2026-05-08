"""1) 分发 wilson 系数的味指标; 2) 分发振幅模方函数; 3) 乘以相空间因子"""

from .DLEFT_B_0_2_0_1 import amp_square_2_0_1 as amp_square_B_0_2_0_1
from .DLEFT_B_P_2_1_1 import amp_square_2_1_1 as amp_square_B_P_2_1_1
from .DLEFT_B_P_3_1_1 import amp_square_3_1_1 as amp_square_B_P_3_1_1
from .DLEFT_B_V_2_1_1 import amp_square_2_1_1 as amp_square_B_V_2_1_1
from .DLEFT_B_V_3_1_1 import amp_square_3_1_1 as amp_square_B_V_3_1_1
from .DLEFT_Lb_L_3_1_1 import amp_square_3_1_1 as amp_square_Lb_L_3_1_1

from hadron2np import Phase_space_factors as ps
import numpy as np
from hmff.classes import Impl


def width_2_0_1(wcs: dict, f_B: float, m_sm: list, m_dm: list, fcnc_hadron: str, flavor_index: list):
    wc_S = wcs['L_S_dphi2'][*flavor_index]
    wc_P = wcs['L_P_dphi2'][*flavor_index]
    wc_V = wcs['L_V_dphi2'][*flavor_index]
    wc_A = wcs['L_A_dphi2'][*flavor_index]
    wcs_dict = {'S': wc_S, 'P': wc_P, 'V': wc_V, 'A': wc_A}
    m_dm_1, m_dm_2 = m_dm
    m_IS, _, m_iq, m_fq = m_sm

    match fcnc_hadron:
        case 'B->0':
            amp_square = amp_square_B_0_2_0_1
        # case 'K->0':
        #     amp_square = amp_square_K_0_2_0_1
        # case 'Upsilon->0':
        #     amp_square = amp_square_Upsilon_0_2_0_1
        # case 'Jpsi->0':
        #     amp_square = amp_square_Jpsi_0_2_0_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')
    f_phase_space = ps.two_body_phase_space_factor(m_IS, m_dm_1, m_dm_2)
    return f_phase_space * amp_square(wcs_dict, f_B, m_iq, m_fq, m_IS, m_dm_1, m_dm_2)


def width_2_1_1(
    wcs: dict, ff_imp: Impl, m_sm: list, m_dm: list, fcnc_hadron: str, flavor_index: list
):
    wc_S = wcs['L_S_dphi'][*flavor_index]
    wc_P = wcs['L_P_dphi'][*flavor_index]
    wc_V = wcs['L_V_dphi'][*flavor_index]
    wc_A = wcs['L_A_dphi'][*flavor_index]
    wcs_dict = {'S': wc_S, 'P': wc_P, 'V': wc_V, 'A': wc_A}
    m_dm_1 = m_dm[0]
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq=m_dm_1**2)

    match fcnc_hadron:
        case 'B->P':
            amp_square = amp_square_B_P_2_1_1
        case 'B->V':
            amp_square = amp_square_B_V_2_1_1
        # case 'K->P':
        #     amp_square = amp_square_K_V_2_1_1
        # case 'Lambdab->Lambda':
        #     amp_square = amp_square_Lb_L_2_1_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    f_phase_space = ps.two_body_phase_space_factor(m_IS, m_FS, m_dm_1)
    return f_phase_space * amp_square(wcs_dict, ffs, m_iq, m_fq, m_IS, m_FS, m_dm_1)

def partial_width_3_1_1(
    wcs: dict, ff_imp: Impl, m_sm: list, m_dm: list, 
    fcnc_hadron: str, flavor_index: list, qsq, dm_mode=0
):
    wc_S = wcs['L_S_dphi2'][*flavor_index]
    wc_P = wcs['L_P_dphi2'][*flavor_index]
    wc_V = wcs['L_V_dphi2'][*flavor_index]
    wc_A = wcs['L_A_dphi2'][*flavor_index]
    wcs_dict = {'S': wc_S, 'P': wc_P, 'V': wc_V, 'A': wc_A}
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq)

    match fcnc_hadron:
        case 'B->P':
            amp_square = amp_square_B_P_3_1_1
        case 'B->V':
            amp_square = amp_square_B_V_3_1_1
        # case 'K->P':
        #     amp_square = amp_square_K_P_3_1_1
        case 'Lambdab->Lambda':
            amp_square = amp_square_Lb_L_3_1_1
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')
    f_phase_space = ps.three_body_phase_space_factor(m_IS, m_dm_1, m_dm_2, m_FS, np.sqrt(qsq))
    return f_phase_space * amp_square(wcs_dict, ffs, m_iq, m_fq, m_IS, m_FS, m_dm_1, m_dm_2, qsq)