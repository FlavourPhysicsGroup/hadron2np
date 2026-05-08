"""1) 分发 wilson 系数的味指标; 2) 分发振幅模方函数; 3) 乘以相空间因子"""

from .DLEFT_B_0_2_0_1_1 import amp_square_2_0_1 as amp_square_B_0_2_0_1_1
from .DLEFT_B_0_2_0_1_2 import amp_square_2_0_1 as amp_square_B_0_2_0_1_2
from .DLEFT_B_0_2_0_1_3 import amp_square_2_0_1 as amp_square_B_0_2_0_1_3

from .DLEFT_B_P_3_1_1_1 import amp_square_3_1_1 as amp_square_B_P_3_1_1_1
from .DLEFT_B_P_3_1_1_2 import amp_square_3_1_1 as amp_square_B_P_3_1_1_2
from .DLEFT_B_P_3_1_1_3 import amp_square_3_1_1 as amp_square_B_P_3_1_1_3

from .DLEFT_B_V_3_1_1_1 import amp_square_3_1_1 as amp_square_B_V_3_1_1_1
from .DLEFT_B_V_3_1_1_2 import amp_square_3_1_1 as amp_square_B_V_3_1_1_2
from .DLEFT_B_V_3_1_1_3 import amp_square_3_1_1 as amp_square_B_V_3_1_1_3

from .DLEFT_Lb_L_3_1_1_1 import amp_square_3_1_1 as amp_square_Lb_L_3_1_1_1
from .DLEFT_Lb_L_3_1_1_2 import amp_square_3_1_1 as amp_square_Lb_L_3_1_1_2
from .DLEFT_Lb_L_3_1_1_3 import amp_square_3_1_1 as amp_square_Lb_L_3_1_1_3

from hadron2np import Phase_space_factors as ps
import numpy as np
from hmff.classes import Impl

AMP_SQUARE_MAP = {
    ('B->0', 1): amp_square_B_0_2_0_1_1,
    ('B->0', 2): amp_square_B_0_2_0_1_2,
    ('B->0', 3): amp_square_B_0_2_0_1_3,
    ('B->P', 1): amp_square_B_P_3_1_1_1,
    ('B->P', 2): amp_square_B_P_3_1_1_2,
    ('B->P', 3): amp_square_B_P_3_1_1_3,
    ('B->V', 1): amp_square_B_V_3_1_1_1,
    ('B->V', 2): amp_square_B_V_3_1_1_2,
    ('B->V', 3): amp_square_B_V_3_1_1_3,
    ('Lambdab->Lambda', 1): amp_square_Lb_L_3_1_1_1,
    ('Lambdab->Lambda', 2): amp_square_Lb_L_3_1_1_2,
    ('Lambdab->Lambda', 3): amp_square_Lb_L_3_1_1_3,
}


def width_2_0_1(wcs: dict, f_B: float, m_sm: list, m_dm: list, fcnc_hadron: str, flavor_index: list, dm_mode=1):
    wc_S = wcs['L_S_dchi2'][*flavor_index]
    wc_P = wcs['L_P_dchi2'][*flavor_index]
    wc_V = wcs['L_V_dchi2'][*flavor_index]
    wc_A = wcs['L_A_dchi2'][*flavor_index]
    wc_T = wcs['L_T_dchi2'][*flavor_index]
    wc_T5 = wcs['L_T5_dchi2'][*flavor_index]
    wcs_dict = {'S': wc_S, 'P': wc_P, 'V': wc_V, 'A': wc_A, 'T': wc_T, 'T5': wc_T5}
    m_dm_1, m_dm_2 = m_dm
    m_IS, _, m_iq, m_fq = m_sm

    amp_square = AMP_SQUARE_MAP.get((fcnc_hadron, dm_mode))
    f_phase_space = ps.two_body_phase_space_factor(m_IS, m_dm_1, m_dm_2)
    return f_phase_space * amp_square(wcs_dict, f_B, m_iq, m_fq, m_IS, m_dm_1, m_dm_2)


def partial_width_3_1_1(
    wcs: dict, ff_imp: Impl, m_sm: list, m_dm: list, 
    fcnc_hadron: str, flavor_index: list, qsq, dm_mode=1
):
    wc_S = wcs['L_S_dchi2'][*flavor_index]
    wc_P = wcs['L_P_dchi2'][*flavor_index]
    wc_V = wcs['L_V_dchi2'][*flavor_index]
    wc_A = wcs['L_A_dchi2'][*flavor_index]
    wc_T = wcs['L_T_dchi2'][*flavor_index]
    wc_T5 = wcs['L_T5_dchi2'][*flavor_index]
    wcs_dict = {'S': wc_S, 'P': wc_P, 'V': wc_V, 'A': wc_A, 'T': wc_T, 'T5': wc_T5}
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_central_values(qsq)

    amp_square = AMP_SQUARE_MAP.get((fcnc_hadron, dm_mode))
    f_phase_space = ps.three_body_phase_space_factor(m_IS, m_dm_1, m_dm_2, m_FS, np.sqrt(qsq))
    return f_phase_space * amp_square(wcs_dict, ffs, m_iq, m_fq, m_IS, m_FS, m_dm_1, m_dm_2, qsq)
