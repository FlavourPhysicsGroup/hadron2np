from . import DLEFT_B_0, DLEFT_B_P, DLEFT_B_V
from . import DLEFT_K_0, DLEFT_K_P
from . import DLEFT_Quarkonium_0
from . import DLEFT_Lambdab_Lambda


def Gamma_IS_XX(fcnc_hadron, index, wcs, m_dm, m_sm, decay_constant):
    L_S = wcs['L_S_dphi2'][*index]
    L_P = wcs['L_P_dphi2'][*index]
    L_V = wcs['L_V_dphi2'][*index]
    L_A = wcs['L_A_dphi2'][*index]
    m_dm_1, m_dm_2 = m_dm
    m_IS, _, m_iq, m_fq = m_sm

    match fcnc_hadron:
        case 'B->0':
            gamma_func = DLEFT_B_0.Gamma
        case 'K->0':
            gamma_func = DLEFT_K_0.Gamma
        case 'Upsilon->0' | 'Jpsi->0':
            gamma_func = DLEFT_Quarkonium_0.Gamma
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_dm_2, m_IS, m_iq, m_fq, decay_constant)


def Gamma_IS_FSX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp):
    L_S = wcs['L_S_dphi'][*index]
    L_P = wcs['L_P_dphi'][*index]
    L_V = wcs['L_V_dphi'][*index]
    L_A = wcs['L_A_dphi'][*index]
    m_dm_1 = m_dm[0]
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=m_dm_1**2)

    match fcnc_hadron:
        case 'B->P':
            gamma_func = DLEFT_B_P.Gamma_PX
        case 'B->V':
            gamma_func = DLEFT_B_V.Gamma_VX
        case 'K->P':
            gamma_func = DLEFT_K_P.Gamma_PX
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.Gamma_LambdaX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_IS, m_FS, m_iq, m_fq, ffs)


def dGamma_IS_FSXX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp, qsq):
    L_S = wcs['L_S_dphi2'][*index]
    L_P = wcs['L_P_dphi2'][*index]
    L_V = wcs['L_V_dphi2'][*index]
    L_A = wcs['L_A_dphi2'][*index]
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=qsq)

    match fcnc_hadron:
        case 'B->P':
            gamma_func = DLEFT_B_P.dGamma_PXX
        case 'B->V':
            gamma_func = DLEFT_B_V.dGamma_VXX
        case 'K->P':
            gamma_func = DLEFT_K_P.dGamma_PXX
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.dGamma_LambdaXX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, m_dm_1, m_dm_2, m_IS, m_FS, m_iq, m_fq, ffs, qsq)
