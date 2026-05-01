from . import DLEFT_Lambdab_Lambda


def Gamma_IS_FSX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp):
    L_V = wcs['L_V_dX'][*index]
    L_A = wcs['L_A_dX'][*index]
    L_T = wcs['L_T_dX'][*index]
    L_T5 = wcs['L_T5_dX'][*index]
    m_dm_1 = m_dm[0]
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=m_dm_1**2)

    match fcnc_hadron:
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.Gamma_LambdaX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_V, L_A, L_T, L_T5, m_dm_1, m_IS, m_FS, m_iq, m_fq, ffs)


def dGamma_IS_FSXX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp, qsq):
    L_S = wcs['L_S_dXX'][*index]
    L_P = wcs['L_P_dXX'][*index]
    L_V = wcs['L_V_dXX'][*index]
    L_A = wcs['L_A_dXX'][*index]
    L_V_dXtX = wcs['L_V_dXtX'][*index]
    L_A_dXtX = wcs['L_A_dXtX'][*index]
    L_T = wcs['L_T_dXX'][*index]
    L_T5 = wcs['L_T5_dXX'][*index]
    L_DV = wcs['L_DV_dXX'][*index]
    L_DA = wcs['L_DA_dXX'][*index]
    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=qsq)

    match fcnc_hadron:
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.dGamma_LambdaXX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, L_V_dXtX, L_A_dXtX, L_T, L_T5, L_DV, L_DA, m_dm_1, m_dm_2,
                      m_IS, m_FS, m_iq, m_fq, ffs, qsq)
