from . import DLEFT_Lambdab_Lambda


def dGamma_IS_FSXX(fcnc_hadron, index, wcs, m_dm, m_sm, ff_imp, qsq):
    L_S = wcs['L_S_dchi2'][*index]
    L_P = wcs['L_P_dchi2'][*index]
    L_V = wcs['L_V_dchi2'][*index]
    L_A = wcs['L_A_dchi2'][*index]
    L_T = wcs['L_T_dchi2'][*index]
    L_T5 = wcs['L_T5_dchi2'][*index]

    m_dm_1, m_dm_2 = m_dm
    m_IS, m_FS, m_iq, m_fq = m_sm
    ffs = ff_imp.get_values(wc_obj=None, par_dict=None, q2=qsq)

    match fcnc_hadron:
        case 'Lambdab->Lambda':
            gamma_func = DLEFT_Lambdab_Lambda.dGamma_LambdaXX
        case _:
            raise NotImplementedError(f'Decay process not implementd yet: {fcnc_hadron}')

    return gamma_func(L_S, L_P, L_V, L_A, L_T, L_T5, m_dm_1, m_dm_2,
                      m_IS, m_FS, m_iq, m_fq, ffs, qsq)
