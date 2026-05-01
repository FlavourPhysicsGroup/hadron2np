import numpy as np
from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f


def Gamma(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, miq, mfq, fB) -> float:
    lam_sqrt = sqrt(kallen_f(mIS**2, mdm1**2, mdm2**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    J_11 = abs(L_P) ** 2 * (pow(fB, 2) * pow(mfq + miq, -2) * pow(mIS, 4))
    J_22 = abs(L_A) ** 2 * (pow(fB, 2) * pow(pow(mdm1, 2) - pow(mdm2, 2), 2))
    J_12 = (
        2
        * (L_P * L_A.conjugate()).imag
        * (
            pow(fB, 2)
            * (pow(mdm1, 2) - pow(mdm2, 2))
            * pow(mfq + miq, -1)
            * pow(mIS, 2)
        )
    )

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq


def amp_square_2_0_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_dm1, m_dm2) -> float:
    wc_P = wcs["P"]
    wc_A = wcs["A"]
    f_B = wcs["fB"]

    ampSq22 = (
        4
        * wc_P
        * wc_P.conjugate()
        * np.power(f_B, 2)
        * np.power(m_fq + m_iq, -2)
        * np.power(m_IS, 4)
    )
    ampSq24 = (
        4
        * 1j
        * wc_A
        * wc_P.conjugate()
        * np.power(f_B, 2)
        * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
        * np.power(m_fq + m_iq, -1)
        * np.power(m_IS, 2)
    )
    ampSq42 = (
        -4
        * 1j
        * wc_A.conjugate()
        * wc_P
        * np.power(f_B, 2)
        * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
        * np.power(m_fq + m_iq, -1)
        * np.power(m_IS, 2)
    )
    ampSq44 = (
        4
        * wc_A
        * wc_A.conjugate()
        * np.power(f_B, 2)
        * np.power(np.power(m_dm1, 2) - np.power(m_dm2, 2), 2)
    )

    return ampSq22 + ampSq24 + ampSq42 + ampSq44
