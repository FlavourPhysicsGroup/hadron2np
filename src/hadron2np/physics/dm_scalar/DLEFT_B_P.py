import numpy as np
from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f
from hadron2np.methods import phase_space_3_body


def Gamma_PX(L_S, L_P, L_V, L_A, mdm1, mIS, mFS, miq, mfq, ffs):
    lam_sqrt = sqrt(kallen_f(mIS**2, mFS**2, mdm1**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    FFf0 = ffs['f0']

    J_11 = (
        abs(L_S) ** 2
        * pow(mfq - miq, -2)
        * pow(FFf0, 2)
        * pow(pow(mFS, 2) - pow(mIS, 2), 2)
    )
    J_22 = abs(L_V) ** 2 * pow(FFf0, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)
    J_12 = (
        2
        * (L_S * L_V.conjugate()).imag
        * (-(pow(FFf0, 2) * pow(mfq - miq, -1) * pow(pow(mFS, 2) - pow(mIS, 2), 2)))
    )

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq


def dGamma_PXX(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, mFS, miq, mfq, ffs, qsq):
    f_phase_space = 1 / (256 * pi**3 * mIS**3)

    FFf0 = ffs['f0']
    FFfp = ffs['f+']

    ps3 = phase_space_3_body(mIS, mFS, mdm1, mdm2, qsq)
    m23sq = ps3.m23sq()
    m23sqsq = ps3.m23sq_sq()
    m23sqcube = ps3.m23sq_cube()

    J_11 = (
        abs(L_S) ** 2
        * m23sq
        * pow(FFf0, 2)
        * pow(mfq - miq, -2)
        * pow(pow(mFS, 2) - pow(mIS, 2), 2)
    )
    J_22 = abs(L_V) ** 2 * (
        (4 * m23sqcube * pow(FFfp, 2)) / 3.0
        + 2
        * FFfp
        * m23sqsq
        * pow(qsq, -1)
        * (
            FFf0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2))
            - FFfp
            * (
                pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2))
                + pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2))
                + qsq * (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                - pow(qsq, 2)
            )
        )
        + m23sq
        * pow(qsq, -2)
        * pow(
            FFf0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2))
            - FFfp
            * (
                pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2))
                + pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2))
                + qsq * (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                - pow(qsq, 2)
            ),
            2,
        )
    )
    J_12 = (
        2
        * (L_S * L_V.conjugate()).real
        * (
            FFf0
            * pow(mfq - miq, -1)
            * (pow(mFS, 2) - pow(mIS, 2))
            * pow(qsq, -1)
            * (
                -(FFf0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2)))
                + FFfp
                * (
                    pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2))
                    + pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2))
                    + qsq * (-2 * m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                    - pow(qsq, 2)
                )
            )
        )
    )

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_S = wcs['S']
    wc_V = wcs['V']
    FFf0 = ffs['f0']

    ampSq11 = (
        wc_S
        * wc_S.conjugate()
        * np.power(m_fq - m_iq, -2)
        * np.power(FFf0, 2)
        * np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
    )
    ampSq13 = (
        1
        * 1j
        * wc_S.conjugate()
        * wc_V
        * np.power(-m_fq + m_iq, -1)
        * np.power(FFf0, 2)
        * np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
    )
    ampSq31 = (
        1
        * 1j
        * wc_S
        * wc_V.conjugate()
        * np.power(m_fq - m_iq, -1)
        * np.power(FFf0, 2)
        * np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
    )
    ampSq33 = (
        wc_V
        * wc_V.conjugate()
        * np.power(FFf0, 2)
        * np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
    )

    return ampSq11 + ampSq13 + ampSq31 + ampSq33


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    wc_S = wcs['S']
    wc_V = wcs['V']
    FFf0 = ffs['f0']
    FFfp = ffs['f+']

    ampSq11 = (
        4
        * wc_S
        * wc_S.conjugate()
        * np.power(m_fq - m_iq, -2)
        * np.power(FFf0, 2)
        * np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
    )
    ampSq13 = (
        -4
        * wc_S.conjugate()
        * wc_V
        * FFf0
        * np.power(m_fq - m_iq, -1)
        * (np.power(m_FS, 2) - np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq31 = (
        -4
        * wc_S
        * wc_V.conjugate()
        * FFf0
        * np.power(m_fq - m_iq, -1)
        * (np.power(m_FS, 2) - np.power(m_IS, 2))
        * np.power(qsq, -1)
        * (
            FFf0
            * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2),
                0.5,
            )
        )
    )
    ampSq33 = (
        4
        * wc_V
        * wc_V.conjugate()
        * np.power(qsq, -2)
        * np.power(
            FFf0
            * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (np.power(m_FS, 2) - np.power(m_IS, 2))
            + FFfp
            * np.cos(theta1)
            * np.power(
                np.power(m_dm1, 4)
                - 2 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
                + np.power(qsq - np.power(m_dm2, 2), 2),
                0.5,
            )
            * np.power(
                -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2),
                0.5,
            ),
            2,
        )
    )

    return ampSq11 + ampSq13 + ampSq31 + ampSq33
