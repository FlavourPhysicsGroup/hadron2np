import numpy as np
from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f
from hadron2np.methods import phase_space_3_body


def Gamma_VX(L_S, L_P, L_V, L_A, mdm1, mIS, mFS, miq, mfq, ffs):
    '''单个标量粒子时的总宽度: Gamma(B->VX)'''
    lam_sqrt = sqrt(kallen_f(mIS**2, mFS**2, mdm1**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    FFA0 = ffs['A0']
    FFA1 = ffs['A1']
    FFA2 = ffs['A2']
    FFA3 = ffs['A3']

    func1 = (
        pow(mdm1, 4)
        - 2 * pow(mdm1, 2) * (pow(mFS, 2) + pow(mIS, 2))
        + pow(pow(mFS, 2) - pow(mIS, 2), 2)
    )

    J_11 = abs(L_P) ** 2 * 3 * func1 * pow(mfq + miq, -2) * pow(FFA0, 2)
    J_22 = abs(L_A) ** 2 * (
        (
            3
            * func1
            * pow(mFS, -2)
            * pow(
                2 * FFA0 * mFS
                + FFA2 * mFS
                - 2 * FFA3 * mFS
                - FFA2 * mIS
                + FFA1 * (mFS + mIS),
                2,
            )
        )
        / 4.0
    )
    J_12 = (
        2
        * (L_P * L_A.conjugate()).real
        * (
            -3
            * FFA0
            * (
                2 * FFA0 * mFS
                + FFA2 * mFS
                - 2 * FFA3 * mFS
                - FFA2 * mIS
                + FFA1 * (mFS + mIS)
            )
            * pow(mFS, -1)
            * pow(mfq + miq, -1)
            * (
                pow(mdm1, 4)
                - 2 * pow(mdm1, 2) * (pow(mFS, 2) + pow(mIS, 2))
                + pow(pow(mFS, 2) - pow(mIS, 2), 2)
            )
        )
        / 2.0
    )

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq


def dGamma_VXX(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, mFS, miq, mfq, ffs, qsq):
    f_phase_space = 1 / (256 * pi**3 * mIS**3)

    FFV = ffs['V']
    FFA0 = ffs['A0']
    FFA1 = ffs['A1']
    FFA2 = ffs['A2']
    FFA3 = ffs['A3']

    ps3 = phase_space_3_body(mIS, mFS, mdm1, mdm2, qsq)
    m23sq = ps3.m23sq()
    m23sqsq = ps3.m23sq_sq()
    # m23sqcube = ps3.m23sq_cube()

    J_11 = abs(L_P) ** 2 * (
        3
        * m23sq
        * pow(FFA0, 2)
        * pow(mfq + miq, -2)
        * (
            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
            + pow(qsq, 2)
            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
        )
    )
    J_22 = abs(L_V) ** 2 * (
        -2
        * m23sq
        * pow(FFV, 2)
        * pow(mFS + mIS, -2)
        * (
            6 * pow(mdm1, 4) * pow(mFS, 2)
            + 3
            * pow(mdm2, 2)
            * (
                m23sq * (pow(mFS, 2) - pow(mIS, 2))
                + 2 * pow(mIS, 2) * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))
            )
            - 3
            * pow(mdm1, 2)
            * (
                qsq * (m23sq - 2 * pow(mdm2, 2) + 2 * pow(mFS, 2))
                + (m23sq - 2 * pow(mFS, 2)) * (pow(mFS, 2) - pow(mIS, 2))
                + 2 * pow(mdm2, 2) * (pow(mFS, 2) + pow(mIS, 2))
            )
            - qsq
            * (
                -2 * m23sqsq
                - 6 * pow(mFS, 2) * pow(mIS, 2)
                + 3 * m23sq * (pow(mFS, 2) + pow(mIS, 2))
                + 3 * pow(mdm2, 2) * (m23sq + 2 * pow(mIS, 2))
            )
            + 3 * m23sq * pow(qsq, 2)
        )
    )
    J_33 = abs(L_A) ** 2 * (
        (
            m23sq
            * pow(mFS, -2)
            * pow(mFS + mIS, -2)
            * pow(qsq, -2)
            * (
                4
                * m23sqsq
                * pow(qsq, 2)
                * (
                    2
                    * FFA1
                    * FFA2
                    * (qsq + pow(mFS, 2) - pow(mIS, 2))
                    * pow(mFS + mIS, 2)
                    + pow(FFA1, 2) * pow(mFS + mIS, 4)
                    + pow(FFA2, 2)
                    * (
                        -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                        + pow(qsq, 2)
                        + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                    )
                )
                - 6
                * m23sq
                * qsq
                * (
                    -(
                        FFA1
                        * (
                            2
                            * FFA0
                            * mFS
                            * (pow(mdm1, 2) - pow(mdm2, 2))
                            * (qsq + pow(mFS, 2) - pow(mIS, 2))
                            - 2
                            * FFA3
                            * mFS
                            * (pow(mdm1, 2) - pow(mdm2, 2))
                            * (qsq + pow(mFS, 2) - pow(mIS, 2))
                            - FFA1
                            * (mFS + mIS)
                            * qsq
                            * (-qsq + 2 * pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                        )
                        * pow(mFS + mIS, 3)
                    )
                    + qsq
                    * pow(FFA2, 2)
                    * (-qsq + pow(mdm1, 2) + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                    * (
                        -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                        + pow(qsq, 2)
                        + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                    )
                    - FFA2
                    * (mFS + mIS)
                    * (
                        FFA1
                        * (mFS + mIS)
                        * qsq
                        * (
                            -3 * qsq * pow(mdm2, 2)
                            + pow(mdm2, 2) * pow(mFS, 2)
                            - 2 * pow(mFS, 4)
                            - 4 * qsq * pow(mIS, 2)
                            + 3 * pow(mdm2, 2) * pow(mIS, 2)
                            + pow(mdm1, 2) * (-qsq - 5 * pow(mFS, 2) + pow(mIS, 2))
                            + 2 * pow(mIS, 4)
                            + 2 * pow(qsq, 2)
                        )
                        + 2
                        * FFA0
                        * mFS
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                        - 2
                        * FFA3
                        * mFS
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                    )
                )
                + 3
                * (
                    -2
                    * FFA2
                    * (mFS + mIS)
                    * qsq
                    * (-qsq + pow(mdm1, 2) + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))
                    * (
                        FFA1
                        * (mFS + mIS)
                        * qsq
                        * (
                            -4 * pow(mdm1, 2) * pow(mFS, 2)
                            - 2 * qsq * (pow(mdm2, 2) + pow(mIS, 2))
                            + (2 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))
                            * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                        )
                        + 2
                        * FFA0
                        * mFS
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                        - 2
                        * FFA3
                        * mFS
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                    )
                    + pow(mFS + mIS, 2)
                    * (
                        -4
                        * FFA1
                        * FFA3
                        * mFS
                        * (mFS + mIS)
                        * qsq
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -4 * pow(mdm1, 2) * pow(mFS, 2)
                            - 2 * qsq * (pow(mdm2, 2) + pow(mIS, 2))
                            + (2 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))
                            * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                        )
                        + pow(FFA1, 2)
                        * pow(mFS + mIS, 2)
                        * pow(qsq, 2)
                        * (
                            4 * pow(mdm2, 4)
                            - 8 * pow(mdm1, 2) * pow(mFS, 2)
                            + pow(mFS, 4)
                            - 4 * pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2))
                            + 2 * pow(mFS, 2) * pow(mIS, 2)
                            - 2 * qsq * (2 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))
                            + pow(mIS, 4)
                            + pow(qsq, 2)
                        )
                        + 4
                        * pow(FFA0, 2)
                        * pow(mFS, 2)
                        * pow(pow(mdm1, 2) - pow(mdm2, 2), 2)
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                        + 4
                        * pow(FFA3, 2)
                        * pow(mFS, 2)
                        * pow(pow(mdm1, 2) - pow(mdm2, 2), 2)
                        * (
                            -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                            + pow(qsq, 2)
                            + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                        )
                        - 4
                        * FFA0
                        * mFS
                        * (pow(mdm1, 2) - pow(mdm2, 2))
                        * (
                            -(
                                FFA1
                                * (mFS + mIS)
                                * qsq
                                * (
                                    -4 * pow(mdm1, 2) * pow(mFS, 2)
                                    - 2 * qsq * (pow(mdm2, 2) + pow(mIS, 2))
                                    + (2 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))
                                    * (pow(mFS, 2) + pow(mIS, 2))
                                    + pow(qsq, 2)
                                )
                            )
                            + 2
                            * FFA3
                            * mFS
                            * (pow(mdm1, 2) - pow(mdm2, 2))
                            * (
                                -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                                + pow(qsq, 2)
                                + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                            )
                        )
                    )
                    + pow(FFA2, 2)
                    * pow(qsq, 2)
                    * (
                        -2 * qsq * (pow(mFS, 2) + pow(mIS, 2))
                        + pow(qsq, 2)
                        + pow(pow(mFS, 2) - pow(mIS, 2), 2)
                    )
                    * pow(
                        -qsq + pow(mdm1, 2) + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2),
                        2,
                    )
                )
            )
        )
        / 4.0
    )
    J_13 = (
        2
        * (L_P * L_A.conjugate()).imag
        * (
            (
                3
                * FFA0
                * pow(mFS, -1)
                * pow(mfq + miq, -1)
                * pow(mFS + mIS, -1)
                * pow(qsq, -1)
                * (
                    -(
                        FFA2
                        * qsq
                        * (
                            -2 * m23sq
                            - qsq
                            + pow(mdm1, 2)
                            + pow(mdm2, 2)
                            + pow(mFS, 2)
                            + pow(mIS, 2)
                        )
                        * (
                            -2 * (qsq + pow(mFS, 2)) * pow(mIS, 2)
                            + pow(mIS, 4)
                            + pow(qsq - pow(mFS, 2), 2)
                        )
                    )
                    + (mFS + mIS)
                    * (
                        FFA1
                        * (mFS + mIS)
                        * qsq
                        * (
                            -(
                                pow(mFS, 2)
                                * (
                                    4 * pow(mdm1, 2)
                                    - 2 * (m23sq + pow(mdm2, 2))
                                    + pow(mFS, 2)
                                )
                            )
                            + 2 * (-m23sq + pow(mdm2, 2)) * pow(mIS, 2)
                            - 2 * qsq * (-m23sq + pow(mdm2, 2) + pow(mIS, 2))
                            + pow(mIS, 4)
                            + pow(qsq, 2)
                        )
                        + 2
                        * FFA0
                        * (mdm1 - mdm2)
                        * (mdm1 + mdm2)
                        * mFS
                        * (
                            -2 * (qsq + pow(mFS, 2)) * pow(mIS, 2)
                            + pow(mIS, 4)
                            + pow(qsq - pow(mFS, 2), 2)
                        )
                        - 2
                        * FFA3
                        * (mdm1 - mdm2)
                        * (mdm1 + mdm2)
                        * mFS
                        * (
                            -2 * (qsq + pow(mFS, 2)) * pow(mIS, 2)
                            + pow(mIS, 4)
                            + pow(qsq - pow(mFS, 2), 2)
                        )
                    )
                )
            )
            / 2.0
        )
    )

    amp_sq = J_11 + J_22 + J_33 + J_13
    return f_phase_space * amp_sq


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_P = wcs['P']
    wc_A = wcs['A']
    FFA0, FFA1, FFA2, FFA3 = ffs['A0'], ffs['A1'], ffs['A2'], ffs['A3']

    ampSq22 = (
        wc_P
        * wc_P.conjugate()
        * np.power(m_fq + m_iq, -2)
        * np.power(FFA0, 2)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq24 = -0.5 * (
        wc_A
        * wc_P.conjugate()
        * FFA0
        * (
            2 * m_FS * FFA0
            + (m_FS + m_IS) * FFA1
            + m_FS * FFA2
            - m_IS * FFA2
            - 2 * m_FS * FFA3
        )
        * np.power(m_FS, -1)
        * np.power(m_fq + m_iq, -1)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq42 = -0.5 * (
        wc_A.conjugate()
        * wc_P
        * FFA0
        * (
            2 * m_FS * FFA0
            + (m_FS + m_IS) * FFA1
            + m_FS * FFA2
            - m_IS * FFA2
            - 2 * m_FS * FFA3
        )
        * np.power(m_FS, -1)
        * np.power(m_fq + m_iq, -1)
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq44 = (
        wc_A
        * wc_A.conjugate()
        * np.power(m_FS, -2)
        * np.power(
            2 * m_FS * FFA0
            + (m_FS + m_IS) * FFA1
            + m_FS * FFA2
            - m_IS * FFA2
            - 2 * m_FS * FFA3,
            2,
        )
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    ) / 4.0

    return ampSq22 + ampSq24 + ampSq42 + ampSq44


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq, theta1
) -> float:
    wc_P = wcs['P']
    wc_V = wcs['V']
    wc_A = wcs['A']
    FFV, FFA0, FFA1, FFA2, FFA3 = ffs['V'], ffs['A0'], ffs['A1'], ffs['A2'], ffs['A3']

    ampSq22 = (
        4
        * wc_P
        * wc_P.conjugate()
        * np.power(m_fq + m_iq, -2)
        * np.power(FFA0, 2)
        * (
            -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq24 = (
        2
        * 1j
        * wc_A
        * wc_P.conjugate()
        * FFA0
        * np.power(m_FS, -1)
        * np.power(m_fq + m_iq, -1)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            (m_FS + m_IS)
            * (
                2 * m_FS * FFA0
                + (m_FS + m_IS) * FFA1
                + m_FS * FFA2
                - m_IS * FFA2
                - 2 * m_FS * FFA3
            )
            * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (
                -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            )
            + np.cos(theta1)
            * (
                FFA1
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                * np.power(m_FS + m_IS, 2)
                + FFA2
                * (
                    -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
                )
            )
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
        -4
        * wc_V
        * wc_V.conjugate()
        * np.power(m_FS + m_IS, -2)
        * np.power(FFV, 2)
        * np.power(qsq, -1)
        * (-1 + np.power(np.cos(theta1), 2))
        * (
            np.power(m_dm1, 4)
            - 2 * np.power(m_dm1, 2) * (qsq + np.power(m_dm2, 2))
            + np.power(qsq - np.power(m_dm2, 2), 2)
        )
        * (
            -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
            + np.power(qsq, 2)
            + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
        )
    )
    ampSq42 = (
        -2
        * 1j
        * wc_A.conjugate()
        * wc_P
        * FFA0
        * np.power(m_FS, -1)
        * np.power(m_fq + m_iq, -1)
        * np.power(m_FS + m_IS, -1)
        * np.power(qsq, -1)
        * (
            (m_FS + m_IS)
            * (
                2 * m_FS * FFA0
                + (m_FS + m_IS) * FFA1
                + m_FS * FFA2
                - m_IS * FFA2
                - 2 * m_FS * FFA3
            )
            * (np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (
                -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                + np.power(qsq, 2)
                + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            )
            + np.cos(theta1)
            * (
                FFA1
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                * np.power(m_FS + m_IS, 2)
                + FFA2
                * (
                    -2 * qsq * (np.power(m_FS, 2) + np.power(m_IS, 2))
                    + np.power(qsq, 2)
                    + np.power(np.power(m_FS, 2) - np.power(m_IS, 2), 2)
                )
            )
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
    ampSq44 = (
        wc_A
        * wc_A.conjugate()
        * np.power(m_FS, -2)
        * np.power(m_FS + m_IS, -2)
        * np.power(qsq, -4)
        * (
            128
            * FFA1
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 3)
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 3)
            + 128
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 4)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 3)
            - 64
            * np.power(m_dm1, 2)
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 4)
            - 64
            * np.power(m_dm2, 2)
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 4)
            - 64
            * (-qsq + np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 4)
            - 64
            * FFA1
            * (FFA0 - FFA3)
            * np.power(m_FS, 3)
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 3)
            * np.power(qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2), 2)
            - 64
            * np.power(m_FS, 4)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 3)
            * np.power(qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2), 2)
            - 64
            * FFA1
            * (FFA0 - FFA3)
            * np.power(m_FS, 3)
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 3)
            * np.power(qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2), 2)
            - 64
            * np.power(m_FS, 4)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 3)
            * np.power(qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2), 2)
            - 32
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            + 16
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2), 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            + 16
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(FFA0 - FFA3, 2)
            * np.power(qsq, 2)
            * np.power(qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2), 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            + 64
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(m_FS, 3)
            * np.power(qsq, 3)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            - 64
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 3)
            * np.power(qsq, 3)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 32
            * FFA1
            * FFA2
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 3)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            - 32
            * FFA1
            * FFA2
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 3)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 16
            * m_FS
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 16
            * m_FS
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * (
                -(
                    (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            - 16
            * m_FS
            * FFA1
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 16
            * m_FS
            * FFA1
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 8
            * FFA1
            * FFA2
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            + 16
            * m_FS
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * (
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 64
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(m_FS, 3)
            * np.power(qsq, 3)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 64
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 3)
            * np.power(qsq, 3)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 32
            * FFA1
            * FFA2
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 3)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 32
            * FFA1
            * FFA2
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * np.power(m_FS, 2)
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 3)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 16
            * m_FS
            * (m_FS + m_IS)
            * FFA2
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 32
            * np.power(m_FS, 2)
            * np.power(FFA2, 2)
            * np.power(qsq, 3)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 8
            * FFA1
            * FFA2
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 8
            * np.power(FFA2, 2)
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * (
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 16
            * m_FS
            * FFA1
            * (FFA0 - FFA3)
            * (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 2)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 16
            * m_FS
            * FFA1
            * (FFA0 - FFA3)
            * (qsq - np.power(m_dm1, 2) + np.power(m_dm2, 2))
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 3)
            * np.power(qsq, 2)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 8
            * FFA1
            * FFA2
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 2)
            * (
                -(
                    (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                    * (np.power(m_FS, 2) - np.power(m_IS, 2))
                )
                - np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 8
            * FFA1
            * FFA2
            * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
            * np.power(m_FS + m_IS, 2)
            * np.power(qsq, 2)
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 8
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 2)
            * (
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            * (
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 16
            * np.power(m_FS, 2)
            * np.power(FFA2, 2)
            * np.power(qsq, 3)
            * np.power(
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 4
            * np.power(FFA2, 2)
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * np.power(
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            - 16
            * np.power(m_FS, 2)
            * np.power(FFA2, 2)
            * np.power(qsq, 3)
            * np.power(
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 4
            * np.power(FFA2, 2)
            * np.power(qsq, 2)
            * np.power(qsq + np.power(m_FS, 2) - np.power(m_IS, 2), 2)
            * np.power(
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 4
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 2)
            * np.power(
                (-qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
            + 4
            * np.power(m_FS + m_IS, 4)
            * np.power(FFA1, 2)
            * np.power(qsq, 2)
            * np.power(
                (qsq + np.power(m_dm1, 2) - np.power(m_dm2, 2))
                * (qsq + np.power(m_FS, 2) - np.power(m_IS, 2))
                + np.cos(theta1)
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
    ) / 16.0

    return ampSq22 + ampSq24 + ampSq42 + ampSq44 + ampSq33
