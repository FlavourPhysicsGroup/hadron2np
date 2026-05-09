import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chibarchibar"""
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_T = wcs["T"]

    FFV = ffs["V"]
    FFA0 = ffs["A0"]
    FFA1 = ffs["A1"]
    FFA2 = ffs["A2"]
    FFA3 = (m_IS + m_FS) / (2 * m_FS) * FFA1 - (m_IS - m_FS) / (2 * m_FS) * FFA2
    FFT1 = ffs["T1"]
    FFT2 = ffs["T2"]
    FFT3 = ffs["T3"]

    ampSq1_1 = 0
    ampSq1_2 = 0
    ampSq1_3 = 0

    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = 0

    return (
        ampSq1_1
        + ampSq1_2
        + ampSq1_3
        + ampSq2_1
        + ampSq2_2
        + ampSq2_3
        + ampSq3_1
        + ampSq3_2
        + ampSq3_3
    ).real
