import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chibarchibar"""
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_T = wcs["T"]

    FFFp = ffs["f+"]
    FFF0 = ffs["f0"]
    FFFv = ffs["fp"]
    FFGp = ffs["g+"]
    FFG0 = ffs["g0"]
    FFGv = ffs["gp"]
    FFhp = ffs["h+"]
    FFhv = ffs["hp"]
    FFhtp = ffs["ht+"]
    FFhtv = ffs["htp"]

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
