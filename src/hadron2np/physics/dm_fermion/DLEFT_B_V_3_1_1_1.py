import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    """For Bs -> chichi"""
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
    ampSq2_2=-8.*wc_P*wc_P.conjugate()*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_fq + m_iq,-2)*np.power(FFA0,2)*(-1.*m_FS -m_IS + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))
    ampSq2_3 = 0

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3=-2.6666666666666665*np.power(m_FS,-2)*np.power(m_FS -m_IS,-2)*np.power(m_FS + m_IS,-2)*np.power(qsq,-2)*(2.*np.power(m_dm1,4) -qsq*np.power(m_dm2,2) -np.power(m_dm1,2)*(qsq + 4.*np.power(m_dm2,2)) + 2.*np.power(m_dm2,4) -np.power(qsq,2))*np.power(np.abs(wc_T),2)*(-2.*(m_FS -m_IS)*(m_FS + m_IS)*FFT2*FFT3*qsq*(qsq - 3.*np.power(m_FS,2) -np.power(m_IS,2))*(-1.*m_FS -m_IS + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5)) + np.power(m_FS -m_IS,2)*np.power(m_FS + m_IS,2)*(8.*np.power(m_FS,2)*np.power(FFT1,2)*(m_FS -m_IS -np.power(qsq,0.5))*(m_FS + m_IS -np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5)) + np.power(FFT2,2)*(-2.*(3.*np.power(m_FS,2) + np.power(m_IS,2))*np.power(qsq,2) + np.power(qsq,3) + qsq*np.power(3.*np.power(m_FS,2) + np.power(m_IS,2),2) + 8.*np.power(np.power(m_FS,3) -m_FS*np.power(m_IS,2),2))) + qsq*np.power(FFT3,2)*np.power(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),2))

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
