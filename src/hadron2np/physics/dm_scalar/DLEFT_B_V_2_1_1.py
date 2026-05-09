import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    # 提取 wcs: S, P, V, A
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    FFV = ffs["V"]
    FFA0 = ffs["A0"]
    FFA1 = ffs["A1"]
    FFA2 = ffs["A2"]
    FFA3 = (m_IS + m_FS) / (2 * m_FS) * FFA1 - (m_IS - m_FS) / (2 * m_FS) * FFA2
    FFT1 = ffs["T1"]
    FFT2 = ffs["T2"]
    FFT3 = ffs["T3"]

    # 初始化 4x4 = 16 个振幅平方项
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1 = 0
    ampSq1_2 = 0
    ampSq1_3 = 0
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2=wc_P*wc_P.conjugate()*np.power(m_fq + m_iq,-2)*np.power(FFA0,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq2_3 = 0
    ampSq2_4=-0.5*wc_A*wc_P.conjugate()*FFA0*(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3)*np.power(m_FS,-1)*np.power(m_fq + m_iq,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))

    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = 0
    ampSq3_4 = 0

    ampSq4_1 = 0
    ampSq4_2=-0.5*wc_A.conjugate()*wc_P*FFA0*(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3)*np.power(m_FS,-1)*np.power(m_fq + m_iq,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq4_3 = 0
    ampSq4_4=0.25*wc_A*wc_A.conjugate()*np.power(m_FS,-2)*np.power(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))

    # 返回16个元素和的实部
    return (
        ampSq1_1
        + ampSq1_2
        + ampSq1_3
        + ampSq1_4
        + ampSq2_1
        + ampSq2_2
        + ampSq2_3
        + ampSq2_4
        + ampSq3_1
        + ampSq3_2
        + ampSq3_3
        + ampSq3_4
        + ampSq4_1
        + ampSq4_2
        + ampSq4_3
        + ampSq4_4
    ).real