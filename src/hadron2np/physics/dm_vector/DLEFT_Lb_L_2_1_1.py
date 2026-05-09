import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    wc_V = wcs["V"]
    wc_A = wcs["A"]
    wc_T = wcs["T"]
    wc_T5 = wcs["T5"]

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

    # 初始化 4x4 = 16 个振幅平方项
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1 = -1.*wc_V*wc_V.conjugate()*np.power(m_dm1,-2)*(np.power(m_dm1,2) -np.power(m_FS -m_IS,2))*(np.power(m_FS + m_IS,2)*np.power(FFFp,2) + 2.*np.power(m_dm1,2)*np.power(FFFv,2))
    ampSq1_2 = 0
    ampSq1_3 = 2.*(m_FS + m_IS)*wc_T*wc_V.conjugate()*(FFFp*FFhp + 2.*FFFv*FFhv)*(-1.*np.power(m_dm1,2) + np.power(m_FS -m_IS,2))
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2 = -1.*wc_A*wc_A.conjugate()*np.power(m_dm1,-2)*(np.power(m_dm1,2) -np.power(m_FS + m_IS,2))*(np.power(m_FS -m_IS,2)*np.power(FFGp,2) + 2.*np.power(m_dm1,2)*np.power(FFGv,2))
    ampSq2_3 = 0
    ampSq2_4 = 2.*1j*(m_FS -m_IS)*wc_A.conjugate()*wc_T5*(FFGp*FFhtp + 2.*FFGv*FFhtv)*(-1.*np.power(m_dm1,2) + np.power(m_FS + m_IS,2))

    ampSq3_1 = 2.*(m_FS + m_IS)*wc_T.conjugate()*wc_V*(FFFp*FFhp + 2.*FFFv*FFhv)*(-1.*np.power(m_dm1,2) + np.power(m_FS -m_IS,2))
    ampSq3_2 = 0
    ampSq3_3 = -4.*wc_T*wc_T.conjugate()*(np.power(m_dm1,2) -np.power(m_FS -m_IS,2))*(np.power(m_dm1,2)*np.power(FFhp,2) + 2.*np.power(m_FS + m_IS,2)*np.power(FFhv,2))
    ampSq3_4 = 0

    ampSq4_1 = 0
    ampSq4_2 = -2.*1j*(m_FS -m_IS)*wc_A*wc_T5.conjugate()*(FFGp*FFhtp + 2.*FFGv*FFhtv)*(-1.*np.power(m_dm1,2) + np.power(m_FS + m_IS,2))
    ampSq4_3 = 0
    ampSq4_4 = -4.*wc_T5*wc_T5.conjugate()*(np.power(m_dm1,2) -np.power(m_FS + m_IS,2))*(np.power(m_dm1,2)*np.power(FFhtp,2) + 2.*np.power(m_FS -m_IS,2)*np.power(FFhtv,2))

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