import numpy as np


def amp_square_2_1_1(wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1) -> float:
    # 仅提取 V, A, T, T5
    wc_V = wcs["V"]
    wc_A = wcs["A"]
    wc_T = wcs["T"]
    wc_T5 = wcs["T5"]

    FFV = ffs["V"]
    FFA0 = ffs["A0"]
    FFA1 = ffs["A1"]
    FFA2 = ffs["A2"]
    FFA3 = (m_IS + m_FS) / (2 * m_FS) * FFA1 - (m_IS - m_FS) / (2 * m_FS) * FFA2
    FFT1 = ffs["T1"]
    FFT2 = ffs["T2"]
    FFT3 = ffs["T3"]

    # 初始化 4x4 = 16 个振幅平方项
    ampSq1_1=2.*wc_V*wc_V.conjugate()*np.power(m_FS + m_IS,-2)*np.power(FFV,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_2 = 0
    ampSq1_3=-4.*wc_T*wc_V.conjugate()*FFT1*FFV*np.power(m_FS + m_IS,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq1_4 = 0

    ampSq2_1 = 0
    ampSq2_2=0.25*wc_A*wc_A.conjugate()*np.power(m_dm1,-2)*np.power(m_FS,-2)*np.power(m_FS + m_IS,-2)*(np.power(m_FS + m_IS,4)*np.power(FFA1,2)*(np.power(m_dm1,4) + 2.*np.power(m_dm1,2)*(5.*np.power(m_FS,2) -np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + 2.*FFA1*FFA2*np.power(m_FS + m_IS,2)*(np.power(m_dm1,6) -np.power(m_dm1,4)*(np.power(m_FS,2) + 3.*np.power(m_IS,2)) -np.power(m_dm1,2)*(np.power(m_FS,4) + 2.*np.power(m_FS,2)*np.power(m_IS,2) - 3.*np.power(m_IS,4)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),3)) + np.power(FFA2,2)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),2))
    ampSq2_3 = 0
    ampSq2_4=0.5*1j*wc_A.conjugate()*wc_T5*np.power(m_FS,-2)*np.power(m_FS -m_IS,-1)*np.power(m_FS + m_IS,-2)*(FFA2*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))*(-1.*FFT2*(np.power(m_dm1,2) - 3.*np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_FS,2) -np.power(m_IS,2)) + FFT3*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))) + FFA1*np.power(m_FS + m_IS,2)*(-1.*FFT2*(np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_dm1,4) - 11.*np.power(m_FS,4) + 10.*np.power(m_FS,2)*np.power(m_IS,2) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(m_IS,4)) + FFT3*(np.power(m_dm1,6) -np.power(m_dm1,4)*(np.power(m_FS,2) + 3.*np.power(m_IS,2)) -np.power(m_dm1,2)*(np.power(m_FS,4) + 2.*np.power(m_FS,2)*np.power(m_IS,2) - 3.*np.power(m_IS,4)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),3))))

    ampSq3_1=-4.*wc_T.conjugate()*wc_V*FFT1*FFV*np.power(m_FS + m_IS,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq3_2 = 0
    ampSq3_3=8.*wc_T*wc_T.conjugate()*np.power(FFT1,2)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq3_4 = 0

    ampSq4_1 = 0
    ampSq4_2=-0.5*1j*wc_A*wc_T5.conjugate()*np.power(m_FS,-2)*np.power(m_FS -m_IS,-1)*np.power(m_FS + m_IS,-2)*(FFA2*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))*(-1.*FFT2*(np.power(m_dm1,2) - 3.*np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_FS,2) -np.power(m_IS,2)) + FFT3*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))) + FFA1*np.power(m_FS + m_IS,2)*(-1.*FFT2*(np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_dm1,4) - 11.*np.power(m_FS,4) + 10.*np.power(m_FS,2)*np.power(m_IS,2) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(m_IS,4)) + FFT3*(np.power(m_dm1,6) -np.power(m_dm1,4)*(np.power(m_FS,2) + 3.*np.power(m_IS,2)) -np.power(m_dm1,2)*(np.power(m_FS,4) + 2.*np.power(m_FS,2)*np.power(m_IS,2) - 3.*np.power(m_IS,4)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),3))))
    ampSq4_3 = 0
    ampSq4_4=wc_T5*wc_T5.conjugate()*np.power(np.power(m_FS,3) -m_FS*np.power(m_IS,2),-2)*(-2.*FFT2*FFT3*np.power(m_dm1,2)*(np.power(m_FS,2) -np.power(m_IS,2))*(np.power(m_dm1,6) -np.power(m_dm1,4)*(5.*np.power(m_FS,2) + 3.*np.power(m_IS,2)) + np.power(m_dm1,2)*(7.*np.power(m_FS,4) + 6.*np.power(m_FS,2)*np.power(m_IS,2) + 3.*np.power(m_IS,4)) -(3.*np.power(m_FS,2) + np.power(m_IS,2))*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)) + np.power(FFT2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)*(np.power(m_dm1,6) - 2.*np.power(m_dm1,4)*(3.*np.power(m_FS,2) + np.power(m_IS,2)) + np.power(m_dm1,2)*np.power(3.*np.power(m_FS,2) + np.power(m_IS,2),2) + 8.*np.power(np.power(m_FS,3) -m_FS*np.power(m_IS,2),2)) + np.power(m_dm1,2)*np.power(FFT3,2)*np.power(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2),2))

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
