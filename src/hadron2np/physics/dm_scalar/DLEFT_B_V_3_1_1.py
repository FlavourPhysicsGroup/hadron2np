import numpy as np

def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    # 提取 wcs: 只有 S, P, V, A
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_V = wcs["V"]
    wc_A = wcs["A"]

    # 提取 ffs for B -> V (Vector meson): 与模板相同
    FFV = ffs["V"]
    FFA0 = ffs["A0"]
    FFA1 = ffs["A1"]
    FFA2 = ffs["A2"]
    FFA3 = (m_IS + m_FS) / (2 * m_FS) * FFA1 - (m_IS - m_FS) / (2 * m_FS) * FFA2
    FFT1 = ffs["T1"]
    FFT2 = ffs["T2"]
    FFT3 = ffs["T3"]

    # 初始化 16 个振幅平方项
    # 注意：此处公式需要根据具体物理过程填充，目前设为 0
    ampSq1_1 = 0
    ampSq1_2 = 0
    ampSq1_3 = 0
    ampSq1_4 = 0
    
    ampSq2_1 = 0
    ampSq2_2=8.*wc_P*wc_P.conjugate()*np.power(m_fq + m_iq,-2)*np.power(FFA0,2)*(-1.*m_FS -m_IS + np.power(qsq,0.5))*(m_FS -m_IS + np.power(qsq,0.5))*(-1.*m_FS + m_IS + np.power(qsq,0.5))*(m_FS + m_IS + np.power(qsq,0.5))
    ampSq2_3 = 0
    ampSq2_4=4.*1j*wc_A*wc_P.conjugate()*FFA0*(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3)*(np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_FS,-1)*np.power(m_fq + m_iq,-1)*np.power(qsq,-1)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    
    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3=5.333333333333333*wc_V*wc_V.conjugate()*np.power(m_FS + m_IS,-2)*np.power(FFV,2)*np.power(qsq,-1)*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq3_4 = 0
    
    ampSq4_1 = 0
    ampSq4_2=-4.*1j*wc_A.conjugate()*wc_P*FFA0*(2.*m_FS*FFA0 + (m_FS + m_IS)*FFA1 + m_FS*FFA2 -m_IS*FFA2 - 2.*m_FS*FFA3)*(np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_FS,-1)*np.power(m_fq + m_iq,-1)*np.power(qsq,-1)*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))
    ampSq4_3 = 0
    ampSq4_4=0.16666666666666666*wc_A*wc_A.conjugate()*np.power(m_FS,-2)*np.power(m_FS + m_IS,-2)*np.power(qsq,-2)*(4.*(np.power(m_dm1,4) - 2.*np.power(m_dm1,2)*(qsq + np.power(m_dm2,2)) + np.power(qsq -np.power(m_dm2,2),2))*(-2.*qsq*(np.power(m_FS,2) + np.power(m_IS,2)) + np.power(qsq,2) + np.power(np.power(m_FS,2) -np.power(m_IS,2),2))*(2.*FFA1*FFA2*(qsq + np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2) + np.power(m_FS + m_IS,4)*np.power(FFA1,2) - 4.*qsq*np.power(m_FS,2)*np.power(FFA2,2) + np.power(FFA2,2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2)) + 3.*(32.*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*qsq*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,3)*(np.power(m_FS,2) -np.power(m_IS,2)) + 16.*FFA1*FFA2*qsq*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,2)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2) + 32.*FFA1*(FFA0 -FFA3)*qsq*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,3)*np.power(m_FS + m_IS,3) + 32.*qsq*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,4)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2) - 16.*np.power(m_dm1,2)*np.power(m_FS,2)*np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq,2) - 16.*np.power(m_dm2,2)*np.power(m_FS,2)*np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq,2) - 16.*(-1.*qsq + np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,2)*np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq,2) - 16.*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*qsq*np.power(m_FS,3)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2) - 8.*FFA1*FFA2*qsq*np.power(m_FS,2)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2) - 16.*FFA1*(FFA0 -FFA3)*qsq*np.power(m_FS,3)*np.power(m_FS + m_IS,3)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2) - 16.*qsq*np.power(m_FS,4)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2) - 16.*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*qsq*np.power(m_FS,3)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2) - 8.*FFA1*FFA2*qsq*np.power(m_FS,2)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2) - 16.*FFA1*(FFA0 -FFA3)*qsq*np.power(m_FS,3)*np.power(m_FS + m_IS,3)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2) - 16.*qsq*np.power(m_FS,4)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2) + 8.*qsq*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,2)*np.power(FFA2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) - 4.*qsq*np.power(m_FS,2)*np.power(FFA2,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) - 4.*qsq*np.power(m_FS,2)*np.power(FFA2,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*(-1.*qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) - 4.*m_FS*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*FFA1*FFA2*(-1.*qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*FFA1*(FFA0 -FFA3)*(-1.*qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_FS + m_IS,3)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) - 4.*m_FS*FFA1*(FFA0 -FFA3)*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS + m_IS,3)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 2.*(-1.*qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) - 8.*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq -np.power(m_dm1,2) + np.power(m_dm2,2))*np.power(m_FS,2)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 2.*FFA1*FFA2*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*FFA1*(FFA0 -FFA3)*np.power(m_FS + m_IS,3)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*np.power(m_FS,2)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*(m_FS + m_IS)*FFA2*(FFA0 -FFA3)*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 2.*FFA1*FFA2*(np.power(m_FS,2) -np.power(m_IS,2))*np.power(m_FS + m_IS,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*m_FS*FFA1*(FFA0 -FFA3)*np.power(m_FS + m_IS,3)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(m_FS + m_IS,4)*np.power(FFA1,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 4.*np.power(m_FS,2)*np.power(m_FS + m_IS,2)*np.power(FFA0 -FFA3,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + 2.*(-1.*qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*(qsq + np.power(m_dm1,2) -np.power(m_dm2,2))*np.power(FFA2,2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(FFA2,2)*np.power(qsq + np.power(m_dm1,2) -np.power(m_dm2,2),2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2) + np.power(FFA2,2)*np.power(qsq -np.power(m_dm1,2) + np.power(m_dm2,2),2)*np.power(np.power(m_FS,2) -np.power(m_IS,2),2)*np.power(qsq + np.power(m_FS,2) -np.power(m_IS,2),2)))

    return (
        ampSq1_1 + ampSq1_2 + ampSq1_3 + ampSq1_4 +
        ampSq2_1 + ampSq2_2 + ampSq2_3 + ampSq2_4 +
        ampSq3_1 + ampSq3_2 + ampSq3_3 + ampSq3_4 +
        ampSq4_1 + ampSq4_2 + ampSq4_3 + ampSq4_4
    ).real