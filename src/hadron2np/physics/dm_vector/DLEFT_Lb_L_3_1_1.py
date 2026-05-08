import numpy as np


def amp_square_3_1_1(
    wcs: dict, ffs: dict, m_iq, m_fq, m_IS, m_FS, m_dm1, m_dm2, qsq
) -> float:
    wc_S = wcs["S"]
    wc_P = wcs["P"]
    wc_V = wcs["V"]
    wc_A = wcs["A"]
    wc_V_tilde = wcs["V_tilde"]
    wc_A_tilde = wcs["A_tilde"]
    wc_T = wcs["T"]
    wc_T5 = wcs["T5"]
    wc_V_partial = wcs["V_partial"]
    wc_A_partial = wcs["A_partial"]
    wc_V_21 = wcs["V_21"]
    wc_A_21 = wcs["A_21"]
    wc_V_tilde_21 = wcs["V_tilde_21"]
    wc_A_tilde_21 = wcs["A_tilde_21"]

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
    ampSq1_4 = 0
    ampSq1_5 = 0
    ampSq1_6 = 0
    ampSq1_7 = 0
    ampSq1_8 = 0
    ampSq1_9 = 0
    ampSq1_10 = 0
    ampSq1_11 = 0
    ampSq1_12 = 0
    ampSq1_13 = 0
    ampSq1_14 = 0
    ampSq2_1 = 0
    ampSq2_2 = 0
    ampSq2_3 = 0
    ampSq2_4 = 0
    ampSq2_5 = 0
    ampSq2_6 = 0
    ampSq2_7 = 0
    ampSq2_8 = 0
    ampSq2_9 = 0
    ampSq2_10 = 0
    ampSq2_11 = 0
    ampSq2_12 = 0
    ampSq2_13 = 0
    ampSq2_14 = 0
    ampSq3_1 = 0
    ampSq3_2 = 0
    ampSq3_3 = 0
    ampSq3_4 = 0
    ampSq3_5 = 0
    ampSq3_6 = 0
    ampSq3_7 = 0
    ampSq3_8 = 0
    ampSq3_9 = 0
    ampSq3_10 = 0
    ampSq3_11 = 0
    ampSq3_12 = 0
    ampSq3_13 = 0
    ampSq3_14 = 0
    ampSq4_1 = 0
    ampSq4_2 = 0
    ampSq4_3 = 0
    ampSq4_4 = 0
    ampSq4_5 = 0
    ampSq4_6 = 0
    ampSq4_7 = 0
    ampSq4_8 = 0
    ampSq4_9 = 0
    ampSq4_10 = 0
    ampSq4_11 = 0
    ampSq4_12 = 0
    ampSq4_13 = 0
    ampSq4_14 = 0
    ampSq5_1 = 0
    ampSq5_2 = 0
    ampSq5_3 = 0
    ampSq5_4 = 0
    ampSq5_5 = 0
    ampSq5_6 = 0
    ampSq5_7 = 0
    ampSq5_8 = 0
    ampSq5_9 = 0
    ampSq5_10 = 0
    ampSq5_11 = 0
    ampSq5_12 = 0
    ampSq5_13 = 0
    ampSq5_14 = 0
    ampSq6_1 = 0
    ampSq6_2 = 0
    ampSq6_3 = 0
    ampSq6_4 = 0
    ampSq6_5 = 0
    ampSq6_6 = 0
    ampSq6_7 = 0
    ampSq6_8 = 0
    ampSq6_9 = 0
    ampSq6_10 = 0
    ampSq6_11 = 0
    ampSq6_12 = 0
    ampSq6_13 = 0
    ampSq6_14 = 0
    ampSq7_1 = 0
    ampSq7_2 = 0
    ampSq7_3 = 0
    ampSq7_4 = 0
    ampSq7_5 = 0
    ampSq7_6 = 0
    ampSq7_7 = 0
    ampSq7_8 = 0
    ampSq7_9 = 0
    ampSq7_10 = 0
    ampSq7_11 = 0
    ampSq7_12 = 0
    ampSq7_13 = 0
    ampSq7_14 = 0
    ampSq8_1 = 0
    ampSq8_2 = 0
    ampSq8_3 = 0
    ampSq8_4 = 0
    ampSq8_5 = 0
    ampSq8_6 = 0
    ampSq8_7 = 0
    ampSq8_8 = 0
    ampSq8_9 = 0
    ampSq8_10 = 0
    ampSq8_11 = 0
    ampSq8_12 = 0
    ampSq8_13 = 0
    ampSq8_14 = 0
    ampSq9_1 = 0
    ampSq9_2 = 0
    ampSq9_3 = 0
    ampSq9_4 = 0
    ampSq9_5 = 0
    ampSq9_6 = 0
    ampSq9_7 = 0
    ampSq9_8 = 0
    ampSq9_9 = 0
    ampSq9_10 = 0
    ampSq9_11 = 0
    ampSq9_12 = 0
    ampSq9_13 = 0
    ampSq9_14 = 0
    ampSq10_1 = 0
    ampSq10_2 = 0
    ampSq10_3 = 0
    ampSq10_4 = 0
    ampSq10_5 = 0
    ampSq10_6 = 0
    ampSq10_7 = 0
    ampSq10_8 = 0
    ampSq10_9 = 0
    ampSq10_10 = 0
    ampSq10_11 = 0
    ampSq10_12 = 0
    ampSq10_13 = 0
    ampSq10_14 = 0
    ampSq11_1 = 0
    ampSq11_2 = 0
    ampSq11_3 = 0
    ampSq11_4 = 0
    ampSq11_5 = 0
    ampSq11_6 = 0
    ampSq11_7 = 0
    ampSq11_8 = 0
    ampSq11_9 = 0
    ampSq11_10 = 0
    ampSq11_11 = 0
    ampSq11_12 = 0
    ampSq11_13 = 0
    ampSq11_14 = 0
    ampSq12_1 = 0
    ampSq12_2 = 0
    ampSq12_3 = 0
    ampSq12_4 = 0
    ampSq12_5 = 0
    ampSq12_6 = 0
    ampSq12_7 = 0
    ampSq12_8 = 0
    ampSq12_9 = 0
    ampSq12_10 = 0
    ampSq12_11 = 0
    ampSq12_12 = 0
    ampSq12_13 = 0
    ampSq12_14 = 0
    ampSq13_1 = 0
    ampSq13_2 = 0
    ampSq13_3 = 0
    ampSq13_4 = 0
    ampSq13_5 = 0
    ampSq13_6 = 0
    ampSq13_7 = 0
    ampSq13_8 = 0
    ampSq13_9 = 0
    ampSq13_10 = 0
    ampSq13_11 = 0
    ampSq13_12 = 0
    ampSq13_13 = 0
    ampSq13_14 = 0
    ampSq14_1 = 0
    ampSq14_2 = 0
    ampSq14_3 = 0
    ampSq14_4 = 0
    ampSq14_5 = 0
    ampSq14_6 = 0
    ampSq14_7 = 0
    ampSq14_8 = 0
    ampSq14_9 = 0
    ampSq14_10 = 0
    ampSq14_11 = 0
    ampSq14_12 = 0
    ampSq14_13 = 0
    ampSq14_14 = 0

    return (
        ampSq1_1 + ampSq1_2 + ampSq1_3 + ampSq1_4 + ampSq1_5 + ampSq1_6 + ampSq1_7 + ampSq1_8 + ampSq1_9 + ampSq1_10 + ampSq1_11 + ampSq1_12 + ampSq1_13 + ampSq1_14 +
        ampSq2_1 + ampSq2_2 + ampSq2_3 + ampSq2_4 + ampSq2_5 + ampSq2_6 + ampSq2_7 + ampSq2_8 + ampSq2_9 + ampSq2_10 + ampSq2_11 + ampSq2_12 + ampSq2_13 + ampSq2_14 +
        ampSq3_1 + ampSq3_2 + ampSq3_3 + ampSq3_4 + ampSq3_5 + ampSq3_6 + ampSq3_7 + ampSq3_8 + ampSq3_9 + ampSq3_10 + ampSq3_11 + ampSq3_12 + ampSq3_13 + ampSq3_14 +
        ampSq4_1 + ampSq4_2 + ampSq4_3 + ampSq4_4 + ampSq4_5 + ampSq4_6 + ampSq4_7 + ampSq4_8 + ampSq4_9 + ampSq4_10 + ampSq4_11 + ampSq4_12 + ampSq4_13 + ampSq4_14 +
        ampSq5_1 + ampSq5_2 + ampSq5_3 + ampSq5_4 + ampSq5_5 + ampSq5_6 + ampSq5_7 + ampSq5_8 + ampSq5_9 + ampSq5_10 + ampSq5_11 + ampSq5_12 + ampSq5_13 + ampSq5_14 +
        ampSq6_1 + ampSq6_2 + ampSq6_3 + ampSq6_4 + ampSq6_5 + ampSq6_6 + ampSq6_7 + ampSq6_8 + ampSq6_9 + ampSq6_10 + ampSq6_11 + ampSq6_12 + ampSq6_13 + ampSq6_14 +
        ampSq7_1 + ampSq7_2 + ampSq7_3 + ampSq7_4 + ampSq7_5 + ampSq7_6 + ampSq7_7 + ampSq7_8 + ampSq7_9 + ampSq7_10 + ampSq7_11 + ampSq7_12 + ampSq7_13 + ampSq7_14 +
        ampSq8_1 + ampSq8_2 + ampSq8_3 + ampSq8_4 + ampSq8_5 + ampSq8_6 + ampSq8_7 + ampSq8_8 + ampSq8_9 + ampSq8_10 + ampSq8_11 + ampSq8_12 + ampSq8_13 + ampSq8_14 +
        ampSq9_1 + ampSq9_2 + ampSq9_3 + ampSq9_4 + ampSq9_5 + ampSq9_6 + ampSq9_7 + ampSq9_8 + ampSq9_9 + ampSq9_10 + ampSq9_11 + ampSq9_12 + ampSq9_13 + ampSq9_14 +
        ampSq10_1 + ampSq10_2 + ampSq10_3 + ampSq10_4 + ampSq10_5 + ampSq10_6 + ampSq10_7 + ampSq10_8 + ampSq10_9 + ampSq10_10 + ampSq10_11 + ampSq10_12 + ampSq10_13 + ampSq10_14 +
        ampSq11_1 + ampSq11_2 + ampSq11_3 + ampSq11_4 + ampSq11_5 + ampSq11_6 + ampSq11_7 + ampSq11_8 + ampSq11_9 + ampSq11_10 + ampSq11_11 + ampSq11_12 + ampSq11_13 + ampSq11_14 +
        ampSq12_1 + ampSq12_2 + ampSq12_3 + ampSq12_4 + ampSq12_5 + ampSq12_6 + ampSq12_7 + ampSq12_8 + ampSq12_9 + ampSq12_10 + ampSq12_11 + ampSq12_12 + ampSq12_13 + ampSq12_14 +
        ampSq13_1 + ampSq13_2 + ampSq13_3 + ampSq13_4 + ampSq13_5 + ampSq13_6 + ampSq13_7 + ampSq13_8 + ampSq13_9 + ampSq13_10 + ampSq13_11 + ampSq13_12 + ampSq13_13 + ampSq13_14 +
        ampSq14_1 + ampSq14_2 + ampSq14_3 + ampSq14_4 + ampSq14_5 + ampSq14_6 + ampSq14_7 + ampSq14_8 + ampSq14_9 + ampSq14_10 + ampSq14_11 + ampSq14_12 + ampSq14_13 + ampSq14_14
    ).real
