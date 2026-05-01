from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f
from hadron2np.methods import phase_space_3_body


def Gamma_LambdaX(L_V, L_A, L_T, L_T5, mdm1, mIS, mFS, miq, mfq, ffs):
    lam_sqrt = sqrt(kallen_f(mIS**2, mFS**2, mdm1**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    FFFp = ffs['f+']
    FFFv = ffs['fp']
    FFGp = ffs['g+']
    FFGv = ffs['gp']
    FFhp = ffs['h+']
    FFhv = ffs['hp']
    FFhtp = ffs['h~+']
    FFhtv = ffs['h~p']

    J_11 = abs(L_V)**2 * (-(
        pow(mdm1, -2) * (pow(mdm1, 2) - pow(mFS - mIS, 2)) *
        (2 * pow(FFFv, 2) * pow(mdm1, 2) + pow(FFFp, 2) * pow(mFS + mIS, 2))))
    J_22 = abs(L_A)**2 * (-(
        pow(mdm1, -2) *
        (2 * pow(FFGv, 2) * pow(mdm1, 2) + pow(FFGp, 2) * pow(mFS - mIS, 2)) *
        (pow(mdm1, 2) - pow(mFS + mIS, 2))))
    J_33 = abs(L_T)**2 * (
        -4 * (pow(mdm1, 2) - pow(mFS - mIS, 2)) *
        (pow(FFhp, 2) * pow(mdm1, 2) + 2 * pow(FFhv, 2) * pow(mFS + mIS, 2)))
    J_44 = abs(L_T5)**2 * (-4 * (pow(FFhtp, 2) * pow(mdm1, 2) +
                                 2 * pow(FFhtv, 2) * pow(mFS - mIS, 2)) *
                           (pow(mdm1, 2) - pow(mFS + mIS, 2)))
    J_13 = 2 * (L_V *
                L_T.conjugate()).real * (2 * (FFFp * FFhp + 2 * FFFv * FFhv) *
                                         (mFS + mIS) *
                                         (-pow(mdm1, 2) + pow(mFS - mIS, 2)))
    J_24 = -2 * (L_A *
                 L_T5.conjugate()).imag * (-2 *
                                           (FFGp * FFhtp + 2 * FFGv * FFhtv) *
                                           (mFS - mIS) *
                                           (-pow(mdm1, 2) + pow(mFS + mIS, 2)))

    amp_sq = J_11 + J_22 + J_33 + J_44 + J_13 + J_24
    return f_phase_space * amp_sq


def dGamma_LambdaXX(L_S, L_P, L_V, L_A, L_V_dXtX, L_A_dXtX, L_T, L_T5, L_DV,
                    L_DA, mdm1, mdm2, mIS, mFS, miq, mfq, ffs, qsq):
    f_phase_space = 1 / (256 * pi**3 * mIS**3)

    FFF0 = ffs['f0']
    FFFp = ffs['f+']
    FFFv = ffs['fp']
    FFG0 = ffs['g0']
    FFGp = ffs['g+']
    FFGv = ffs['gp']
    FFhp = ffs['h+']
    FFhv = ffs['hp']
    FFhtp = ffs['h~+']
    FFhtv = ffs['h~p']

    ps3 = phase_space_3_body(mIS, mFS, mdm1, mdm2, qsq)
    m23sq = ps3.m23sq()
    m23sqsq = ps3.m23sq_sq()
    # m23sqcube = ps3.m23sq_cube()

    J_11 = abs(L_S)**2 * (
        -0.25 *
        (m23sq * pow(FFF0, 2) * pow(mdm1, -2) * pow(mdm2, -2) *
         pow(mfq - miq, -2) * pow(mFS - mIS, 2) * (qsq - pow(mFS + mIS, 2)) *
         (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
          (qsq - 5 * pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2))))
    J_22 = abs(L_P)**2 * (
        -0.25 *
        (m23sq * pow(FFG0, 2) * pow(mdm1, -2) * pow(mdm2, -2) *
         pow(mfq + miq, -2) * (qsq - pow(mFS - mIS, 2)) * pow(mFS + mIS, 2) *
         (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
          (qsq - 5 * pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2))))
    J_33 = abs(L_V)**2 * ((
        m23sq * pow(mdm2, -2) * pow(qsq - pow(mFS + mIS, 2), -1) *
        (4 * m23sqsq * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
         (qsq * pow(FFFv, 2) - pow(FFFp, 2) * pow(mFS + mIS, 2)) + 6 * m23sq *
         (-(pow(FFFv, 2) * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) + FFFp *
          (mFS + mIS) * pow(qsq, -1) *
          (FFFp * (mFS + mIS) * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) - FFF0 *
           (mFS - mIS) *
           (pow(mdm1, 4) * (-qsq + pow(mFS + mIS, 2)) + 2 * pow(mdm2, 2) *
            (-(pow(mdm2, 2) * pow(mFS + mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS + mIS, 2)) - pow(qsq, 2)) + pow(mdm1, 2) *
            (pow(mdm2, 2) * pow(mFS + mIS, 2) - qsq *
             (pow(mdm2, 2) + pow(mFS + mIS, 2)) + pow(qsq, 2))))) -
         3 * pow(qsq, -2) *
         (2 * pow(FFFv, 2) * pow(qsq, 2) *
          (-2 * pow(mdm1, 6) * pow(mFS, 2) - 2 * qsq * pow(mdm2, 2) *
           (pow(mdm2, 2) * (pow(mFS, 2) - 2 * pow(mIS, 2)) + pow(mFS, 2) *
            (pow(mFS, 2) + pow(mIS, 2))) + pow(mdm2, 4) *
           (pow(mFS, 4) - 4 * pow(mdm2, 2) * pow(mIS, 2) +
            2 * pow(mFS, 2) * pow(mIS, 2) - 3 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
           pow(mdm1, 4) *
           (-2 * qsq * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) -
            (pow(mFS, 2) - pow(mIS, 2)) *
            (2 * pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           2 * pow(mdm1, 2) *
           (-(pow(mdm2, 2) *
              (-3 * pow(mFS, 4) + 5 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
               (2 * pow(mFS, 2) + pow(mIS, 2)) - 2 * pow(mIS, 4))) + qsq *
            (2 * pow(mdm2, 4) - pow(mdm2, 2) *
             (3 * pow(mFS, 2) + 2 * pow(mIS, 2)) + pow(mIS, 4)) -
            (pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) + pow(qsq, 3)) +
           pow(qsq, 4)) - 2 * FFF0 * FFFp * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 6) * (2 * mIS * (mFS + mIS) * qsq +
                           (mFS - mIS) * pow(mFS + mIS, 3) - pow(qsq, 2)) +
           2 * qsq * pow(mdm1, 4) *
           (-(qsq *
              (2 * mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2))) +
            (pow(mdm2, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2) + pow(qsq, 2)) -
           pow(mdm1, 2) *
           (-2 * (mFS + mIS) * qsq * pow(mdm2, 2) *
            ((mFS - 2 * mIS) * pow(mdm2, 2) + 2 * mIS * pow(mFS, 2) +
             2 * pow(mFS, 3) - mFS * pow(mIS, 2) - pow(mIS, 3)) + 3 *
            (mFS - mIS) * pow(mdm2, 4) * pow(mFS + mIS, 3) +
            (-pow(mdm2, 4) + pow(mdm2, 2) *
             (4 * pow(mFS, 2) - 2 * pow(mIS, 2)) +
             (pow(mFS, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2)) * pow(qsq, 2) - 2 *
            (mFS * mIS + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4)) + 2 * pow(mdm2, 2) *
           (-2 * mFS * (mFS + mIS) * qsq * pow(mdm2, 2) *
            (mFS * (mFS + mIS) + pow(mdm2, 2)) +
            (mFS - mIS) * pow(mdm2, 4) * pow(mFS + mIS, 3) +
            (pow(mdm2, 4) + 2 * pow(mdm2, 2) *
             (2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) +
             (pow(mFS, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2)) * pow(qsq, 2) - 2 *
            (mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) *
            pow(qsq, 3) + pow(qsq, 4))) + pow(FFFp, 2) * pow(mFS + mIS, 2) *
          (-2 * qsq * pow(mdm1, 4) *
           (-2 * qsq * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) -
            (pow(mFS, 2) - pow(mIS, 2)) *
            (2 * pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
           pow(mdm1, 2) *
           (-2 * qsq * pow(mdm2, 2) *
            (-pow(mFS, 4) + pow(mdm2, 2) *
             (pow(mFS, 2) - pow(mIS, 2)) + pow(mIS, 4)) +
            (5 * pow(mdm2, 4) - 3 * pow(mFS, 4) +
             6 * pow(mFS, 2) * pow(mIS, 2) + 4 * pow(mdm2, 2) *
             (2 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) -
            2 * (5 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) - 3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           + 2 *
           (-2 * qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 4) + pow(mFS, 4) + pow(mdm2, 2) *
             (4 * pow(mFS, 2) - 2 * pow(mIS, 2)) +
             4 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 4) + pow(mdm2, 2) * pow(mFS, 2) + pow(mFS, 2) *
             pow(mIS, 2)) * pow(qsq, 3) + pow(mdm2, 2) * pow(qsq, 4) +
            pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 6) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2)) +
          pow(FFF0, 2) * pow(mFS - mIS, 2) *
          (-2 * qsq * pow(mdm1, 4) + pow(mdm1, 6) + pow(mdm1, 2) *
           (6 * qsq * pow(mdm2, 2) - 3 * pow(mdm2, 4) + pow(qsq, 2)) +
           2 * pow(-(mdm2 * qsq) + pow(mdm2, 3), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2)))) / 12.)
    J_44 = abs(L_A)**2 * ((
        m23sq * pow(mdm2, -2) * pow(qsq - pow(mFS - mIS, 2), -1) *
        (4 * m23sqsq * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
         (qsq * pow(FFGv, 2) - pow(FFGp, 2) * pow(mFS - mIS, 2)) + 6 * m23sq *
         (-(pow(FFGv, 2) * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) + FFGp *
          (mFS - mIS) * pow(qsq, -1) *
          (FFGp * (mFS - mIS) * (-qsq + pow(mdm1, 2) + 2 * pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFG0 *
           (mFS + mIS) *
           (pow(mdm1, 4) * (qsq - pow(mFS - mIS, 2)) + pow(mdm1, 2) *
            (-(pow(mdm2, 2) * pow(mFS - mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS - mIS, 2)) - pow(qsq, 2)) +
            2 * pow(mdm2, 2) *
            (pow(mdm2, 2) * pow(mFS - mIS, 2) - qsq *
             (pow(mdm2, 2) + pow(mFS - mIS, 2)) + pow(qsq, 2))))) -
         3 * pow(qsq, -2) *
         (2 * pow(FFGv, 2) * pow(qsq, 2) *
          (-2 * pow(mdm1, 6) * pow(mFS, 2) - 2 * qsq * pow(mdm2, 2) *
           (pow(mdm2, 2) * (pow(mFS, 2) - 2 * pow(mIS, 2)) + pow(mFS, 2) *
            (pow(mFS, 2) + pow(mIS, 2))) + pow(mdm2, 4) *
           (pow(mFS, 4) - 4 * pow(mdm2, 2) * pow(mIS, 2) +
            2 * pow(mFS, 2) * pow(mIS, 2) - 3 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
           pow(mdm1, 4) *
           (-2 * qsq * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) -
            (pow(mFS, 2) - pow(mIS, 2)) *
            (2 * pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           2 * pow(mdm1, 2) *
           (-(pow(mdm2, 2) *
              (-3 * pow(mFS, 4) + 5 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
               (2 * pow(mFS, 2) + pow(mIS, 2)) - 2 * pow(mIS, 4))) + qsq *
            (2 * pow(mdm2, 4) - pow(mdm2, 2) *
             (3 * pow(mFS, 2) + 2 * pow(mIS, 2)) + pow(mIS, 4)) -
            (pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) + pow(qsq, 3)) +
           pow(qsq, 4)) + 2 * FFG0 * FFGp * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 6) * (2 * (mFS - mIS) * mIS * qsq -
                           (mFS + mIS) * pow(mFS - mIS, 3) + pow(qsq, 2)) -
           2 * qsq * pow(mdm1, 4) *
           (pow(mFS - mIS, 2) * (pow(mdm2, 2) + pow(mIS, 2)) - qsq *
            (-2 * mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) +
            pow(qsq, 2)) - pow(mdm1, 2) *
           (-3 * (mFS + mIS) * pow(mdm2, 4) * pow(mFS - mIS, 3) + 2 *
            (mFS - mIS) * qsq * pow(mdm2, 2) *
            ((mFS + 2 * mIS) * pow(mdm2, 2) - 2 * mIS * pow(mFS, 2) +
             2 * pow(mFS, 3) - mFS * pow(mIS, 2) + pow(mIS, 3)) +
            (pow(mdm2, 4) - pow(mFS - mIS, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
             (-4 * pow(mFS, 2) + 2 * pow(mIS, 2))) * pow(qsq, 2) + 2 *
            (-(mFS * mIS) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
            pow(qsq, 4)) - 2 * pow(mdm2, 2) *
           (-2 * mFS * (mFS - mIS) * qsq * pow(mdm2, 2) *
            (mFS * (mFS - mIS) + pow(mdm2, 2)) +
            (mFS + mIS) * pow(mdm2, 4) * pow(mFS - mIS, 3) +
            (pow(mdm2, 4) + pow(mFS - mIS, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + 2 * pow(mdm2, 2) *
             (-2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) -
            2 * (-(mFS * mIS) + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) *
            pow(qsq, 3) + pow(qsq, 4))) + pow(FFG0, 2) * pow(mFS + mIS, 2) *
          (-2 * qsq * pow(mdm1, 4) + pow(mdm1, 6) + pow(mdm1, 2) *
           (6 * qsq * pow(mdm2, 2) - 3 * pow(mdm2, 4) + pow(qsq, 2)) +
           2 * pow(-(mdm2 * qsq) + pow(mdm2, 3), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) + pow(FFGp, 2) * pow(mFS - mIS, 2) *
          (-2 * qsq * pow(mdm1, 4) *
           (-2 * qsq * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) -
            (pow(mFS, 2) - pow(mIS, 2)) *
            (2 * pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
           pow(mdm1, 2) *
           (-2 * qsq * pow(mdm2, 2) *
            (-pow(mFS, 4) + pow(mdm2, 2) *
             (pow(mFS, 2) - pow(mIS, 2)) + pow(mIS, 4)) +
            (5 * pow(mdm2, 4) - 3 * pow(mFS, 4) +
             6 * pow(mFS, 2) * pow(mIS, 2) + 4 * pow(mdm2, 2) *
             (2 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) -
            2 * (5 * pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) - 3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           + 2 *
           (-2 * qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 4) + pow(mFS, 4) + pow(mdm2, 2) *
             (4 * pow(mFS, 2) - 2 * pow(mIS, 2)) +
             4 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 4) + pow(mdm2, 2) * pow(mFS, 2) + pow(mFS, 2) *
             pow(mIS, 2)) * pow(qsq, 3) + pow(mdm2, 2) * pow(qsq, 4) +
            pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 6) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2))))) / 12.)
    J_55 = abs(L_V_dXtX)**2 * (-0.16666666666666666 * (
        m23sq * pow(mdm2, -2) * pow(qsq, -2) *
        pow(qsq - pow(mFS + mIS, 2), -1) *
        (2 * m23sqsq * (qsq - 2 * pow(mdm2, 2)) *
         (qsq * pow(FFFv, 2) - pow(FFFp, 2) * pow(mFS + mIS, 2)) * pow(qsq, 2)
         + 3 * m23sq * qsq *
         (-(FFFp * (mFS + mIS) *
            (2 * FFF0 * (mFS - mIS) *
             (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * pow(mdm2, 2) *
             (qsq - pow(mFS + mIS, 2)) - FFFp * (mFS + mIS) *
             (qsq - 2 * pow(mdm2, 2)) *
             (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
              (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
              (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)))) +
          qsq * pow(FFFv, 2) * (qsq - 2 * pow(mdm2, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2))) + 3 *
         (-2 * FFF0 * FFFp *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * pow(mdm2, 2) *
          (pow(mFS, 2) - pow(mIS, 2)) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          pow(FFFv, 2) * pow(qsq, 2) *
          (-2 * qsq * pow(mdm2, 2) *
           (pow(mdm2, 2) * (pow(mFS, 2) - 2 * pow(mIS, 2)) + pow(mFS, 2) *
            (pow(mFS, 2) + pow(mIS, 2))) + pow(mdm2, 4) *
           (pow(mFS, 4) - 4 * pow(mdm2, 2) * pow(mIS, 2) +
            2 * pow(mFS, 2) * pow(mIS, 2) - 3 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           2 * pow(mdm1, 2) *
           (qsq *
            (2 * pow(mdm2, 4) - pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (5 * pow(mFS, 2) + 7 * pow(mIS, 2)) + pow(mIS, 4)) - pow(mdm2, 2) *
            (pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + 3 * pow(mIS, 4)) -
            (4 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3)) + pow(qsq, 4) + pow(mdm1, 4) *
           (-4 * pow(mdm2, 2) * pow(mFS, 2) - 2 * qsq * pow(mIS, 2) +
            pow(qsq, 2) + pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
          pow(FFFp, 2) * pow(mFS + mIS, 2) *
          (-2 * qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
           (pow(mdm2, 4) + pow(mFS, 4) + pow(mdm2, 2) *
            (4 * pow(mFS, 2) - 2 * pow(mIS, 2)) +
            4 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) * pow(qsq, 2) - 2 *
           (pow(mdm2, 4) + pow(mdm2, 2) * pow(mFS, 2) + pow(mFS, 2) *
            pow(mIS, 2)) * pow(qsq, 3) + pow(mdm2, 2) * pow(qsq, 4) +
           pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2) + pow(mdm1, 4) *
           (2 * qsq * pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) +
            (pow(mdm2, 2) - 2 * pow(mFS, 2)) * pow(qsq, 2) + pow(mdm2, 2) *
            pow(pow(mFS, 2) - pow(mIS, 2), 2)) + 2 * pow(mdm1, 2) *
           (qsq * pow(mdm2, 2) *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 4) - pow(mFS, 4) + pow(mFS, 2) * pow(mIS, 2) -
             pow(mdm2, 2) *
             (3 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(mFS, 2) *
            pow(qsq, 3) - pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
          pow(FFF0, 2) * pow(mdm2, 2) * pow(mFS - mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2)))))
    J_66 = abs(L_A_dXtX)**2 * ((
        m23sq * pow(mdm2, -2) * pow(qsq - pow(mFS - mIS, 2), -1) *
        (-2 * m23sqsq * (qsq - 2 * pow(mdm2, 2)) *
         (qsq * pow(FFGv, 2) - pow(FFGp, 2) * pow(mFS - mIS, 2)) + 3 * m23sq *
         (-(FFGp * (mFS - mIS) * pow(qsq, -1) *
            (-2 * FFG0 * (mFS + mIS) *
             (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * pow(mdm2, 2) *
             (qsq - pow(mFS - mIS, 2)) + FFGp * (mFS - mIS) *
             (qsq - 2 * pow(mdm2, 2)) *
             (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
              (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
              (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)))) +
          pow(FFGv, 2) * (qsq - 2 * pow(mdm2, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) -
         3 * pow(qsq, -2) *
         (-2 * FFG0 * FFGp *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * pow(mdm2, 2) *
          (qsq - pow(mFS - mIS, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          pow(FFG0, 2) * pow(mdm2, 2) * pow(mFS + mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) + pow(FFGv, 2) * pow(qsq, 2) *
          (-2 * qsq * pow(mdm2, 2) *
           (pow(mdm2, 2) * (pow(mFS, 2) - 2 * pow(mIS, 2)) + pow(mFS, 2) *
            (pow(mFS, 2) + pow(mIS, 2))) + pow(mdm2, 4) *
           (pow(mFS, 4) - 4 * pow(mdm2, 2) * pow(mIS, 2) +
            2 * pow(mFS, 2) * pow(mIS, 2) - 3 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           2 * pow(mdm1, 2) *
           (qsq *
            (2 * pow(mdm2, 4) - pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (5 * pow(mFS, 2) + 7 * pow(mIS, 2)) + pow(mIS, 4)) - pow(mdm2, 2) *
            (pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + 3 * pow(mIS, 4)) -
            (4 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3)) + pow(qsq, 4) + pow(mdm1, 4) *
           (-4 * pow(mdm2, 2) * pow(mFS, 2) - 2 * qsq * pow(mIS, 2) +
            pow(qsq, 2) + pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
          pow(FFGp, 2) * pow(mFS - mIS, 2) *
          (-2 * qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
           (pow(mdm2, 4) + pow(mFS, 4) + pow(mdm2, 2) *
            (4 * pow(mFS, 2) - 2 * pow(mIS, 2)) +
            4 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) * pow(qsq, 2) - 2 *
           (pow(mdm2, 4) + pow(mdm2, 2) * pow(mFS, 2) + pow(mFS, 2) *
            pow(mIS, 2)) * pow(qsq, 3) + pow(mdm2, 2) * pow(qsq, 4) +
           pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2) + pow(mdm1, 4) *
           (2 * qsq * pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) +
            (pow(mdm2, 2) - 2 * pow(mFS, 2)) * pow(qsq, 2) + pow(mdm2, 2) *
            pow(pow(mFS, 2) - pow(mIS, 2), 2)) + 2 * pow(mdm1, 2) *
           (qsq * pow(mdm2, 2) *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 4) - pow(mFS, 4) + pow(mFS, 2) * pow(mIS, 2) -
             pow(mdm2, 2) *
             (3 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(mFS, 2) *
            pow(qsq, 3) - pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))))))
                               / 6.)
    J_77 = abs(L_T)**2 * ((
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) *
        pow(qsq - pow(mFS - mIS, 2), -1) *
        (4 * m23sqsq *
         (2 * mFS * mIS * qsq * pow(FFhtv, 2) * pow(mdm1, 2) - 2 * mFS * mIS *
          qsq * pow(FFhv, 2) * pow(mdm1, 2) + 2 * mFS * mIS * qsq *
          pow(FFhtv, 2) * pow(mdm2, 2) - 2 * mFS * mIS * qsq * pow(FFhv, 2) *
          pow(mdm2, 2) - qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) - qsq
          * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) - qsq * pow(FFhtv, 2) *
          pow(mdm2, 2) * pow(mFS, 2) - qsq * pow(FFhv, 2) * pow(mdm2, 2) *
          pow(mFS, 2) - qsq * pow(FFhv, 2) * pow(mFS, 4) + pow(FFhtv, 2) *
          pow(mdm1, 2) * pow(mFS, 4) + pow(FFhv, 2) * pow(mdm1, 2) *
          pow(mFS, 4) + pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) +
          pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) + qsq * pow(FFhp, 2) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS - mIS, 2)) - qsq * pow(FFhtv, 2) * pow(mdm1, 2) *
          pow(mIS, 2) - qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 2) -
          qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 2) -
          qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhv, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) -
          qsq * pow(FFhv, 2) * pow(mIS, 4) + pow(FFhtv, 2) * pow(mdm1, 2) *
          pow(mIS, 4) + pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 4) +
          pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 4) +
          pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 4) + qsq * pow(FFhtp, 2) *
          (pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS + mIS, 2)) + 2 * mFS * mIS * pow(FFhv, 2) *
          pow(qsq, 2) + pow(FFhv, 2) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mIS, 2) * pow(qsq, 2)) +
         6 * m23sq * pow(qsq, -1) *
         (-2 * mIS * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 3) +
          2 * mIS * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 3) +
          2 * mIS * qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 3) -
          2 * mIS * qsq * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 3) -
          4 * FFhtv * FFhv * qsq * pow(mdm1, 4) * pow(mFS, 4) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 4) -
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 4) +
          4 * FFhtv * FFhv * qsq * pow(mdm2, 4) * pow(mFS, 4) - 2 * qsq *
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 4) - 2 * qsq * pow(FFhv, 2) *
          pow(mdm2, 4) * pow(mFS, 4) - 2 * FFhtv * FFhv * qsq * pow(mdm1, 2) *
          pow(mFS, 6) - qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 6) +
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 6) - pow(FFhtv, 2) *
          pow(mdm1, 4) * pow(mFS, 6) - pow(FFhv, 2) * pow(mdm1, 4) *
          pow(mFS, 6) + 2 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 6) -
          qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 6) -
          2 * qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 6) - 2 * FFhtv *
          FFhv * pow(mdm2, 4) * pow(mFS, 6) + pow(FFhtv, 2) * pow(mdm2, 4) *
          pow(mFS, 6) + pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 6) +
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 2) +
          4 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 2) *
          pow(mIS, 2) + 4 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) *
          pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) +
          6 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) +
          qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) -
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) -
          6 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) +
          3 * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) +
          3 * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) -
          6 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          4 * qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          6 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) -
          3 * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) -
          3 * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) +
          2 * mFS * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 3) -
          2 * mFS * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 3) - 2 * mFS *
          qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mIS, 3) + 2 * mFS * qsq *
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mIS, 3) + 4 * FFhtv * FFhv * qsq *
          pow(mdm1, 4) * pow(mIS, 4) - 2 * qsq * pow(FFhtv, 2) * pow(mdm1, 4) *
          pow(mIS, 4) - 2 * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 4) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 4) -
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 4) -
          4 * FFhtv * FFhv * qsq * pow(mdm2, 4) * pow(mIS, 4) -
          6 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          4 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          6 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) -
          3 * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) -
          3 * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) +
          6 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) +
          qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) -
          2 * qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) -
          6 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          3 * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          3 * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          2 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mIS, 6) - qsq * pow(
              FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 6) - 2 * qsq * pow(FFhv, 2) *
          pow(mdm1, 2) * pow(mIS, 6) - 2 * FFhtv * FFhv * pow(mdm1, 4) * pow(
              mIS, 6) + pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 6) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 6) - 2 * FFhtv * FFhv * qsq *
          pow(mdm2, 2) * pow(mIS, 6) - qsq * pow(FFhtv, 2) * pow(mdm2, 2) *
          pow(mIS, 6) + 2 * FFhtv * FFhv * pow(mdm2, 4) * pow(mIS, 6) -
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mIS, 6) -
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mIS, 6) - qsq * pow(FFhp, 2) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
          qsq * pow(FFhtp, 2) * (pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
          2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm1, 4) * pow(qsq, 2) +
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm1, 4) * pow(qsq, 2) - 4 * mFS *
          mIS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(qsq, 2) + 4 *
          mFS * mIS * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(qsq, 2)
          - 2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm2, 4) * pow(qsq, 2) +
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm2, 4) * pow(qsq, 2) +
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(
              qsq, 2) + 2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(
                  mFS, 2) * pow(qsq, 2) -
          2 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) -
          2 * mIS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 3) * pow(qsq, 2) -
          2 * mIS * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 3) * pow(qsq, 2) +
          4 * mIS * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 3) * pow(qsq, 2) +
          4 * FFhtv * FFhv * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) -
          4 * FFhtv * FFhv * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          4 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mFS, 6) * pow(qsq, 2) -
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 2) * pow(
              qsq, 2) + 2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) *
          pow(mIS, 2) * pow(qsq, 2) + 2 * FFhtv * FFhv * pow(mdm2, 4) * pow(
              mIS, 2) * pow(qsq, 2) + pow(FFhtv, 2) * pow(mdm2, 4) *
          pow(mIS, 2) * pow(qsq, 2) + pow(FFhv, 2) * pow(mdm2, 4) * pow(
              mIS, 2) * pow(qsq, 2) - 2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(
                  mFS, 2) * pow(mIS, 2) * pow(qsq, 2) - 2 * pow(FFhv, 2) * pow(
                      mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 2) -
          pow(FFhv, 2) * pow(mFS, 4) * pow(mIS, 2) * pow(qsq, 2) -
          2 * mFS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 3) * pow(qsq, 2) +
          4 * mFS * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 3) * pow(qsq, 2) -
          2 * mFS * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 3) * pow(qsq, 2) -
          4 * FFhtv * FFhv * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          4 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          4 * FFhtv * FFhv * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) -
          pow(FFhv, 2) * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mIS, 6) * pow(qsq, 2) +
          2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(qsq, 3) -
          4 * mFS * mIS * pow(FFhv, 2) * pow(mdm1, 2) * pow(qsq, 3) +
          2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm2, 2) * pow(qsq, 3) -
          4 * mFS * mIS * pow(FFhv, 2) * pow(mdm2, 2) * pow(qsq, 3) -
          2 * FFhtv * FFhv * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) -
          pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) -
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) +
          2 * FFhtv * FFhv * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) -
          pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) - 2 * pow(
              FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) - 2 * mIS *
          pow(FFhv, 2) * pow(mFS, 3) * pow(qsq, 3) - 2 * pow(FFhv, 2) * pow(
              mFS, 4) * pow(qsq, 3) + 2 * FFhtv * FFhv * pow(mdm1, 2) * pow(
                  mIS, 2) * pow(qsq, 3) - pow(FFhtv, 2) * pow(mdm1, 2) * pow(
                      mIS, 2) * pow(qsq, 3) - 2 * pow(FFhv, 2) * pow(mdm1, 2) *
          pow(mIS, 2) * pow(qsq, 3) - 2 * FFhtv * FFhv * pow(mdm2, 2) * pow(
              mIS, 2) * pow(qsq, 3) - pow(FFhtv, 2) * pow(mdm2, 2) *
          pow(mIS, 2) * pow(qsq, 3) - 2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(
              mIS, 2) * pow(qsq, 3) - 2 * mFS * pow(FFhv, 2) * pow(mIS, 3) *
          pow(qsq, 3) - 2 * pow(FFhv, 2) * pow(mIS, 4) * pow(qsq, 3) +
          2 * mFS * mIS * pow(FFhv, 2) * pow(qsq, 4) + pow(FFhv, 2) * pow(
              mFS, 2) * pow(qsq, 4) + pow(FFhv, 2) * pow(mIS, 2) * pow(qsq, 4))
         + 6 * pow(qsq, -2) *
         (-2 * FFhtv * FFhv * (pow(mdm1, 2) - pow(mdm2, 2)) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) *
          (-2 * qsq * (pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2) +
           pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
          pow(FFhv, 2) * pow(mFS + mIS, 2) *
          (4 * mIS * qsq * pow(mdm2, 6) * pow(mFS, 3) - 3 * qsq * pow(mdm2, 6)
           * pow(mFS, 4) + 4 * mIS * qsq * pow(mdm2, 4) * pow(mFS, 5) - 2 * mIS
           * pow(mdm2, 6) * pow(mFS, 5) - 2 * qsq * pow(mdm2, 4) * pow(mFS, 6)
           + pow(mdm2, 6) * pow(mFS, 6) - pow(mdm2, 6) * pow(mFS, 4) *
           pow(mIS, 2) - 4 * qsq * pow(mdm2, 4) * pow(mFS, 3) * pow(mIS, 3) +
           4 * pow(mdm2, 6) * pow(mFS, 3) * pow(mIS, 3) - qsq * pow(mdm2, 6) *
           pow(mIS, 4) + 2 * qsq * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) -
           pow(mdm2, 6) * pow(mFS, 2) * pow(mIS, 4) - 2 * mFS * pow(mdm2, 6) *
           pow(mIS, 5) + pow(mdm2, 6) * pow(mIS, 6) - 2 * mFS * mIS *
           pow(mdm2, 6) * pow(qsq, 2) + 3 * pow(mdm2, 6) * pow(mFS, 2) *
           pow(qsq, 2) - 8 * mIS * pow(mdm2, 4) * pow(mFS, 3) * pow(qsq, 2) + 6
           * pow(mdm2, 4) * pow(mFS, 4) * pow(qsq, 2) - 2 * mIS * pow(mdm2, 2)
           * pow(mFS, 5) * pow(qsq, 2) + pow(mdm2, 2) * pow(mFS, 6) *
           pow(qsq, 2) + pow(mdm2, 6) * pow(mIS, 2) * pow(qsq, 2) +
           2 * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 2) +
           3 * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) * pow(qsq, 2) -
           4 * pow(mdm2, 2) * pow(mFS, 3) * pow(mIS, 3) * pow(qsq, 2) +
           pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 2) +
           2 * mFS * pow(mdm2, 2) * pow(mIS, 5) * pow(qsq, 2) - pow(mdm2, 2) *
           pow(mIS, 6) * pow(qsq, 2) + 4 * mFS * mIS * pow(mdm2, 4) *
           pow(qsq, 3) - pow(mdm2, 6) * pow(qsq, 3) - 6 * pow(mdm2, 4) *
           pow(mFS, 2) * pow(qsq, 3) + 4 * mIS * pow(mdm2, 2) * pow(mFS, 3) *
           pow(qsq, 3) - 3 * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 3) -
           2 * pow(mdm2, 4) * pow(mIS, 2) * pow(qsq, 3) - 4 * pow(mdm2, 2) *
           pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 3) - 2 * pow(mFS, 4) *
           pow(mIS, 2) * pow(qsq, 3) + 4 * pow(mFS, 3) * pow(mIS, 3) *
           pow(qsq, 3) + pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 3) -
           2 * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 3) - pow(mdm1, 6) *
           (qsq * (pow(mFS, 4) - 4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4)) -
            pow(mFS - mIS, 4) * pow(mFS + mIS, 2) -
            (-2 * mFS * mIS + pow(mFS, 2) + 3 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3)) - 2 * mFS * mIS * pow(mdm2, 2) * pow(qsq, 4) +
           2 * pow(mdm2, 4) * pow(qsq, 4) + 3 * pow(mdm2, 2) * pow(mFS, 2) *
           pow(qsq, 4) + pow(mdm2, 2) * pow(mIS, 2) * pow(qsq, 4) +
           2 * pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 4) -
           pow(mdm2, 2) * pow(qsq, 5) + pow(mdm1, 4) *
           (qsq - pow(mFS - mIS, 2)) *
           (-2 * qsq * (pow(mdm2, 2) * pow(mFS, 2) +
                        pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) -
            (pow(mdm2, 2) + 4 * pow(mIS, 2)) * pow(qsq, 2) + 2 * pow(qsq, 3) +
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 2) *
           (qsq - pow(mFS - mIS, 2)) *
           (2 * qsq * pow(mdm2, 2) *
            (-pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) +
             2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) +
            (pow(mdm2, 4) - pow(mFS, 4) + 2 * pow(mFS, 2) * pow(mIS, 2) +
             8 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (3 * pow(mdm2, 2) + pow(mIS, 2)) * pow(qsq, 3) + pow(qsq, 4) -
            pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
          pow(FFhtv, 2) * pow(mFS - mIS, 2) *
          (pow(mdm1, 6) *
           (-(qsq * (pow(mFS, 4) + 4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4))) +
            pow(mFS - mIS, 2) * pow(mFS + mIS, 4) +
            (2 * mFS * mIS + pow(mFS, 2) + 3 * pow(mIS, 2)) * pow(qsq, 2) -
            pow(qsq, 3)) + pow(mdm1, 4) * (qsq - pow(mFS + mIS, 2)) *
           (-2 * qsq * (pow(mdm2, 2) * pow(mFS, 2) +
                        pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) -
            (pow(mdm2, 2) + 2 * pow(mFS, 2) + 4 * pow(mIS, 2)) * pow(qsq, 2) +
            2 * pow(qsq, 3) + pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           - pow(mdm1, 2) * (qsq - pow(mFS + mIS, 2)) *
           (2 * qsq * pow(mdm2, 2) *
            (3 * pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) -
             6 * pow(mFS, 2) * pow(mIS, 2) + 3 * pow(mIS, 4)) +
            (pow(mdm2, 4) + pow(mFS, 4) - 10 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
            (4 * pow(mdm2, 2) - 2 *
             (pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 3) + pow(qsq, 4) -
            pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm2, 2) *
           (qsq - pow(mFS + mIS, 2)) *
           (-2 * qsq * pow(mdm2, 2) * pow(mFS, 2) *
            (pow(mdm2, 2) + pow(mFS, 2) - pow(mIS, 2)) +
            (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
             (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) + pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) -
          2 * pow(FFhtp, 2) * pow(qsq, 2) *
          (pow(mdm1, 6) * pow(mFS, 2) *
           (-qsq + pow(mFS + mIS, 2)) - pow(mdm1, 4) *
           (2 * (mFS + mIS) * qsq * pow(mFS, 3) - qsq * pow(mdm2, 2) *
            (2 * mFS * mIS + pow(mFS, 2) + 2 * pow(mIS, 2)) -
            (pow(mFS, 4) - pow(mdm2, 2) * pow(mIS, 2) -
             pow(mFS, 2) * pow(mIS, 2)) * pow(mFS + mIS, 2) +
            (pow(mdm2, 2) - pow(mFS, 2)) * pow(qsq, 2)) +
           pow(mdm2, 2) * pow(mIS, 2) *
           (pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2) +
            qsq * (-2 * mIS * (mFS + mIS) * pow(mdm2, 2) - pow(mdm2, 4) +
                   pow(mFS, 2) * pow(mFS + mIS, 2)) +
            (pow(mdm2, 2) - pow(mFS, 2)) * pow(qsq, 2)) + pow(mdm1, 2) *
           (qsq *
            (pow(mdm2, 4) * (2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) +
             pow(mFS, 2) * pow(mIS, 2) * pow(mFS + mIS, 2) + 2 * pow(mdm2, 2) *
             (-(mFS * mIS) + 2 * pow(mFS, 2) + 2 * pow(mIS, 2)) *
             pow(mFS + mIS, 2)) -
            (pow(mdm2, 4) + pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (4 * mFS * mIS + 5 * pow(mFS, 2) + 5 * pow(mIS, 2))) *
            pow(qsq, 2) + 2 * pow(mdm2, 2) * pow(qsq, 3) -
            pow(mdm2, 2) * pow(mFS + mIS, 2) *
            (pow(mdm2, 2) * pow(mFS, 2) + pow(pow(mFS, 2) - pow(mIS, 2), 2)))))
         - 3 * pow(FFhp, 2) *
         (4 * pow(mdm1, 6) * pow(mFS, 2) *
          (-qsq + pow(mFS - mIS, 2)) + qsq * pow(mdm2, 2) *
          (-4 * pow(mdm2, 4) * pow(mIS, 2) + pow(mdm2, 2) *
           (-4 * mIS * pow(mFS, 3) + 3 * pow(mFS, 4) - 2 * pow(mFS, 2) *
            pow(mIS, 2) + 12 * mFS * pow(mIS, 3) - 9 * pow(mIS, 4)) +
           2 * pow(mFS - mIS, 2) *
           (pow(mFS, 4) + 2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4))) +
          pow(mdm2, 4) * pow(mFS - mIS, 2) *
          (-pow(mFS, 4) + 4 * pow(mdm2, 2) * pow(mIS, 2) -
           2 * pow(mFS, 2) * pow(mIS, 2) + 3 * pow(mIS, 4)) + pow(mdm1, 4) *
          (qsq - pow(mFS - mIS, 2)) *
          (-3 * pow(mFS, 4) + 4 * pow(mdm2, 2) * pow(mIS, 2) +
           2 * pow(mFS, 2) * pow(mIS, 2) - 2 * qsq *
           (2 * pow(mdm2, 2) - 3 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4) +
           pow(qsq, 2)) +
          (pow(mdm2, 4) - 4 * mIS * pow(mFS, 3) + 3 * pow(mFS, 4) +
           6 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
           (-2 * mFS * mIS + 3 * pow(mFS, 2) + pow(mIS, 2)) -
           4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4)) * pow(qsq, 3) -
          (-2 * mFS * mIS + 2 * pow(mdm2, 2) + 3 * pow(mFS, 2) +
           3 * pow(mIS, 2)) * pow(qsq, 4) + pow(qsq, 5) - 2 * pow(mdm1, 2) *
          (qsq *
           (-2 * pow(mdm2, 4) *
            (-2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) +
            pow(mdm2, 2) * pow(mFS - mIS, 2) *
            (6 * mFS * mIS + 5 * pow(mFS, 2) + 5 * pow(mIS, 2)) +
            pow(mFS - mIS, 2) *
            (pow(mFS, 4) - 2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4))) +
           (2 * pow(mdm2, 4) - pow(mFS, 4) + 4 * pow(mFS, 2) * pow(mIS, 2) -
            4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4) -
            pow(mdm2, 2) * pow(mFS + mIS, 2)) * pow(qsq, 2) -
           (-2 * mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + 3 * pow(mIS, 2)) *
           pow(qsq, 3) + pow(qsq, 4) + pow(mdm2, 2) * pow(mFS - mIS, 2) *
           (2 * pow(mdm2, 2) * pow(mFS, 2) -
            3 * pow(pow(mFS, 2) - pow(mIS, 2), 2))) - pow(qsq, 2) *
          (pow(mdm2, 4) *
           (-2 * mFS * mIS + 3 * pow(mFS, 2) - 5 * pow(mIS, 2)) + pow(mdm2, 2) *
           (-8 * mIS * pow(mFS, 3) + 6 * pow(mFS, 4) +
            8 * pow(mFS, 2) * pow(mIS, 2) - 2 * pow(mIS, 4)) + pow(
                -(mIS * pow(mFS, 2)) + pow(mFS, 3) + mFS * pow(mIS, 2) -
                pow(mIS, 3), 2)))) * pow(qsq - pow(mFS + mIS, 2), -1)) / 12.)
    J_88 = abs(L_T5)**2 * ((
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) *
        pow(qsq - pow(mFS - mIS, 2), -1) *
        (4 * m23sqsq *
         (2 * mFS * mIS * qsq * pow(FFhtv, 2) * pow(mdm1, 2) - 2 * mFS * mIS *
          qsq * pow(FFhv, 2) * pow(mdm1, 2) + 2 * mFS * mIS * qsq *
          pow(FFhtv, 2) * pow(mdm2, 2) - 2 * mFS * mIS * qsq * pow(FFhv, 2) *
          pow(mdm2, 2) - qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) - qsq
          * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) - qsq * pow(FFhtv, 2) *
          pow(mdm2, 2) * pow(mFS, 2) - qsq * pow(FFhv, 2) * pow(mdm2, 2) *
          pow(mFS, 2) - qsq * pow(FFhtv, 2) * pow(mFS, 4) + pow(FFhtv, 2) *
          pow(mdm1, 2) * pow(mFS, 4) + pow(FFhv, 2) * pow(mdm1, 2) *
          pow(mFS, 4) + pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) +
          pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) + qsq * pow(FFhp, 2) *
          (pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS - mIS, 2)) - qsq * pow(FFhtv, 2) * pow(mdm1, 2) *
          pow(mIS, 2) - qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 2) -
          qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 2) -
          qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhtv, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) -
          2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) -
          qsq * pow(FFhtv, 2) * pow(mIS, 4) + pow(FFhtv, 2) * pow(mdm1, 2) *
          pow(mIS, 4) + pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 4) +
          pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 4) +
          pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 4) + qsq * pow(FFhtp, 2) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS + mIS, 2)) - 2 * mFS * mIS * pow(FFhtv, 2) *
          pow(qsq, 2) + pow(FFhtv, 2) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mIS, 2) * pow(qsq, 2)) +
         6 * m23sq * pow(qsq, -1) *
         (-2 * mIS * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 3) +
          2 * mIS * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 3) +
          2 * mIS * qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 3) -
          2 * mIS * qsq * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 3) -
          4 * FFhtv * FFhv * qsq * pow(mdm1, 4) * pow(mFS, 4) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 4) -
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 4) +
          4 * FFhtv * FFhv * qsq * pow(mdm2, 4) * pow(mFS, 4) - 2 * qsq *
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 4) - 2 * qsq * pow(FFhv, 2) *
          pow(mdm2, 4) * pow(mFS, 4) - 2 * FFhtv * FFhv * qsq * pow(mdm1, 2) *
          pow(mFS, 6) - qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 6) +
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 6) - pow(FFhtv, 2) *
          pow(mdm1, 4) * pow(mFS, 6) - pow(FFhv, 2) * pow(mdm1, 4) *
          pow(mFS, 6) + 2 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 6) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 6) -
          qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 6) - 2 * FFhtv * FFhv *
          pow(mdm2, 4) * pow(mFS, 6) + pow(FFhtv, 2) * pow(mdm2, 4) *
          pow(mFS, 6) + pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 6) +
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 2) +
          4 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 2) *
          pow(mIS, 2) + 4 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) *
          pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) +
          2 * qsq * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) +
          6 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) +
          qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(mIS, 2) -
          6 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) +
          3 * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) +
          3 * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 4) * pow(mIS, 2) -
          6 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          4 * qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) +
          6 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) -
          3 * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) -
          3 * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 4) * pow(mIS, 2) +
          2 * mFS * qsq * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 3) -
          2 * mFS * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 3) - 2 * mFS *
          qsq * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mIS, 3) + 2 * mFS * qsq *
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mIS, 3) + 4 * FFhtv * FFhv * qsq *
          pow(mdm1, 4) * pow(mIS, 4) - 2 * qsq * pow(FFhtv, 2) * pow(mdm1, 4) *
          pow(mIS, 4) - 2 * qsq * pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 4) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 4) -
          2 * qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 4) -
          4 * FFhtv * FFhv * qsq * pow(mdm2, 4) * pow(mIS, 4) -
          6 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          4 * qsq * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          qsq * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(mIS, 4) +
          6 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) -
          3 * pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) -
          3 * pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(mIS, 4) +
          6 * FFhtv * FFhv * qsq * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) -
          2 * qsq * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) +
          qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) -
          6 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          3 * pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          3 * pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) +
          2 * FFhtv * FFhv * qsq * pow(mdm1, 2) * pow(mIS, 6) - 2 * qsq * pow(
              FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 6) - qsq * pow(FFhv, 2) *
          pow(mdm1, 2) * pow(mIS, 6) - 2 * FFhtv * FFhv * pow(mdm1, 4) * pow(
              mIS, 6) + pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 6) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 6) - 2 * FFhtv * FFhv * qsq *
          pow(mdm2, 2) * pow(mIS, 6) - qsq * pow(FFhv, 2) * pow(mdm2, 2) * pow(
              mIS, 6) + 2 * FFhtv * FFhv * pow(mdm2, 4) * pow(mIS, 6) -
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mIS, 6) -
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mIS, 6) - qsq * pow(FFhp, 2) *
          (pow(mdm1, 2) + pow(mdm2, 2)) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
          qsq * pow(FFhtp, 2) * (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
          (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
          2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm1, 4) * pow(qsq, 2) +
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm1, 4) * pow(qsq, 2) - 4 * mFS *
          mIS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(qsq, 2) + 4 *
          mFS * mIS * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(qsq, 2)
          - 2 * mFS * mIS * pow(FFhtv, 2) * pow(mdm2, 4) * pow(qsq, 2) +
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm2, 4) * pow(qsq, 2) +
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mFS, 2) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(
              qsq, 2) + 2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(
                  mFS, 2) * pow(qsq, 2) -
          2 * FFhtv * FFhv * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm2, 4) * pow(mFS, 2) * pow(qsq, 2) +
          2 * mIS * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 3) * pow(qsq, 2) -
          4 * mIS * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 3) * pow(qsq, 2) +
          2 * mIS * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 3) * pow(qsq, 2) +
          4 * FFhtv * FFhv * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 4) * pow(qsq, 2) -
          4 * FFhtv * FFhv * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          4 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mFS, 6) * pow(qsq, 2) -
          2 * FFhtv * FFhv * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          pow(FFhv, 2) * pow(mdm1, 4) * pow(mIS, 2) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mdm2, 2) * pow(mIS, 2) * pow(
              qsq, 2) + 2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mdm2, 2) *
          pow(mIS, 2) * pow(qsq, 2) + 2 * FFhtv * FFhv * pow(mdm2, 4) * pow(
              mIS, 2) * pow(qsq, 2) + pow(FFhtv, 2) * pow(mdm2, 4) *
          pow(mIS, 2) * pow(qsq, 2) + pow(FFhv, 2) * pow(mdm2, 4) * pow(
              mIS, 2) * pow(qsq, 2) - 2 * pow(FFhtv, 2) * pow(mdm1, 2) *
          pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 2) - 2 * pow(FFhtv, 2) * pow(
              mdm2, 2) * pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 2) -
          pow(FFhtv, 2) * pow(mFS, 4) * pow(mIS, 2) * pow(qsq, 2) -
          4 * mFS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 3) * pow(qsq, 2) +
          2 * mFS * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 3) * pow(qsq, 2) +
          2 * mFS * pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 3) * pow(qsq, 2) -
          4 * FFhtv * FFhv * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          4 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm1, 2) * pow(mIS, 4) * pow(qsq, 2) +
          4 * FFhtv * FFhv * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) +
          2 * pow(FFhv, 2) * pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 2) -
          pow(FFhtv, 2) * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 2) +
          pow(FFhtv, 2) * pow(mIS, 6) * pow(qsq, 2) +
          4 * mFS * mIS * pow(FFhtv, 2) * pow(mdm1, 2) * pow(qsq, 3) -
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm1, 2) * pow(qsq, 3) +
          4 * mFS * mIS * pow(FFhtv, 2) * pow(mdm2, 2) * pow(qsq, 3) -
          2 * mFS * mIS * pow(FFhv, 2) * pow(mdm2, 2) * pow(qsq, 3) -
          2 * FFhtv * FFhv * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) -
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) -
          pow(FFhv, 2) * pow(mdm1, 2) * pow(mFS, 2) * pow(qsq, 3) +
          2 * FFhtv * FFhv * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) -
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) - pow(
              FFhv, 2) * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 3) +
          2 * mIS * pow(FFhtv, 2) * pow(mFS, 3) * pow(qsq, 3) - 2 * pow(
              FFhtv, 2) * pow(mFS, 4) * pow(qsq, 3) + 2 * FFhtv * FFhv * pow(
                  mdm1, 2) * pow(mIS, 2) * pow(qsq, 3) -
          2 * pow(FFhtv, 2) * pow(mdm1, 2) * pow(mIS, 2) * pow(qsq, 3) - pow(
              FFhv, 2) * pow(mdm1, 2) * pow(mIS, 2) * pow(qsq, 3) -
          2 * FFhtv * FFhv * pow(mdm2, 2) * pow(mIS, 2) * pow(qsq, 3) -
          2 * pow(FFhtv, 2) * pow(mdm2, 2) * pow(mIS, 2) * pow(qsq, 3) - pow(
              FFhv, 2) * pow(mdm2, 2) * pow(mIS, 2) * pow(qsq, 3) + 2 * mFS *
          pow(FFhtv, 2) * pow(mIS, 3) * pow(qsq, 3) - 2 * pow(FFhtv, 2) * pow(
              mIS, 4) * pow(qsq, 3) - 2 * mFS * mIS * pow(FFhtv, 2) * pow(
                  qsq, 4) + pow(FFhtv, 2) * pow(mFS, 2) * pow(qsq, 4) +
          pow(FFhtv, 2) * pow(mIS, 2) * pow(qsq, 4)) + 6 * pow(qsq, -2) *
         (-2 * FFhtv * FFhv * (pow(mdm1, 2) - pow(mdm2, 2)) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) *
          (-2 * qsq * (pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2) +
           pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
          pow(FFhtv, 2) * pow(mFS - mIS, 2) *
          (-4 * mIS * qsq * pow(mdm2, 6) * pow(mFS, 3) -
           3 * qsq * pow(mdm2, 6) * pow(mFS, 4) - 4 * mIS * qsq *
           pow(mdm2, 4) * pow(mFS, 5) + 2 * mIS * pow(mdm2, 6) * pow(mFS, 5) -
           2 * qsq * pow(mdm2, 4) * pow(mFS, 6) + pow(mdm2, 6) * pow(mFS, 6) -
           pow(mdm2, 6) * pow(mFS, 4) * pow(mIS, 2) + 4 * qsq * pow(mdm2, 4) *
           pow(mFS, 3) * pow(mIS, 3) - 4 * pow(mdm2, 6) * pow(mFS, 3) *
           pow(mIS, 3) - qsq * pow(mdm2, 6) * pow(mIS, 4) +
           2 * qsq * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 4) - pow(mdm2, 6) *
           pow(mFS, 2) * pow(mIS, 4) + 2 * mFS * pow(mdm2, 6) * pow(mIS, 5) +
           pow(mdm2, 6) * pow(mIS, 6) + 2 * mFS * mIS * pow(mdm2, 6) *
           pow(qsq, 2) + 3 * pow(mdm2, 6) * pow(mFS, 2) * pow(qsq, 2) + 8 *
           mIS * pow(mdm2, 4) * pow(mFS, 3) * pow(qsq, 2) + 6 * pow(mdm2, 4) *
           pow(mFS, 4) * pow(qsq, 2) + 2 * mIS * pow(mdm2, 2) * pow(mFS, 5) *
           pow(qsq, 2) + pow(mdm2, 2) * pow(mFS, 6) * pow(qsq, 2) +
           pow(mdm2, 6) * pow(mIS, 2) * pow(qsq, 2) +
           2 * pow(mdm2, 4) * pow(mFS, 2) * pow(mIS, 2) * pow(qsq, 2) +
           3 * pow(mdm2, 2) * pow(mFS, 4) * pow(mIS, 2) * pow(qsq, 2) +
           4 * pow(mdm2, 2) * pow(mFS, 3) * pow(mIS, 3) * pow(qsq, 2) +
           pow(mdm2, 2) * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 2) -
           2 * mFS * pow(mdm2, 2) * pow(mIS, 5) * pow(qsq, 2) -
           pow(mdm2, 2) * pow(mIS, 6) * pow(qsq, 2) + pow(mdm1, 6) *
           (-(qsq * (pow(mFS, 4) + 4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4))) +
            pow(mFS - mIS, 2) * pow(mFS + mIS, 4) +
            (2 * mFS * mIS + pow(mFS, 2) + 3 * pow(mIS, 2)) * pow(qsq, 2) -
            pow(qsq, 3)) - 4 * mFS * mIS * pow(mdm2, 4) * pow(qsq, 3) -
           pow(mdm2, 6) * pow(qsq, 3) - 6 * pow(mdm2, 4) * pow(mFS, 2) *
           pow(qsq, 3) - 4 * mIS * pow(mdm2, 2) * pow(mFS, 3) * pow(qsq, 3) -
           3 * pow(mdm2, 2) * pow(mFS, 4) * pow(qsq, 3) - 2 * pow(mdm2, 4) *
           pow(mIS, 2) * pow(qsq, 3) - 4 * pow(mdm2, 2) * pow(mFS, 2) *
           pow(mIS, 2) * pow(qsq, 3) - 2 * pow(mFS, 4) * pow(mIS, 2) *
           pow(qsq, 3) - 4 * pow(mFS, 3) * pow(mIS, 3) * pow(qsq, 3) +
           pow(mdm2, 2) * pow(mIS, 4) * pow(qsq, 3) -
           2 * pow(mFS, 2) * pow(mIS, 4) * pow(qsq, 3) + 2 * mFS * mIS *
           pow(mdm2, 2) * pow(qsq, 4) + 2 * pow(mdm2, 4) * pow(qsq, 4) +
           3 * pow(mdm2, 2) * pow(mFS, 2) * pow(qsq, 4) + pow(mdm2, 2) *
           pow(mIS, 2) * pow(qsq, 4) + 2 * pow(mFS, 2) * pow(mIS, 2) *
           pow(qsq, 4) - pow(mdm2, 2) * pow(qsq, 5) + pow(mdm1, 4) *
           (qsq - pow(mFS + mIS, 2)) *
           (-2 * qsq * (pow(mdm2, 2) * pow(mFS, 2) +
                        pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) -
            (pow(mdm2, 2) + 4 * pow(mIS, 2)) * pow(qsq, 2) + 2 * pow(qsq, 3) +
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 2) *
           (qsq - pow(mFS + mIS, 2)) *
           (2 * qsq * pow(mdm2, 2) *
            (-pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) +
             2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) +
            (pow(mdm2, 4) - pow(mFS, 4) + 2 * pow(mFS, 2) * pow(mIS, 2) +
             8 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (3 * pow(mdm2, 2) + pow(mIS, 2)) * pow(qsq, 3) + pow(qsq, 4) -
            pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
          pow(FFhv, 2) * pow(mFS + mIS, 2) *
          (-(pow(mdm1, 6) *
             (qsq * (pow(mFS, 4) - 4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4)) -
              pow(mFS - mIS, 4) * pow(mFS + mIS, 2) -
              (-2 * mFS * mIS + pow(mFS, 2) + 3 * pow(mIS, 2)) * pow(qsq, 2) +
              pow(qsq, 3))) + pow(mdm1, 4) * (qsq - pow(mFS - mIS, 2)) *
           (-2 * qsq * (pow(mdm2, 2) * pow(mFS, 2) +
                        pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) -
            (pow(mdm2, 2) + 2 * pow(mFS, 2) + 4 * pow(mIS, 2)) * pow(qsq, 2) +
            2 * pow(qsq, 3) + pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           - pow(mdm1, 2) * (qsq - pow(mFS - mIS, 2)) *
           (2 * qsq * pow(mdm2, 2) *
            (3 * pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) -
             6 * pow(mFS, 2) * pow(mIS, 2) + 3 * pow(mIS, 4)) +
            (pow(mdm2, 4) + pow(mFS, 4) - 10 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
            (4 * pow(mdm2, 2) - 2 *
             (pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 3) + pow(qsq, 4) -
            pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm2, 2) *
           (qsq - pow(mFS - mIS, 2)) *
           (-2 * qsq * pow(mdm2, 2) * pow(mFS, 2) *
            (pow(mdm2, 2) + pow(mFS, 2) - pow(mIS, 2)) +
            (pow(mdm2, 4) + pow(mFS, 4) + 2 * pow(mdm2, 2) *
             (2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) + pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) -
          2 * pow(FFhp, 2) * pow(qsq, 2) *
          (pow(mdm1, 6) * pow(mFS, 2) *
           (-qsq + pow(mFS - mIS, 2)) + pow(mdm2, 2) * pow(mIS, 2) *
           (qsq *
            (2 *
             (mFS - mIS) * mIS * pow(mdm2, 2) - pow(mdm2, 4) + pow(mFS, 2) *
             pow(mFS - mIS, 2)) + pow(mdm2, 2) * pow(mFS - mIS, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) +
            (pow(mdm2, 2) - pow(mFS, 2)) * pow(qsq, 2)) + pow(mdm1, 4) *
           (2 * (-mFS + mIS) * qsq * pow(mFS, 3) + qsq * pow(mdm2, 2) *
            (-2 * mFS * mIS + pow(mFS, 2) + 2 * pow(mIS, 2)) +
            pow(mFS - mIS, 2) * (pow(mFS, 4) - pow(mdm2, 2) * pow(mIS, 2) -
                                 pow(mFS, 2) * pow(mIS, 2)) +
            (-pow(mdm2, 2) + pow(mFS, 2)) * pow(qsq, 2)) + pow(mdm1, 2) *
           (qsq *
            (pow(mFS, 2) * pow(mFS - mIS, 2) * pow(mIS, 2) + pow(mdm2, 4) *
             (-2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) +
             2 * pow(mdm2, 2) * pow(mFS - mIS, 2) *
             (mFS * mIS + 2 * pow(mFS, 2) + 2 * pow(mIS, 2))) -
            (pow(mdm2, 4) + pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (-4 * mFS * mIS + 5 * pow(mFS, 2) + 5 * pow(mIS, 2))) *
            pow(qsq, 2) + 2 * pow(mdm2, 2) * pow(qsq, 3) -
            pow(mdm2, 2) * pow(mFS - mIS, 2) *
            (pow(mdm2, 2) * pow(mFS, 2) + pow(pow(mFS, 2) - pow(mIS, 2), 2)))))
         - 3 * pow(FFhtp, 2) *
         (pow(mdm2, 4) * (-pow(mFS, 4) + 4 * pow(mdm2, 2) * pow(mIS, 2) -
                          2 * pow(mFS, 2) * pow(mIS, 2) + 3 * pow(mIS, 4)) *
          pow(mFS + mIS, 2) + 4 * pow(mdm1, 6) * pow(mFS, 2) *
          (-qsq + pow(mFS + mIS, 2)) + qsq * pow(mdm2, 2) *
          (-4 * pow(mdm2, 4) * pow(mIS, 2) + pow(mdm2, 2) *
           (4 * mIS * pow(mFS, 3) + 3 * pow(mFS, 4) - 2 * pow(mFS, 2) *
            pow(mIS, 2) - 12 * mFS * pow(mIS, 3) - 9 * pow(mIS, 4)) + 2 *
           (pow(mFS, 4) + 2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) *
           pow(mFS + mIS, 2)) + pow(mdm1, 4) * (qsq - pow(mFS + mIS, 2)) *
          (-3 * pow(mFS, 4) + 4 * pow(mdm2, 2) * pow(mIS, 2) +
           2 * pow(mFS, 2) * pow(mIS, 2) - 2 * qsq *
           (2 * pow(mdm2, 2) - 3 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4) +
           pow(qsq, 2)) +
          (pow(mdm2, 4) + 4 * mIS * pow(mFS, 3) + 3 * pow(mFS, 4) +
           6 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
           (2 * mFS * mIS + 3 * pow(mFS, 2) + pow(mIS, 2)) +
           4 * mFS * pow(mIS, 3) + 3 * pow(mIS, 4)) * pow(qsq, 3) -
          (2 * mFS * mIS + 2 * pow(mdm2, 2) + 3 * pow(mFS, 2) +
           3 * pow(mIS, 2)) * pow(qsq, 4) + pow(qsq, 5) - 2 * pow(mdm1, 2) *
          (qsq *
           (-2 * pow(mdm2, 4) *
            (2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
            (-6 * mFS * mIS + 5 * pow(mFS, 2) + 5 * pow(mIS, 2)) *
            pow(mFS + mIS, 2) +
            (pow(mFS, 4) - 2 * pow(mFS, 2) * pow(mIS, 2) - pow(mIS, 4)) *
            pow(mFS + mIS, 2)) +
           (2 * pow(mdm2, 4) - pow(mFS, 4) - pow(mdm2, 2) * pow(mFS - mIS, 2) +
            4 * pow(mFS, 2) * pow(mIS, 2) + 4 * mFS * pow(mIS, 3) +
            3 * pow(mIS, 4)) * pow(qsq, 2) -
           (2 * mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + 3 * pow(mIS, 2)) *
           pow(qsq, 3) + pow(qsq, 4) + pow(mdm2, 2) * pow(mFS + mIS, 2) *
           (2 * pow(mdm2, 2) * pow(mFS, 2) -
            3 * pow(pow(mFS, 2) - pow(mIS, 2), 2))) - pow(qsq, 2) *
          (pow(mdm2, 4) *
           (2 * mFS * mIS + 3 * pow(mFS, 2) - 5 * pow(mIS, 2)) + pow(mdm2, 2) *
           (8 * mIS * pow(mFS, 3) + 6 * pow(mFS, 4) +
            8 * pow(mFS, 2) * pow(mIS, 2) - 2 * pow(mIS, 4)) + pow(
                mIS * pow(mFS, 2) + pow(mFS, 3) + mFS * pow(mIS, 2) +
                pow(mIS, 3), 2)))) * pow(qsq - pow(mFS + mIS, 2), -1)) / 12.)
    J_99 = abs(L_DV)**2 * ((
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) * pow(qsq, -2) *
        pow(qsq - pow(mFS + mIS, 2), -1) *
        (4 * m23sqsq *
         (qsq * pow(FFFv, 2) - pow(FFFp, 2) * pow(mFS + mIS, 2)) * pow(qsq, 2) *
         (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) +
         6 * m23sq * qsq *
         (FFFp * (mFS + mIS) * pow(mdm1, 4) *
          (FFF0 * (mFS - mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS + mIS, 2)) + FFFp * (mFS + mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) -
          2 * FFFp * (mFS + mIS) * pow(mdm1, 2) * (qsq + pow(mdm2, 2)) *
          (FFF0 * (mFS - mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS + mIS, 2)) + FFFp * (mFS + mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) +
          qsq * pow(FFFv, 2) * pow(mdm1, 4) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) -
          2 * qsq * pow(FFFv, 2) * pow(mdm1, 2) * (qsq + pow(mdm2, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) + FFFp *
          (mFS + mIS) *
          (FFF0 * (mFS - mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS + mIS, 2)) + FFFp * (mFS + mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) *
          pow(qsq - pow(mdm2, 2), 2) + qsq * pow(FFFv, 2) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) *
          pow(qsq - pow(mdm2, 2), 2)) - 3 *
         (-2 * FFF0 * FFFp * pow(mdm1, 4) *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          4 * FFF0 * FFFp * pow(mdm1, 2) *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (qsq + pow(mdm2, 2)) *
          (pow(mFS, 2) - pow(mIS, 2)) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) -
          4 * pow(FFFv, 2) * pow(mdm1, 4) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) + 2 *
            pow(mIS, 4) + 2 * pow(qsq, 2))) + 8 * pow(FFFv, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) +
            2 * pow(mIS, 4) + 2 * pow(qsq, 2))) - 2 * FFF0 * FFFp *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) *
          pow(qsq - pow(mdm2, 2), 2) - 4 * pow(FFFv, 2) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) +
            2 * pow(mIS, 4) + 2 * pow(qsq, 2))) * pow(qsq - pow(mdm2, 2), 2) +
          pow(FFF0, 2) * pow(mdm1, 4) * pow(mFS - mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2) - 2 * pow(FFF0, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(mFS - mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2) +
          pow(FFF0, 2) * pow(mFS - mIS, 2) * pow(qsq - pow(mdm2, 2), 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2) +
          pow(FFFp, 2) * pow(mdm1, 4) * pow(mFS + mIS, 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)) -
          2 * pow(FFFp, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(mFS + mIS, 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)) +
          pow(FFFp, 2) * pow(mFS + mIS, 2) * pow(qsq - pow(mdm2, 2), 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)))))
                           / 48.)
    J_00 = abs(L_DA)**2 * ((
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) * pow(qsq, -2) *
        pow(qsq - pow(mFS - mIS, 2), -1) *
        (4 * m23sqsq *
         (qsq * pow(FFGv, 2) - pow(FFGp, 2) * pow(mFS - mIS, 2)) * pow(qsq, 2) *
         (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) +
         6 * m23sq * qsq *
         (FFGp * (mFS - mIS) * pow(mdm1, 4) *
          (FFG0 * (mFS + mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS - mIS, 2)) + FFGp * (mFS - mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) -
          2 * FFGp * (mFS - mIS) * pow(mdm1, 2) * (qsq + pow(mdm2, 2)) *
          (FFG0 * (mFS + mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS - mIS, 2)) + FFGp * (mFS - mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) +
          qsq * pow(FFGv, 2) * pow(mdm1, 4) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) -
          2 * qsq * pow(FFGv, 2) * pow(mdm1, 2) * (qsq + pow(mdm2, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) + FFGp *
          (mFS - mIS) *
          (FFG0 * (mFS + mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (qsq - pow(mFS - mIS, 2)) + FFGp * (mFS - mIS) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) *
          pow(qsq - pow(mdm2, 2), 2) + qsq * pow(FFGv, 2) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) *
          pow(qsq - pow(mdm2, 2), 2)) - 3 *
         (-2 * FFG0 * FFGp * pow(mdm1, 4) *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          4 * FFG0 * FFGp * pow(mdm1, 2) *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (qsq + pow(mdm2, 2)) *
          (qsq - pow(mFS - mIS, 2)) * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) -
          4 * pow(FFGv, 2) * pow(mdm1, 4) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) + 2 *
            pow(mIS, 4) + 2 * pow(qsq, 2))) + 8 * pow(FFGv, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) +
            2 * pow(mIS, 4) + 2 * pow(qsq, 2))) - 2 * FFG0 * FFGp *
          (qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) *
          pow(qsq - pow(mdm2, 2), 2) - 4 * pow(FFGv, 2) * pow(qsq, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + 4 * pow(mIS, 2)) +
            2 * pow(mIS, 4) + 2 * pow(qsq, 2))) * pow(qsq - pow(mdm2, 2), 2) +
          pow(FFG0, 2) * pow(mdm1, 4) * pow(mFS + mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) - 2 * pow(FFG0, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(mFS + mIS, 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) +
          pow(FFG0, 2) * pow(mFS + mIS, 2) * pow(qsq - pow(mdm2, 2), 2) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) +
          pow(FFGp, 2) * pow(mdm1, 4) * pow(mFS - mIS, 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)) -
          2 * pow(FFGp, 2) * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) * pow(mFS - mIS, 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)) +
          pow(FFGp, 2) * pow(mFS - mIS, 2) * pow(qsq - pow(mdm2, 2), 2) *
          (2 * pow(mdm1, 2) *
           (qsq *
            (3 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mIS, 4)) +
            (pow(mdm2, 2) - 2 *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) -
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 4) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2) + pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)))))
                           / 48.)
    J_13 = 2 * (L_S * L_V.conjugate()).imag * (
        -0.25 *
        (FFF0 * m23sq * pow(mdm2, -2) * pow(mfq - miq, -1) * pow(qsq, -1) *
         (FFFp * m23sq * qsq * (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
          (pow(mFS, 2) - pow(mIS, 2)) - FFFp * mFS * (mFS + mIS) *
          (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
          FFFp * mIS * (mFS + mIS) * (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
          FFF0 * mFS * (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) * (qsq - 2 * pow(mdm2, 2)) +
           4 * qsq * pow(mdm2, 2) - 5 * pow(mdm2, 4) + pow(qsq, 2)) + FFF0 *
          (mFS - mIS) * mIS * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) * (qsq - 2 * pow(mdm2, 2)) +
           4 * qsq * pow(mdm2, 2) - 5 * pow(mdm2, 4) + pow(qsq, 2)))))
    J_17 = 2 * (L_S * L_T.conjugate()).imag * (
        -0.25 *
        (FFF0 * FFhp * m23sq * (mFS - mIS) * pow(mdm1, -2) * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * pow(mfq - miq, -1) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))))
    J_19 = 2 * (L_S * L_DV.conjugate()).imag * (
        -0.125 *
        (FFF0 * m23sq * pow(mdm1, -2) * pow(mdm2, -2) * pow(mfq - miq, -1) *
         (FFFp * m23sq * (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 4) - pow(qsq - pow(mdm2, 2), 2)) - (mFS - mIS) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * pow(qsq, -1) *
          (FFFp * (mFS + mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFF0 *
           (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
           (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
            (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2))))))
    J_24 = 2 * (L_P * L_A.conjugate()).real * (
        (FFG0 * m23sq * pow(mdm2, -2) * pow(mfq + miq, -1) *
         (-(FFGp * m23sq * (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
            (pow(mFS, 2) - pow(mIS, 2))) + pow(qsq, -1) *
          (FFGp * mFS * (mFS - mIS) *
           (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFGp *
           (mFS - mIS) * mIS * (-qsq + pow(mdm1, 2) + 5 * pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
           FFG0 * mFS * (mFS + mIS) * (qsq - pow(mFS - mIS, 2)) *
           (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
            (qsq - 2 * pow(mdm2, 2)) + 4 * qsq * pow(mdm2, 2) -
            5 * pow(mdm2, 4) + pow(qsq, 2)) + FFG0 * mIS * (mFS + mIS) *
           (qsq - pow(mFS - mIS, 2)) *
           (pow(mdm1, 4) - 2 * pow(mdm1, 2) * (qsq - 2 * pow(mdm2, 2)) +
            4 * qsq * pow(mdm2, 2) - 5 * pow(mdm2, 4) + pow(qsq, 2))))) / 4.)
    J_28 = 2 * (L_P * L_T5.conjugate()).imag * (
        (FFG0 * FFhtp * m23sq * (mFS + mIS) * pow(mdm1, -2) * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * pow(mfq + miq, -1) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) /
        4.)
    J_20 = 2 * (L_P * L_DA.conjugate()).real * (
        (FFG0 * m23sq * pow(mdm1, -2) * pow(mdm2, -2) * pow(mfq + miq, -1) *
         (-(FFGp * m23sq * (pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mdm1, 4) - pow(qsq - pow(mdm2, 2), 2))) + (mFS + mIS) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * pow(qsq, -1) *
          (FFGp * (mFS - mIS) * (qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFG0 *
           (mFS + mIS) * (qsq - pow(mFS - mIS, 2)) *
           (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
            (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2))))) / 8.)
    J_36 = 2 * (L_V * L_A_dXtX.conjugate()).imag * (
        -0.5 *
        (FFFv * FFGv * m23sq * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))))
    J_37 = 2 * (L_V * L_T.conjugate()).real * ((m23sq * pow(mdm2, -2) * (
        (FFF0 * FFhp + 2 * FFFv * FFhtv) * m23sq * (mFS - mIS) *
        (qsq - pow(mdm1, 2) + pow(mdm2, 2)) + pow(qsq, -1) *
        (FFF0 * FFhp * (mFS - mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
         FFFp * FFhp * (mFS + mIS) * (qsq - pow(mFS - mIS, 2)) *
         (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
          (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) + 2 * FFFv *
         (FFhtv * (mFS - mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFhv *
          (mFS + mIS) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)))))) / 4.)
    J_39 = 2 * (L_V * L_DV.conjugate()).real * ((
        m23sq * pow(mdm2, -2) * pow(qsq - pow(mFS + mIS, 2), -1) *
        (-4 * m23sqsq * (qsq - pow(mdm1, 2) + pow(mdm2, 2)) *
         (qsq * pow(FFFv, 2) - pow(FFFp, 2) * pow(mFS + mIS, 2)) -
         6 * m23sq * pow(FFFv, 2) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
         6 * FFFp * m23sq * (mFS + mIS) * pow(qsq, -1) *
         (-(FFFp * (mFS + mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) + FFF0 *
          (mFS - mIS) *
          (pow(mdm1, 4) * (-qsq + pow(mFS + mIS, 2)) + pow(mdm1, 2) *
           (qsq * (2 * pow(mdm2, 2) - pow(mFS + mIS, 2)) - 2 * pow(mdm2, 2) *
            pow(mFS + mIS, 2) + pow(qsq, 2)) + pow(mdm2, 2) *
           (pow(mdm2, 2) * pow(mFS + mIS, 2) - qsq *
            (pow(mdm2, 2) + pow(mFS + mIS, 2)) + pow(qsq, 2)))) -
         3 * pow(qsq, -2) *
         (4 * pow(FFFv, 2) * pow(qsq, 2) *
          (-(pow(mdm1, 6) * pow(mFS, 2)) - qsq * pow(mdm2, 2) *
           (2 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) + pow(mdm2, 4) *
           (pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) -
            3 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) - pow(mFS, 2) * pow(mIS, 2) +
            pow(mdm2, 2) *
            (4 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
           pow(mdm1, 4) *
           (-(pow(mFS, 2) * pow(mIS, 2)) + pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 2 * pow(mIS, 2)) + pow(mIS, 4) + pow(qsq, 2)) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           pow(mdm1, 2) *
           (-(qsq * (pow(mdm2, 4) - pow(mFS, 4) +
                     2 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
                     (pow(mFS, 2) + 2 * pow(mIS, 2)) - 2 * pow(mIS, 4))) +
            pow(mdm2, 2) *
            (pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (pow(mFS, 2) + 2 * pow(mIS, 2)) + 3 * pow(mIS, 4)) +
            (pow(mdm2, 2) - 3 * pow(mFS, 2) - 4 * pow(mIS, 2)) * pow(qsq, 2) +
            2 * pow(qsq, 3)) + pow(qsq, 4)) + 2 * FFF0 * FFFp *
          (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 6) *
           (-2 * mIS * (mFS + mIS) * qsq -
            (mFS - mIS) * pow(mFS + mIS, 3) + pow(qsq, 2)) - pow(mdm1, 4) *
           (qsq - pow(mFS + mIS, 2)) *
           (qsq * (pow(mdm2, 2) - 2 * pow(mIS, 2)) + 3 * pow(mdm2, 2) *
            (pow(mFS, 2) - pow(mIS, 2)) + 2 * pow(qsq, 2)) + pow(mdm1, 2) *
           (2 * (mFS + mIS) * qsq * pow(mdm2, 2) *
            ((2 * mFS - mIS) * pow(mdm2, 2) + mIS * pow(mFS, 2) + pow(mFS, 3) +
             mFS * pow(mIS, 2) + pow(mIS, 3)) - 3 *
            (mFS - mIS) * pow(mdm2, 4) * pow(mFS + mIS, 3) +
            (-pow(mdm2, 4) - 2 * pow(mdm2, 2) * (pow(mFS, 2) + pow(mIS, 2)) +
             (pow(mFS, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2)) * pow(qsq, 2) - 2 *
            (mFS * mIS + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4)) + pow(mdm2, 2) *
           (-2 * mFS * (mFS + mIS) * qsq * pow(mdm2, 2) *
            (mFS * (mFS + mIS) + pow(mdm2, 2)) +
            (mFS - mIS) * pow(mdm2, 4) * pow(mFS + mIS, 3) +
            (pow(mdm2, 4) + 2 * pow(mdm2, 2) *
             (2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2)) +
             (pow(mFS, 2) + pow(mIS, 2)) * pow(mFS + mIS, 2)) * pow(qsq, 2) - 2 *
            (mFS * mIS + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) *
            pow(qsq, 3) + pow(qsq, 4))) + pow(FFFp, 2) * pow(mFS + mIS, 2) *
          (qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
           (2 * pow(mdm2, 2) + 3 * pow(mFS, 2) + pow(mIS, 2)) - pow(mdm2, 2) *
           (pow(mdm2, 4) + 3 * pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) +
            6 * pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) + 7 * pow(mIS, 4)) * pow(qsq, 2) +
           (3 * pow(mdm2, 4) + pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) +
            2 * pow(mdm2, 2) *
            (3 * pow(mFS, 2) + 5 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 3) -
           (3 * pow(mdm2, 2) + 2 *
            (pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 4) + pow(qsq, 5) -
           pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2) - pow(mdm1, 4) *
           (qsq * (2 * pow(mdm2, 2) - 3 * pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mFS, 2) - pow(mIS, 2)) -
            (pow(mdm2, 2) - 6 * pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3) + 3 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           - pow(mdm1, 2) *
           (2 * qsq * pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + pow(mIS, 2)) +
            (pow(mdm2, 4) + 5 * pow(mFS, 4) - 10 * pow(mFS, 2) * pow(mIS, 2) -
             4 * pow(mdm2, 2) *
             (3 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
            (6 * pow(mdm2, 2) - 2 *
             (3 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 3) + pow(qsq, 4) -
            3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 6) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2)) +
          pow(FFF0, 2) * pow(mFS - mIS, 2) *
          (pow(mdm1, 6) - 3 * pow(mdm1, 4) *
           (qsq + pow(mdm2, 2)) + pow(mdm1, 2) *
           (2 * qsq * pow(mdm2, 2) + 3 * pow(mdm2, 4) + 3 * pow(qsq, 2)) -
           (qsq + pow(mdm2, 2)) * pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS + mIS, 2), 2)))) / 24.)
    J_45 = 2 * (L_A * L_V_dXtX.conjugate()).imag * (
        -0.5 *
        (FFFv * FFGv * m23sq * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))))
    J_48 = 2 * (L_A * L_T5.conjugate()).imag * (-0.25 * (
        m23sq * pow(mdm2, -2) *
        ((FFG0 * FFhtp + 2 * FFGv * FFhv) * m23sq * (mFS + mIS) *
         (qsq - pow(mdm1, 2) + pow(mdm2, 2)) + pow(qsq, -1) *
         (FFG0 * FFhtp * (mFS + mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
          FFGp * FFhtp * (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) + 2 * FFGv *
          (FFhv * (mFS + mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFhtv *
           (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
           (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
            (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)))))))
    J_40 = 2 * (L_A * L_DA.conjugate()).real * ((
        m23sq * pow(mdm2, -2) * pow(qsq - pow(mFS - mIS, 2), -1) *
        (-4 * m23sqsq * (qsq - pow(mdm1, 2) + pow(mdm2, 2)) *
         (qsq * pow(FFGv, 2) - pow(FFGp, 2) * pow(mFS - mIS, 2)) -
         6 * m23sq * pow(FFGv, 2) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) -
         6 * FFGp * m23sq * (mFS - mIS) * pow(qsq, -1) *
         (-(FFGp * (mFS - mIS) * (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) + FFG0 *
          (mFS + mIS) *
          (pow(mdm1, 4) * (-qsq + pow(mFS - mIS, 2)) + pow(mdm1, 2) *
           (qsq * (2 * pow(mdm2, 2) - pow(mFS - mIS, 2)) - 2 * pow(mdm2, 2) *
            pow(mFS - mIS, 2) + pow(qsq, 2)) + pow(mdm2, 2) *
           (pow(mdm2, 2) * pow(mFS - mIS, 2) - qsq *
            (pow(mdm2, 2) + pow(mFS - mIS, 2)) + pow(qsq, 2)))) -
         3 * pow(qsq, -2) *
         (4 * pow(FFGv, 2) * pow(qsq, 2) *
          (-(pow(mdm1, 6) * pow(mFS, 2)) - qsq * pow(mdm2, 2) *
           (2 * pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) + pow(mdm2, 4) *
           (pow(mFS, 4) + pow(mdm2, 2) * pow(mIS, 2) -
            3 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mIS, 4)) +
           (pow(mdm2, 4) + pow(mFS, 4) - pow(mFS, 2) * pow(mIS, 2) +
            pow(mdm2, 2) *
            (4 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
           pow(mdm1, 4) *
           (-(pow(mFS, 2) * pow(mIS, 2)) + pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + 2 * pow(mIS, 2)) + pow(mIS, 4) + pow(qsq, 2)) - 2 *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) -
           pow(mdm1, 2) *
           (-(qsq * (pow(mdm2, 4) - pow(mFS, 4) +
                     2 * pow(mFS, 2) * pow(mIS, 2) + 2 * pow(mdm2, 2) *
                     (pow(mFS, 2) + 2 * pow(mIS, 2)) - 2 * pow(mIS, 4))) +
            pow(mdm2, 2) *
            (pow(mFS, 4) - 4 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (pow(mFS, 2) + 2 * pow(mIS, 2)) + 3 * pow(mIS, 4)) +
            (pow(mdm2, 2) - 3 * pow(mFS, 2) - 4 * pow(mIS, 2)) * pow(qsq, 2) +
            2 * pow(qsq, 3)) + pow(qsq, 4)) + 2 * FFG0 * FFGp *
          (pow(mFS, 2) - pow(mIS, 2)) *
          (pow(mdm1, 6) *
           (2 * (mFS - mIS) * mIS * qsq -
            (mFS + mIS) * pow(mFS - mIS, 3) + pow(qsq, 2)) - pow(mdm1, 4) *
           (qsq - pow(mFS - mIS, 2)) *
           (qsq * (pow(mdm2, 2) - 2 * pow(mIS, 2)) + 3 * pow(mdm2, 2) *
            (pow(mFS, 2) - pow(mIS, 2)) + 2 * pow(qsq, 2)) + pow(mdm1, 2) *
           (-3 * (mFS + mIS) * pow(mdm2, 4) * pow(mFS - mIS, 3) + 2 *
            (mFS - mIS) * qsq * pow(mdm2, 2) *
            ((2 * mFS + mIS) * pow(mdm2, 2) - mIS * pow(mFS, 2) + pow(mFS, 3) +
             mFS * pow(mIS, 2) - pow(mIS, 3)) +
            (-pow(mdm2, 4) - 2 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mFS - mIS, 2) *
             (pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) - 2 *
            (-(mFS * mIS) + pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4)) + pow(mdm2, 2) *
           (-2 * mFS * (mFS - mIS) * qsq * pow(mdm2, 2) *
            (mFS * (mFS - mIS) + pow(mdm2, 2)) +
            (mFS + mIS) * pow(mdm2, 4) * pow(mFS - mIS, 3) +
            (pow(mdm2, 4) + pow(mFS - mIS, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + 2 * pow(mdm2, 2) *
             (-2 * mFS * mIS + 2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) -
            2 * (-(mFS * mIS) + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) *
            pow(qsq, 3) + pow(qsq, 4))) + pow(FFG0, 2) * pow(mFS + mIS, 2) *
          (pow(mdm1, 6) - 3 * pow(mdm1, 4) *
           (qsq + pow(mdm2, 2)) + pow(mdm1, 2) *
           (2 * qsq * pow(mdm2, 2) + 3 * pow(mdm2, 4) + 3 * pow(qsq, 2)) -
           (qsq + pow(mdm2, 2)) * pow(qsq - pow(mdm2, 2), 2)) *
          pow(qsq - pow(mFS - mIS, 2), 2) + pow(FFGp, 2) * pow(mFS - mIS, 2) *
          (qsq * pow(mdm2, 4) * (pow(mFS, 2) - pow(mIS, 2)) *
           (2 * pow(mdm2, 2) + 3 * pow(mFS, 2) + pow(mIS, 2)) - pow(mdm2, 2) *
           (pow(mdm2, 4) + 3 * pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) +
            6 * pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)) + 7 * pow(mIS, 4)) * pow(qsq, 2) +
           (3 * pow(mdm2, 4) + pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) +
            2 * pow(mdm2, 2) *
            (3 * pow(mFS, 2) + 5 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 3) -
           (3 * pow(mdm2, 2) + 2 *
            (pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 4) + pow(qsq, 5) -
           pow(mdm2, 6) * pow(pow(mFS, 2) - pow(mIS, 2), 2) - pow(mdm1, 4) *
           (qsq * (2 * pow(mdm2, 2) - 3 * pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mFS, 2) - pow(mIS, 2)) -
            (pow(mdm2, 2) - 6 * pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3) + 3 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           - pow(mdm1, 2) *
           (2 * qsq * pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) *
            (pow(mdm2, 2) + 3 * pow(mFS, 2) + pow(mIS, 2)) +
            (pow(mdm2, 4) + 5 * pow(mFS, 4) - 10 * pow(mFS, 2) * pow(mIS, 2) -
             4 * pow(mdm2, 2) *
             (3 * pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
            (6 * pow(mdm2, 2) - 2 *
             (3 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 3) + pow(qsq, 4) -
            3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
           pow(mdm1, 6) * pow(qsq + pow(mFS, 2) - pow(mIS, 2), 2))))) / 24.)
    J_58 = 2 * (L_V_dXtX * L_T5.conjugate()).real * (
        (m23sq *
         (2 * (FFFp * FFhp - FFFv * FFhv) * m23sqsq *
          (mFS + mIS) * qsq * pow(mdm2, -2) + 3 * m23sq *
          (-2 * FFF0 * FFhp * (mFS - mIS) *
           (-qsq + pow(mFS + mIS, 2)) + FFFv * pow(mdm2, -2) *
           (FFhtv * (mFS - mIS) * (-qsq + pow(mdm1, 2) + 3 * pow(mdm2, 2)) *
            (qsq - pow(mFS + mIS, 2)) + FFhv * (mFS + mIS) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) -
           FFFp * FFhp * (mFS + mIS) * pow(mdm2, -2) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) +
          3 * pow(qsq, -1) *
          (2 * FFF0 * FFhp * (mFS - mIS) * (-qsq + pow(mFS + mIS, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
           FFFv * FFhtv * (mFS - mIS) * pow(mdm2, -2) *
           (-qsq + pow(mdm1, 2) + 3 * pow(mdm2, 2)) *
           (-qsq + pow(mFS + mIS, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) +
           2 * FFFp * FFhp * (mFS + mIS) * pow(mdm2, -2) *
           (qsq * (pow(mdm1, 2) - pow(mdm2, 2)) *
            (pow(mdm1, 2) * pow(mFS, 2) + pow(mFS, 4) -
             pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (2 * pow(mFS, 2) + pow(mIS, 2))) +
            (pow(mdm2, 4) + pow(mFS, 2) *
             (-pow(mdm1, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
             (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) -
            pow(mdm2, 2) * pow(qsq, 3) + pow(mdm2, 2) *
            (-pow(mdm1, 2) + pow(mdm2, 2)) * pow(pow(mFS, 2) - pow(mIS, 2), 2))
           - FFFv * FFhv * (mFS + mIS) * pow(mdm2, -2) *
           (2 * qsq * pow(mdm2, 2) *
            (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
             (3 * pow(mFS, 2) + 4 * pow(mIS, 2)) + 2 * pow(mIS, 4)) +
            (-3 * pow(mdm2, 4) + pow(mFS, 4) - 2 * pow(mdm2, 2) *
             (2 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
            2 * (pow(mdm2, 2) - pow(mFS, 2) - pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) - 3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2)
            + pow(mdm1, 4) *
            (-2 * qsq * pow(mIS, 2) + pow(qsq, 2) +
             pow(pow(mFS, 2) - pow(mIS, 2), 2)) - 2 * pow(mdm1, 2) *
            (qsq * (-(pow(mFS, 2) * pow(mIS, 2)) + 3 * pow(mdm2, 2) *
                    (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) -
             (2 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
             pow(qsq, 3) - pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)))))
         * pow(qsq - pow(mFS + mIS, 2), -1)) / 6.)
    J_50 = 2 * (L_V_dXtX * L_DA.conjugate()).imag * (
        (FFFv * FFGv * m23sq * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) /
        2.)
    J_67 = 2 * (L_A_dXtX * L_T.conjugate()).imag * (
        (m23sq * pow(mdm2, -2) * pow(qsq, -1) *
         pow(qsq - pow(mFS - mIS, 2), -1) *
         (2 * (FFGp * FFhtp - FFGv * FFhtv) * m23sqsq *
          (mFS - mIS) * pow(qsq, 2) + 6 * FFG0 * FFhtp *
          (mFS + mIS) * pow(mdm2, 2) * (qsq - pow(mFS - mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          3 * FFGv * FFhv * (mFS + mIS) *
          (-qsq + pow(mdm1, 2) + 3 * pow(mdm2, 2)) *
          (qsq - pow(mFS - mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          3 * m23sq * qsq *
          (2 * FFG0 * FFhtp * (mFS + mIS) * pow(mdm2, 2) *
           (qsq - pow(mFS - mIS, 2)) + FFGv *
           (FFhv * (mFS + mIS) * (-qsq + pow(mdm1, 2) + 3 * pow(mdm2, 2)) *
            (qsq - pow(mFS - mIS, 2)) + FFhtv * (mFS - mIS) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) +
           FFGp * FFhtp * (mFS - mIS) *
           (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
            (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2))) -
          6 * FFGp * FFhtp * (mFS - mIS) *
          (-(qsq * (pow(mdm1, 2) - pow(mdm2, 2)) *
             (pow(mdm1, 2) * pow(mFS, 2) + pow(mFS, 4) -
              pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
              (2 * pow(mFS, 2) + pow(mIS, 2)))) -
           (pow(mdm2, 4) + pow(mFS, 2) *
            (-pow(mdm1, 2) + pow(mIS, 2)) + pow(mdm2, 2) *
            (2 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) +
           pow(mdm2, 2) * pow(qsq, 3) +
           (pow(mdm1, 2) - pow(mdm2, 2)) * pow(mdm2, 2) *
           pow(pow(mFS, 2) - pow(mIS, 2), 2)) - 3 * FFGv * FFhtv *
          (mFS - mIS) *
          (2 * qsq * pow(mdm2, 2) *
           (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (3 * pow(mFS, 2) + 4 * pow(mIS, 2)) + 2 * pow(mIS, 4)) +
           (-3 * pow(mdm2, 4) + pow(mFS, 4) - 2 * pow(mdm2, 2) *
            (2 * pow(mFS, 2) + 3 * pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) +
           2 * (pow(mdm2, 2) - pow(mFS, 2) - pow(mIS, 2)) * pow(qsq, 3) +
           pow(qsq, 4) - 3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2) +
           pow(mdm1, 4) *
           (-2 * qsq * pow(mIS, 2) + pow(qsq, 2) +
            pow(pow(mFS, 2) - pow(mIS, 2), 2)) - 2 * pow(mdm1, 2) *
           (qsq * (-(pow(mFS, 2) * pow(mIS, 2)) + 3 * pow(mdm2, 2) *
                   (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) -
            (2 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 2) +
            pow(qsq, 3) - pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)))))
        / 6.)
    J_69 = 2 * (L_A_dXtX * L_DV.conjugate()).imag * (
        (FFFv * FFGv * m23sq * pow(mdm2, -2) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
         (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
          (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
          (-m23sq + pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) /
        2.)
    J_79 = 2 * (L_T * L_DV.conjugate()).real * (-0.041666666666666664 * (
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) * pow(qsq, -1) *
        pow(qsq - pow(mFS + mIS, 2), -1) *
        (4 * (FFFp * FFhp - FFFv * FFhv) * m23sqsq * (mFS + mIS) *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) * pow(qsq, 2) + 3 * m23sq * qsq *
         (4 * FFFv * FFhtv * (mFS - mIS) * pow(mdm1, 2) *
          (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
          (qsq - pow(mFS + mIS, 2)) + 2 * FFFp * FFhp * (mFS + mIS) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) -
          2 * FFFv * FFhv * (mFS + mIS) *
          (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) +
          FFF0 * FFhp * (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2))) + 3 *
         (FFF0 * FFhp * (mFS - mIS) * (qsq - pow(mFS + mIS, 2)) *
          (pow(mdm2, 2) * (pow(mFS, 2) - pow(mIS, 2)) - pow(mdm1, 2) *
           (qsq + pow(mFS, 2) - pow(mIS, 2)) - qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) + 4 * FFFv *
          (FFhtv * (mFS - mIS) * pow(mdm1, 2) *
           (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) * (-qsq + pow(mFS + mIS, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) - FFhv *
           (mFS + mIS) *
           (qsq * pow(mIS, 2) *
            (-(qsq * pow(mdm2, 2) *
               (2 * pow(mdm2, 2) - 2 * pow(mFS, 2) + pow(mIS, 2))) +
             pow(mdm2, 4) * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) +
             (pow(mdm2, 2) - pow(mFS, 2)) * pow(qsq, 2)) + pow(mdm1, 6) *
            (-(qsq * (pow(mFS, 2) + 2 * pow(mIS, 2))) + pow(qsq, 2) +
             pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 4) *
            (qsq *
             (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) - pow(mdm2, 2) *
              (4 * pow(mFS, 2) + 3 * pow(mIS, 2)) + 2 * pow(mIS, 4)) +
             (pow(mdm2, 2) - 2 *
              (pow(mFS, 2) + 2 * pow(mIS, 2))) * pow(qsq, 2) + 2 * pow(qsq, 3)
             + 2 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
            pow(mdm1, 2) *
            ((2 * pow(mdm2, 4) + 4 * pow(mdm2, 2) *
              (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) -
             (3 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 3) +
             pow(qsq, 4) + pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2) -
             qsq * pow(mdm2, 2) *
             (pow(mdm2, 2) * (3 * pow(mFS, 2) + 2 * pow(mIS, 2)) +
              pow(pow(mFS, 2) - pow(mIS, 2), 2))))) - FFFp * FFhp *
          (mFS + mIS) *
          (pow(mdm1, 6) *
           (-2 * qsq * (3 * pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2) +
            pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 4) *
           (qsq * (5 * pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) -
                   2 * pow(mdm2, 2) *
                   (3 * pow(mFS, 2) + 5 * pow(mIS, 2)) + pow(mIS, 4)) +
            (7 * pow(mdm2, 2) - 2 *
             (5 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) + 3 *
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 2) *
           ((pow(mdm2, 4) - 3 * pow(mFS, 4) + 6 * pow(mFS, 2) * pow(mIS, 2) -
             4 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) -
            3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2) + 2 * qsq *
            (pow(mdm2, 4) * (pow(mFS, 2) + 3 * pow(mIS, 2)) +
             3 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
           (qsq - pow(mdm2, 2)) * pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)))))
                                                )
    J_80 = 2 * (L_T5 * L_DA.conjugate()).imag * (-0.041666666666666664 * (
        m23sq * pow(mdm1, -2) * pow(mdm2, -2) *
        (4 * (FFGp * FFhtp - FFGv * FFhtv) * m23sqsq * (mFS - mIS) * qsq *
         (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
         pow(qsq - pow(mFS - mIS, 2), -1) + 3 * m23sq *
         (FFG0 * FFhtp * (mFS + mIS) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) - 2 *
          (FFGv *
           (-2 * FFhv * (mFS + mIS) * pow(mdm1, 2) *
            (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
            (qsq - pow(mFS - mIS, 2)) - FFhtv * (mFS - mIS) *
            (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
            (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) +
           FFGp * FFhtp * (mFS - mIS) * (-qsq + pow(mdm1, 2) + pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2))) *
          pow(qsq - pow(mFS - mIS, 2), -1)) - 3 * pow(qsq, -1) *
         (FFG0 * FFhtp * (mFS + mIS) *
          (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) *
          (pow(mdm1, 4) - 2 * pow(mdm1, 2) *
           (qsq + pow(mdm2, 2)) + pow(qsq - pow(mdm2, 2), 2)) + 4 * FFGv *
          (FFhv * (mFS + mIS) * pow(mdm1, 2) *
           (-qsq + pow(mdm1, 2) - pow(mdm2, 2)) *
           (pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
            (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
            (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2)) + FFhtv *
           (mFS - mIS) * pow(qsq - pow(mFS - mIS, 2), -1) *
           (qsq * pow(mIS, 2) *
            (-(qsq * pow(mdm2, 2) *
               (2 * pow(mdm2, 2) - 2 * pow(mFS, 2) + pow(mIS, 2))) +
             pow(mdm2, 4) * (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) +
             (pow(mdm2, 2) - pow(mFS, 2)) * pow(qsq, 2)) + pow(mdm1, 6) *
            (-(qsq * (pow(mFS, 2) + 2 * pow(mIS, 2))) + pow(qsq, 2) +
             pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 4) *
            (qsq *
             (pow(mFS, 4) - 3 * pow(mFS, 2) * pow(mIS, 2) - pow(mdm2, 2) *
              (4 * pow(mFS, 2) + 3 * pow(mIS, 2)) + 2 * pow(mIS, 4)) +
             (pow(mdm2, 2) - 2 *
              (pow(mFS, 2) + 2 * pow(mIS, 2))) * pow(qsq, 2) + 2 * pow(qsq, 3)
             + 2 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) +
            pow(mdm1, 2) *
            ((2 * pow(mdm2, 4) + 4 * pow(mdm2, 2) *
              (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) -
             (3 * pow(mdm2, 2) + pow(mFS, 2) + 2 * pow(mIS, 2)) * pow(qsq, 3) +
             pow(qsq, 4) + pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2) -
             qsq * pow(mdm2, 2) *
             (pow(mdm2, 2) * (3 * pow(mFS, 2) + 2 * pow(mIS, 2)) +
              pow(pow(mFS, 2) - pow(mIS, 2), 2))))) + FFGp * FFhtp *
          (mFS - mIS) * pow(qsq - pow(mFS - mIS, 2), -1) *
          (pow(mdm1, 6) *
           (-2 * qsq * (3 * pow(mFS, 2) + pow(mIS, 2)) + pow(qsq, 2) +
            pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 4) *
           (qsq * (5 * pow(mFS, 4) - 6 * pow(mFS, 2) * pow(mIS, 2) -
                   2 * pow(mdm2, 2) *
                   (3 * pow(mFS, 2) + 5 * pow(mIS, 2)) + pow(mIS, 4)) +
            (7 * pow(mdm2, 2) - 2 *
             (5 * pow(mFS, 2) + pow(mIS, 2))) * pow(qsq, 2) + pow(qsq, 3) + 3 *
            pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2)) - pow(mdm1, 2) *
           ((pow(mdm2, 4) - 3 * pow(mFS, 4) + 6 * pow(mFS, 2) * pow(mIS, 2) -
             4 * pow(mdm2, 2) *
             (pow(mFS, 2) + pow(mIS, 2)) + pow(mIS, 4)) * pow(qsq, 2) - 2 *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2)) * pow(qsq, 3) +
            pow(qsq, 4) -
            3 * pow(mdm2, 4) * pow(pow(mFS, 2) - pow(mIS, 2), 2) + 2 * qsq *
            (pow(mdm2, 4) * (pow(mFS, 2) + 3 * pow(mIS, 2)) +
             3 * pow(mdm2, 2) * pow(pow(mFS, 2) - pow(mIS, 2), 2))) +
           (qsq - pow(mdm2, 2)) * pow(
               pow(mdm2, 2) * (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
               (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)) - pow(qsq, 2), 2)))))
                                                 )

    amp_sq = (J_11 + J_22 + J_33 + J_44 + J_55 + J_66 + J_77 + J_88 + J_99 + J_00
              + J_13 + J_17 + J_19 + J_24 + J_28 + J_20 + J_36 + J_37 + J_39 + J_45 + J_48 + J_40
              + J_58 + J_50 + J_67 + J_69 + J_79 + J_80)
    return f_phase_space * amp_sq
