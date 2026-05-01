from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f
from hadron2np.methods import phase_space_3_body


def Gamma_LambdaX(L_S, L_P, L_V, L_A, mdm1, mIS, mFS, miq, mfq, ffs):
    func_f = (mIS - mFS)**2 * ((mIS + mFS)**2 - mdm1**2)
    func_g = (mIS + mFS)**2 * ((mIS - mFS)**2 - mdm1**2)

    lam_sqrt = sqrt(kallen_f(mIS**2, mFS**2, mdm1**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    FFF0 = ffs['f0']
    FFG0 = ffs['g0']

    J_11 = abs(L_S)**2 * FFF0**2 * func_f / (miq - mfq)**2
    J_22 = abs(L_P)**2 * FFG0**2 * func_g / (miq + mfq)**2
    J_33 = abs(L_V)**2 * FFF0**2 * func_f
    J_44 = abs(L_A)**2 * FFG0**2 * func_g
    J_13 = 2 * (L_S * L_V.conjugate()).imag * FFF0**2 * func_f / (miq - mfq)
    J_24 = -2 * (L_A * L_P.conjugate()).real * FFG0**2 * func_g / (miq + mfq)

    amp_sq = J_11 + J_22 + J_33 + J_44 + J_13 + J_24
    return f_phase_space * amp_sq


def dGamma_LambdaXX(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, mFS, miq, mfq, ffs, qsq):
    f_phase_space = 1 / (256 * pi**3 * mIS**3)

    FFF0 = ffs['f0']
    FFFp = ffs['f+']
    FFFv = ffs['fp']
    FFG0 = ffs['g0']
    FFGp = ffs['g+']
    FFGv = ffs['gp']

    func_f = (mIS - mFS)**2 * ((mIS + mFS)**2 - qsq)
    func_g = (mIS + mFS)**2 * ((mIS - mFS)**2 - qsq)

    ps3 = phase_space_3_body(mIS, mFS, mdm1, mdm2, qsq)
    m23sq = ps3.m23sq()
    m23sq_sq = ps3.m23sq_sq()
    m23sq_cube = ps3.m23sq_cube()

    J_11 = abs(L_S)**2 * FFF0**2 * m23sq * func_f / (miq - mfq)**2
    J_22 = abs(L_P)**2 * FFG0**2 * m23sq * func_g / (miq + mfq)**2
    J_33 = abs(L_V)**2 * (
        (-4 * m23sq_cube * (-(pow(FFFv, 2) * qsq) + pow(FFFp, 2) * pow(mFS + mIS, 2))) /
        (3. * (qsq - pow(mFS + mIS, 2))) +
        (m23sq *
         (4 * pow(FFFv, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (-pow(mFS, 4) + qsq *
            (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)))) - pow(
                FFF0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (mFS - mIS) *
                (-qsq + pow(mFS + mIS, 2)) - FFFp * (mFS + mIS) *
                (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
                 (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
                 (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))), 2) / pow(qsq, 2))) /
        (qsq - pow(mFS + mIS, 2)) +
        (2 * m23sq_sq *
         (-(pow(FFFv, 2) *
            (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq * (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)))) +
          (FFFp * (mFS + mIS) *
           (-(FFF0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (mFS - mIS) *
              (-qsq + pow(mFS + mIS, 2))) + FFFp * (mFS + mIS) *
            (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))))) / qsq)) / (qsq - pow(mFS + mIS, 2)))
    J_44 = abs(L_A)**2 * (
        (m23sq_cube * (4 * pow(FFGv, 2) * qsq - 4 * pow(FFGp, 2) * pow(mFS - mIS, 2))) /
        (3. * (qsq - pow(mFS - mIS, 2))) +
        (m23sq *
         (4 * pow(FFGv, 2) *
          (pow(mdm1, 4) * pow(mFS, 2) + pow(mIS, 2) *
           (qsq * (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mdm2, 2) *
            (pow(mdm2, 2) - pow(mFS, 2) + pow(mIS, 2))) - pow(mdm1, 2) *
           (-pow(mFS, 4) + qsq *
            (-pow(mdm2, 2) + pow(mFS, 2)) + pow(mFS, 2) * pow(mIS, 2) + pow(mdm2, 2) *
            (pow(mFS, 2) + pow(mIS, 2)))) - pow(
                FFG0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (-qsq + pow(mFS - mIS, 2)) *
                (mFS + mIS) - FFGp * (mFS - mIS) *
                (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
                 (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
                 (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))), 2) / pow(qsq, 2))) /
        (qsq - pow(mFS - mIS, 2)) +
        (2 * m23sq_sq *
         (-(pow(FFGv, 2) *
            (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq * (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2)))) +
          (FFGp * (mFS - mIS) *
           (-(FFG0 * (pow(mdm1, 2) - pow(mdm2, 2)) * (-qsq + pow(mFS - mIS, 2)) *
              (mFS + mIS)) + FFGp * (mFS - mIS) *
            (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
             (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
             (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))))) / qsq)) / (qsq - pow(mFS - mIS, 2)))
    J_13 = 2 * (L_S * L_V.conjugate()).real * (
        (FFF0 * FFFp * m23sq_sq * (pow(mFS, 2) - pow(mIS, 2))) / (miq - mfq) +
        (FFF0 * m23sq *
         (FFF0 * (pow(mdm1, 2) - pow(mdm2, 2)) * pow(mFS - mIS, 2) *
          (-qsq + pow(mFS + mIS, 2)) - FFFp * (pow(mFS, 2) - pow(mIS, 2)) *
          (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))))) / (qsq * (miq - mfq)))
    J_24 = 2 * (L_P * L_A.conjugate()).imag * (
        (FFG0 * FFGp * m23sq_sq * (-pow(mFS, 2) + pow(mIS, 2))) / (miq + mfq) +
        (FFG0 * m23sq *
         (-(FFG0 * (pow(mdm1, 2) - pow(mdm2, 2)) *
            (-qsq + pow(mFS - mIS, 2)) * pow(mFS + mIS, 2)) + FFGp * (pow(mFS, 2) - pow(mIS, 2)) *
          (-pow(qsq, 2) + pow(mdm1, 2) * (qsq + pow(mFS, 2) - pow(mIS, 2)) + pow(mdm2, 2) *
           (-pow(mFS, 2) + pow(mIS, 2)) + qsq *
           (pow(mdm2, 2) + pow(mFS, 2) + pow(mIS, 2))))) / (qsq * (miq + mfq)))

    amp_sq = J_11 + J_22 + J_33 + J_44 + J_13 + J_24
    return f_phase_space * amp_sq
