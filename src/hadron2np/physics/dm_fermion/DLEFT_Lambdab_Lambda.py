from numpy import pi, sqrt
from hadron2np.methods import phase_space_3_body


def dGamma_LambdaXX(L_S, L_P, L_V, L_A, L_T, L_T5, mdm1, mdm2,
                    mIS, mFS, miq, mfq, ffs, qsq):
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

    J_11 = abs(L_S)**2 * ()
    J_22 = abs(L_P)**2 * ()
    J_33 = abs(L_V)**2 * ()
    J_44 = abs(L_A)**2 * ()
    J_55 = abs(L_T)**2 * ()
    J_66 = abs(L_T5)**2 * ()
    J_13 = 2 * (L_S * L_V.conjugate()).real * ()
    J_24 = 2 * (L_P * L_A.conjugate()).imag * ()

    amp_sq = (J_11 + J_22 + J_33 + J_44 + J_55 + J_66
              + J_13 + J_24)
    return f_phase_space * amp_sq
