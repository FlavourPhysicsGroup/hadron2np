from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f
from hadron2np.methods import phase_space_3_body


def Gamma_PX(L_S, L_P, L_V, L_A, mdm1, mIS, mFS, miq, mfq, ffs):
    raise NotImplementedError()
    lam_sqrt = sqrt(kallen_f(mIS**2, mFS**2, mdm1**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    FFf0 = ffs['f0']

    J_11 = abs(L_S)**2 * ()
    J_22 = abs(L_V)**2 * ()
    J_12 = 2 * (L_S * L_V.conjugate()).imag * ()

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq


def dGamma_PXX(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, mFS, miq, mfq, ffs, qsq):
    raise NotImplementedError()
    f_phase_space = 1 / (256 * pi**3 * mIS**3)

    FFf0 = ffs['f0']
    FFfp = ffs['f+']

    ps3 = phase_space_3_body(mIS, mFS, mdm1, mdm2, qsq)
    m23sq = ps3.m23sq()
    m23sqsq = ps3.m23sq_sq()
    m23sqcube = ps3.m23sq_cube()

    J_11 = abs(L_S)**2 * ()
    J_22 = abs(L_V)**2 * ()
    J_12 = 2 * (L_S * L_V.conjugate()).real * ()

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq
