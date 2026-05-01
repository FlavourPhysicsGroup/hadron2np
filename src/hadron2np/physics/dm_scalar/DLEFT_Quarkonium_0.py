from numpy import pi, sqrt
from hadron2np.methods import lambda_f as kallen_f


def Gamma(L_S, L_P, L_V, L_A, mdm1, mdm2, mIS, miq, mfq, fB) -> float:
    raise NotImplementedError()
    lam_sqrt = sqrt(kallen_f(mIS**2, mdm1**2, mdm2**2))
    f_phase_space = lam_sqrt / (16 * pi * mIS**3)

    J_11 = abs(L_P)**2 * ()
    J_22 = abs(L_A)**2 * ()
    J_12 = 2 * (L_P * L_A.conjugate()).imag * ()

    amp_sq = J_11 + J_22 + J_12
    return f_phase_space * amp_sq
