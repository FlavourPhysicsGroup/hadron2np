from particle import Particle
import hadron2np
from numpy import sqrt


def parse_errors(value) -> dict:
    # TODO: 实现字符串中的误差与中心值解析
    # if isinstance(value)
    return {'central': value, 'upper': 0, 'lower': 0}


def get_particle(name) -> Particle:
    return Particle.from_pdgid(hadron2np.config['pdgid'][name])


def lambda_f(asq, bsq, csq):
    """Kallen function.
    两体相空间中的末态粒子动量 $\vec{p}_1 = lambda^{1/2} / 2M"""
    return asq**2 + bsq**2 + csq**2 - 2 * (asq * bsq + asq * csq + bsq * csq)


class phase_space_3_body():
    def __init__(self, m_I, m_F, m_1, m_2, qsq):
        self.E2_star = self.get_E2_star(qsq, m_1, m_2)
        self.E3_star = self.get_E3_star(qsq, m_I, m_F)
        self.m_plus = self.get_m_plus(m_I, m_F, m_1, m_2, qsq)
        self.m_minus = self.get_m_minus(m_I, m_F, m_1, m_2, qsq)

    @staticmethod
    def get_E2_star(qsq, m_1, m_2):
        return (qsq - m_1**2 + m_2**2) / (2 * sqrt(qsq))

    @staticmethod
    def get_E3_star(qsq, m_I, m_F):
        return (m_I**2 - m_F**2 - qsq) / (2 * sqrt(qsq))

    def get_m_plus(self, m_I, m_F, m_1, m_2, qsq):
        E2 = self.E2_star
        E3 = self.E3_star
        return (E2 + E3)**2 - (sqrt(E2**2 - m_2**2) - sqrt(E3**2 - m_F**2))**2

    def get_m_minus(self, m_I, m_F, m_1, m_2, qsq):
        E2 = self.E2_star
        E3 = self.E3_star
        return (E2 + E3)**2 - (sqrt(E2**2 - m_2**2) + sqrt(E3**2 - m_F**2))**2

    def m23sq(self):
        return self.m_plus - self.m_minus

    def m23sq_sq(self):
        return self.m23sq() * (self.m_plus + self.m_minus)

    def m23sq_cube(self):
        return self.m23sq() * (self.m_plus**2 + self.m_minus**2 + self.m_plus * self.m_minus)
