import hadron2np
from hadron2np.physics import dm_scalar, dm_fermion, dm_vector, sm_invisible
from hadron2np.DLEFT import get_zero_wcs, convert_to_SP_basis
from scipy import integrate
from wilson.util import qcd
import hmff

# dm 模式主要用于区分费米型 DM 的三种不同组合
DM_MODES = {
            ('chi', 'chi'): 1,
            ('chibar', 'chi'): 2,
            ('chibar', 'chibar'): 3
        }

class DecayProcessBase():

    def __init__(self, particles: list, basis: str, parameter=hadron2np.parameters_dict, ff_imp=None):
        # 定义 name 字符串
        self.name = f'{particles[0]}->{particles[1]}' + '+' + ''.join(particles[2:])
        if particles[1] == '0':
            self.name = f'{particles[0]}->' + ''.join(particles[2:])

        # 确认初末态粒子
        self.IS = particles[0]
        self.FS = particles[1]
        self.dms = particles[2:]
        self.dm_name = 'chi' if 'chi' in particles[2] else particles[2]
        self.dm_mode = DM_MODES.get(tuple(particles[2:]), 0)
        
        self.particles = particles

        # 确认 FCNC 过程: 强子矩阵元初末态, 夸克流
        self.fcnc_hadron = '->'.join(
            p[:-1] if p.endswith(('+', '0')) else p for p in particles[:2]
        )
        try:
            self.fcnc_quark = hadron2np.config['hadron matrix element'][self.fcnc_hadron][0]
            self.fcnc_hadron_class = hadron2np.config['hadron matrix element'][self.fcnc_hadron][2]
        except KeyError:
            raise ValueError(f'{self.fcnc_hadron} 不是合法的衰变过程.')

        # 确认 DM 流
        if not self.confirm_final_state():
            raise ValueError(f'{self.dm_name} 不是合法的末态粒子.')

        # 确认接口
        self.par = parameter
        # 目前只支持 DLEFT(S/P) 算符基
        if basis == 'DLEFT(S/P)':
            self.basis = basis
        else:
            raise ValueError(f'{basis} 不是合法的算符基矢.')
        self.wcs = self.get_all_wcs()
        self.IS_width = self.par['Gamma_' + self.IS]

        self._scale = self.get_decay_energy_scale()
        self.index = self.get_flavour_index()
        self.m_sm = self.get_sm_masses()
        self.m_dm = [0, 0]

        self.ff_imp_name = hadron2np.config['form factor scheme']
        if ff_imp is not None:
            self.ff_imp_name = ff_imp
        if self.FS != '0':
            self._formfactor = hmff.formfactors[self.fcnc_hadron].get_impl(
                self.ff_imp_name
            )
        else:
            # 对于类似 Bs -> XX 的过程, 应该使用衰变常数而非形状因子
            self._formfactor = parameter['f_' + self.IS]

    def __repr__(self) -> str:
        return f'DecayProcess({self.name}) caused by: {self.fcnc_quark}'

    def confirm_final_state(self) -> bool:
        match self.dm_name:
            case 'phi':
                self.analytic_dir = dm_scalar
            case 'X':
                self.analytic_dir = dm_vector
            case 'chi':
                self.analytic_dir = dm_fermion
            case 'nu':
                self.analytic_dir = sm_invisible
            case _:
                return False
        return True

    def get_decay_energy_scale(self):
        if 'B' in self.IS:
            return hadron2np.config['scales']['B Decays']
        elif 'K' in self.IS:
            return hadron2np.config['scales']['K Decays']
        elif 'Lambda_b' in self.IS:
            return hadron2np.config['scales']['Lambda_b Decays']

    @staticmethod
    def get_nf(scale):
        mt = hadron2np.config['scales']['n flavour critical scale 5-6']
        mb = hadron2np.config['scales']['n flavour critical scale 4-5']
        mc = hadron2np.config['scales']['n flavour critical scale 3-4']
        if scale >= mt:
            return 6
        elif mb <= scale < mt:
            return 5
        elif mc <= scale < mb:
            return 4
        elif scale < mc:
            return 3
        else:
            raise ValueError("Unexpected value: scale={}".format(scale))

    def get_flavour_index(self) -> list:
        """根据夸克层次的FCNC过程, 得到 Wilson 系数的味指标. 如 sbab: (1, 2, 0, 1)"""
        match_flavour_index = {'d': 0, 's': 1, 'b': 2}
        iq, fq = self.fcnc_quark.split('->')
        iq_index, fq_index = match_flavour_index[iq], match_flavour_index[fq]

        if len(self.dms) == 1:
            return [fq_index, iq_index, 0, 0]  # b->s 衰变在夸克流中是 \bar{s} b, 指标应该是 1200
        elif len(self.dms) == 2:
            return [fq_index, iq_index, 0, 1]

    def get_quark_running_mass(self, quark_name: str, scale: float):
        input_par = [self.par['m_' + quark_name], scale, self.get_nf(scale), self.par['alpha_s']]
        match quark_name:
            case 'b':
                return qcd.m_b(*input_par)
            case 's' | 'd':
                return qcd.m_s(*input_par)
            case _:
                raise ValueError("Quark mass running not implemented: " + quark_name)

    def get_sm_masses(self):
        scale = self._scale
        m_IS = self.par['m_' + self.IS]
        m_FS = self.par['m_' + self.FS]
        # 注意这里的夸克质量, m_d 和 m_s 使用相同的跑动函数
        iq, fq = self.fcnc_quark.split('->')
        m_iq = self.get_quark_running_mass(iq, scale)
        m_fq = self.get_quark_running_mass(fq, scale)
        return m_IS, m_FS, m_iq, m_fq

    def get_parameter(self, name):
        return self.par.get(name)

    def set_parameter(self, name, value) -> bool:
        if name in self.par:
            self.par[name] = value
            return True
        else:
            return False
    
    def set_dm_masses(self, m_dm: list) -> None:
        self.m_dm = m_dm

    def get_all_wcs(self):
        return get_zero_wcs(self.basis)

    def set_wcs(self, new_wcs: dict, basis=None) -> None:
        """根据算符基，初始 self.wcs 字典"""
        if basis is not None:
            self.basis = basis
        all_wcs = get_zero_wcs(self.basis)
        all_wcs.update(new_wcs)
        if self.basis == 'DLEFT(L/R)':
            self.wcs = convert_to_SP_basis(all_wcs)
        elif self.basis == 'DLEFT(S/P)':
            self.wcs = all_wcs
        else:
            raise ValueError(f'算符基矢命名错误: {self.basis}')
        self.wcs.update(new_wcs)

    def width(self, wcs: dict, m_dm: list) -> float:
        ...

    def branching_ratio(self, wcs: dict, m_dm: list) -> float:
        ...


class TwoBodyDecayProcess(DecayProcessBase):

    def __init__(self, particles, basis, parameter=hadron2np.parameters_dict, ff_imp=None):
        super().__init__(particles, basis, parameter, ff_imp)

    def width(self, wcs, m_dm) -> float:
        self.set_wcs(wcs)
        self.set_dm_masses(m_dm)
        if self.FS == '0':
            decay_constant = self.par['f_' + self.IS]
            bare_width = self.analytic_dir.width_2_0_1(
                self.wcs, decay_constant, self.m_sm, self.m_dm, self.fcnc_hadron_class, self.index
            )
        elif len(self.dms) < 2:
            bare_width = self.analytic_dir.width_2_1_1(
                self.wcs, self._formfactor, self.m_sm, self.m_dm, self.fcnc_hadron_class, self.index
            )
        else:
            raise ValueError('两体衰变宽度计算错误: ' + self.name)

        CG_factor = 1  # 考虑中性的 pi0 中会有一个 sqrt(2)
        dm_factor = 1  # 考虑末态两个 phi 相同时的全同性
        if self.FS == 'pi0':
            CG_factor = 1 / 2.0
        if self.index[2] == self.index[3]:
            dm_factor = 1 / 2.0
        return bare_width * CG_factor * dm_factor

    def branching_ratio(self, wcs: dict, m_dm: list) -> float:
        width = self.width(wcs, m_dm)
        width_IS = self.par['Gamma_' + self.IS]
        return width / (width + width_IS)


class ThreeBodyDecayProcess(DecayProcessBase):

    def __init__(self, particles: list, basis, parameter=hadron2np.parameters_dict, ff_imp=None):
        super().__init__(particles, basis, parameter, ff_imp)

    def dWidth_over_dqsq(self, wcs, m_dm, qsq) -> float:
        self.set_wcs(wcs)
        self.set_dm_masses(m_dm)
        if (self.FS == '0') or (len(self.dms) < 2):
            raise ValueError(f'物理意义不明确: {self.name} 微分宽度.')
        elif self.dm_name == 'nu':
            # 标准模型过程直接返回
            bare_dWidth = self.analytic_dir.partial_width_3_1_1(
                self._formfactor,
                self.m_sm,
                f'{self.IS}->{self.FS}',
                qsq,
            )
            return bare_dWidth
        else:
            bare_dWidth = self.analytic_dir.partial_width_3_1_1(
                self.wcs,
                self._formfactor,
                self.m_sm,
                self.m_dm,
                self.fcnc_hadron_class,
                self.index,
                qsq,
                self.dm_mode,
            )
        CG_factor = 1  # 考虑中性的 pi0 中会有一个 sqrt(2)
        dm_factor = 1  # 考虑末态两个 phi 相同时的全同性
        if self.FS in ['pi0', 'rho0']:
            CG_factor = 1 / 2.0
        if self.index[2] == self.index[3]:
            dm_factor = 1 / 2.0
        dwidth = bare_dWidth * CG_factor * dm_factor
        return dwidth

    def dBr_over_dqsq(self, wcs, m_dm, qsq) -> float:
        dwidth = self.dWidth_over_dqsq(wcs, m_dm, qsq)
        width_IS = self.IS_width
        # 返回 BR = Gamma / Gamma_total
        return dwidth / (width_IS + self.width(wcs, m_dm))

    def width(self, wcs: dict, m_dm: list):
        """计算新物理衰变过程的衰变总宽度"""
        m_dm_1, m_dm_2 = m_dm
        m_IS, m_FS, _, _ = self.get_sm_masses()

        def _dGamma_dqsq(qsq):
            return self.dWidth_over_dqsq(wcs, m_dm, qsq)

        q2_min = (m_dm_1 + m_dm_2)**2
        q2_max = (m_IS - m_FS)**2
        width = integrate.quad(_dGamma_dqsq, q2_min, q2_max)[0]
        return width

    def branching_ratio(self, wcs: dict, m_dm: list) -> float:
        width = self.width(wcs, m_dm)
        width_IS = self.par['Gamma_' + self.IS]
        return width / (width + width_IS)
