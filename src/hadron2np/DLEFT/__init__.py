import yaml
from pathlib import Path
import numpy as np

with open(Path(__file__).parent / 'conventions.yaml', 'r') as file:
    par_DLEFT = yaml.safe_load(file)


def get_zero_wcs(basis):
    """初始化一个完整的 wcs 字典"""
    if basis == 'DLEFT(S/P)':
        return {
            name[0]: np.zeros((3, 3, 2, 2), dtype=complex)
            for name in par_DLEFT['wcs in S/P basis']
        }
    elif basis == 'DLEFT(L/R)':
        return {
            name[0]: np.zeros((3, 3, 2, 2), dtype=complex)
            for name in par_DLEFT['wcs in L/R basis']
        }
    else:
        raise ValueError(f'算符基矢命名错误: {basis}')


def convert_to_SP_basis(wcs_LR):
    wcs_SP = get_zero_wcs('DLEFT(S/P)')
    wcs_SP['L_S_dphi'] = 1 / 2 * (wcs_LR['L_SR_dphi'].T.conjugate() + wcs_LR['L_SR_dphi'])
    wcs_SP['L_P_dphi'] = 1j / 2 * (wcs_LR['L_SR_dphi'].T.conjugate() - wcs_LR['L_SR_dphi'])
    wcs_SP['L_V_dphi'] = 1 / 2 * (wcs_LR['L_VR_dphi'] + wcs_LR['L_VL_dphi'])
    wcs_SP['L_A_dphi'] = 1 / 2 * (wcs_LR['L_VR_dphi'] - wcs_LR['L_VL_dphi'])

    wcs_SP['L_S_dphi2'] = 1 / 2 * (wcs_LR['L_SR_dphi2'].T.conjugate() + wcs_LR['L_SR_dphi2'])
    wcs_SP['L_P_dphi2'] = 1j / 2 * (wcs_LR['L_SR_dphi2'].T.conjugate() - wcs_LR['L_SR_dphi2'])
    wcs_SP['L_V_dphi2'] = 1 / 2 * (wcs_LR['L_VR_dphi2'] + wcs_LR['L_VL_dphi2'])
    wcs_SP['L_A_dphi2'] = 1 / 2 * (wcs_LR['L_VR_dphi2'] - wcs_LR['L_VL_dphi2'])

    wcs_SP['L_V_dX'] = 1 / 2 * (wcs_LR['L_VR_dX'] + wcs_LR['L_VL_dX'])
    wcs_SP['L_A_dX'] = 1 / 2 * (wcs_LR['L_VR_dX'] - wcs_LR['L_VL_dX'])
    wcs_SP['L_T_dX'] = 1 / 2 * (wcs_LR['L_TR_dX'].T.conjugate() + wcs_LR['L_TR_dX'])
    wcs_SP['L_T5_dX'] = 1j / 2 * (wcs_LR['L_TR_dX'].T.conjugate() - wcs_LR['L_TR_dX'])

    wcs_SP['L_S_dXX'] = 1 / 2 * (wcs_LR['L_SR_dXX'].T.conjugate() + wcs_LR['L_SR_dXX'])
    wcs_SP['L_P_dXX'] = 1j / 2 * (wcs_LR['L_SR_dXX'].T.conjugate() - wcs_LR['L_SR_dXX'])
    wcs_SP['L_V_dXX'] = 1 / 2 * (wcs_LR['L_VR_dXX'] + wcs_LR['L_VL_dXX'])
    wcs_SP['L_A_dXX'] = 1 / 2 * (wcs_LR['L_VR_dXX'] - wcs_LR['L_VL_dXX'])
    wcs_SP['L_V_dXtX'] = 1 / 2 * (wcs_LR['L_VR_dXtX'] + wcs_LR['L_VL_dXtX'])
    wcs_SP['L_A_dXtX'] = 1 / 2 * (wcs_LR['L_VR_dXtX'] - wcs_LR['L_VL_dXtX'])
    wcs_SP['L_T_dXX'] = 1 / 2 * (wcs_LR['L_TR_dXX'].T.conjugate() + wcs_LR['L_TR_dXX'])
    wcs_SP['L_T5_dXX'] = 1j / 2 * (wcs_LR['L_TR_dXX'].T.conjugate() - wcs_LR['L_TR_dXX'])
    wcs_SP['L_DV_dXX'] = 1 / 2 * (wcs_LR['L_DVR_dXX'] + wcs_LR['L_DVL_dXX'])
    wcs_SP['L_DA_dXX'] = 1 / 2 * (wcs_LR['L_DVR_dXX'] - wcs_LR['L_DVL_dXX'])

    wcs_SP['L_S_dchi2'] = 1 / 2 * (0)
    wcs_SP['L_P_dchi2'] = 1j / 2 * (0)
    wcs_SP['L_V_dchi2'] = 1 / 2 * (0)
    wcs_SP['L_A_dchi2'] = 1 / 2 * (0)
    return wcs_SP


def get_wc_dimension(wc_name) -> int:
    """得到 wc 的量纲. 如 6 维算符返回 -2"""
    names = {name[0]: name[1] for name in par_DLEFT['wc names in S/P basis']}
    names.update({name[0]: name[1] for name in par_DLEFT['wc names in L/R basis']})
    return 4 - names[wc_name]
