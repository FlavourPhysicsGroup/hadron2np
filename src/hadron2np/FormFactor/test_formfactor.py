# import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import particle
from hepunits import units
import hadron2np

root_path = '/home/reson/Archive/1-Articles/Preparing/DM_EFT/Code'

# from fdm.classes import Implementation
# pgroup: Implementation =
# print(fdm.default_parameters['B->pi form factors'])
# print(fdm.default_parameters['B->pi form factors'].get_implementation(name='one-pole'))
# pgroup: Implementation = fdm.default_parameters['B->pi form factors'].get_implementation(name='one-pole')
# print(pgroup.get_central(wc_obj=None, par_dict=None, q2=0))

# print(AuxiliaryQuantity['B->pi form factor'].prediction(None, None, q2=0))
# exit()

matplotlib.rcParams.update({
    'axes.linewidth': 2, 'axes.labelsize': 16,  # 'axes.labelweight':'bold',
    'axes.titlesize': 22,  # 'axes.titleweight':'bold',
    'xtick.top': True, 'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
    'xtick.major.size': 7, 'ytick.major.size': 7,
    'xtick.minor.size': 4, 'ytick.minor.size': 4,
    'xtick.major.width': 2, 'ytick.major.width': 2,
    'xtick.minor.width': 2, 'ytick.minor.width': 2,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'legend.frameon': False,
    'mathtext.fontset': 'custom', 'mathtext.cal': 'DejaVu Sans',
    'mathtext.bf': 'sans:bold:italic',
    'mathtext.default': 'it',
})


def test_BtoP():
    _q2 = np.linspace(0, 14, 100)

    # _process = ['B->pi', 'B->K', 'B->eta']

    pi_dict = hadron2np.parameter_groups['B->pi form factors'].get_implementation(name='one-pole') \
        .get_values(wc_obj=None, par_dict=None, q2=_q2)
    K_dict = hadron2np.parameter_groups['B->K form factors'].get_implementation(name='one-pole') \
        .get_values(wc_obj=None, par_dict=None, q2=_q2)
    eta_dict = hadron2np.parameter_groups['B->eta form factors'].get_implementation(name='one-pole') \
        .get_values(wc_obj=None, par_dict=None, q2=_q2)
    res = {
        'B->pi f+': pi_dict['f+'],
        'B->pi f0': pi_dict['f0'],
        'B->pi fT': pi_dict['fT'],
        'B->K f+': K_dict['f+'],
        'B->K f0': K_dict['f0'],
        'B->K fT': K_dict['fT'],
        'B->eta f+': eta_dict['f+'],
        'B->eta f0': eta_dict['f0'],
        'B->eta fT': eta_dict['fT'],
    }
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    ls1 = ['--', ':', '-']
    ls2 = ['--', ':', '-']
    ls3 = ['--', ':', '-']
    for _k, _v in res.items():
        if 'B->pi' in _k:
            axs[0, 0].plot(_q2, _v, ls=ls1.pop(), label=_k)
        elif 'B->K' in _k:
            axs[0, 1].plot(_q2, _v, ls=ls2.pop(), label=_k)
        elif 'B->eta' in _k:
            axs[1, 0].plot(_q2, _v, ls=ls3.pop(), label=_k)
        else:
            pass
    names = ['', r'$B\to\eta$', r'$B\to K$', r'$B\to\pi$']
    for ax in axs.reshape(1, -1)[0]:
        ax.grid(True)
        ax.legend(frameon=True, fontsize=12)
        ax.set_xlim(0, 14)
        ax.set_ylim(0.2, 1.0)
        ax.set_xlabel(r'$q^2$')
        ax.set_ylabel(f'{names.pop()}')
    axs[1, 1].remove()
    # plt.grid()
    # plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig(root_path + '/figures/checkout-0604232.pdf')


def test_BtoV():
    _process = ['B->rho', 'Bs->K*', 'B->K*', 'B->omega', 'Bs->phi']
    names = [r'$B\to\rho$', r'$B_s\to K^*$', r'$B\to K^*$', r'$B\to\omega$', r'$B_s\to\phi$']
    fig, axs = plt.subplots(len(_process), 2, figsize=(9, 3 * len(_process)))

    _q2 = np.linspace(0, 14, 100)

    for _num, _p in enumerate(_process):
        ff_dict = hadron2np.parameter_groups[_p + ' form factors'].get_implementation(name='one-pole') \
            .get_values(wc_obj=None, par_dict=None, q2=_q2)

        lss1 = ['-.', ':', '-', '--']
        for _ff in ['V', 'A0', 'A1', 'A2']:
            axs[_num, 0].plot(_q2, ff_dict[_ff], ls=lss1.pop(), label=_ff)
        axs[_num, 0].set_ylabel(f'{names[_num]}')

        b_str, v_str = _p.split(sep='->')
        if b_str == 'Bs':
            mB = particle.Particle.findall('B(s)0')[0].mass / units.GeV
        elif b_str == 'B':
            mB = particle.Particle.findall('B0')[0].mass / units.GeV
        mV = particle.Particle.findall(v_str)[0].mass / units.GeV
        assert mB is not None
        assert mV is not None
        lss2 = ['-.', ':', '-']
        for _ff in ['T1', 'T2', 'T3']:
            if _ff == 'T3':
                ff_dict['T3'] = ff_dict['T3_notilde_nomass'] * (mB ** 2 - mV ** 2)
            axs[_num, 1].plot(_q2, ff_dict[_ff], ls=lss2.pop(), label=_ff)
        axs[_num, 1].set_ylabel(f'{names[_num]}')

    for ax in axs.reshape(1, -1)[0]:
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.set_xlim(0, 14)
        ax.set_ylim(0., 1.0)
        # ax.set_xlabel(r'$q^2$')
        ax.grid(True)
        ax.legend(frameon=True, fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig(root_path + '/figures/checkout-0412079.pdf')


def test_bpi_bcl_1103():
    xx = np.linspace(0, 26.4, 100)
    yy = hadron2np.parameter_groups['B->pi form factors'].get_implementation(name='bcl').get_values(None, None, q2=xx)
    plt.plot(xx, yy['f0'])
    plt.xlim(0, 26.5)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # test_BtoP()
    # test_BtoV()
    test_bpi_bcl_1103()
