#!/usr/bin/env python

"""B介子衰变到赝标介子过程的形状因子, 参考hep-ph/0406232"""

from pathlib import Path

import yaml

with open(Path(__file__).parent / 'ball_zwicky.yaml', 'r') as file:
    ff_pars = yaml.load(file, yaml.SafeLoader)


def fp_pi(q2, pars):
    r1 = pars.get('r1')
    r2 = pars.get('r2')
    m12 = pars.get('m1') ** 2
    mfit2 = pars.get('mfit^2')
    return r1 / (1 - q2 / m12) + r2 / (1 - q2 / mfit2)


def fp_K(q2, pars):
    r1 = pars.get('r1')
    r2 = pars.get('r2')
    m12 = pars.get('m1') ** 2
    return r1 / (1 - q2 / m12) + r2 / (1 - q2 / m12) ** 2


def f0(q2, pars):
    r2 = pars.get('r2')
    mfit2 = pars.get('mfit^2')
    return r2 / (1 - q2 / mfit2)


def ff(process, q2):
    if 'B->pi' in process:
        _ff = {'f+': fp_pi(q2, pars=ff_pars['B->pi form factor']['f+']),
               'f0': f0(q2, pars=ff_pars['B->pi form factor']['f0']),
               'fT': fp_pi(q2, pars=ff_pars['B->pi form factor']['fT'])}
        return _ff
    elif 'B->K' in process:
        _ff = {'f+': fp_K(q2, pars=ff_pars['B->K form factor']['f+']),
               'f0': f0(q2, pars=ff_pars['B->K form factor']['f0']),
               'fT': fp_K(q2, pars=ff_pars['B->K form factor']['fT'])}
        return _ff
    elif 'B->eta' in process:
        _ff = {'f+': fp_K(q2, pars=ff_pars['B->eta form factor']['f+']),
               'f0': f0(q2, pars=ff_pars['B->eta form factor']['f0']),
               'fT': fp_K(q2, pars=ff_pars['B->eta form factor']['fT'])}
        return _ff
    else:
        return {}
