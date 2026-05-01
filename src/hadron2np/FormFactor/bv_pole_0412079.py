#!/usr/bin/env python

"""B介子衰变常数定义, 参考hep-ph/0412079"""

from pathlib import Path

import yaml

with open(Path(__file__).parent / 'ball_zwicky.yaml', 'r') as file:
    ff_pars = yaml.load(file, yaml.SafeLoader)


def f_eq59(q2, pars):
    r1 = pars.get('r1')
    r2 = pars.get('r2')
    m12 = pars.get('mR') ** 2
    mfit2 = pars.get('mfit^2')
    return r1 / (1 - q2 / m12) + r2 / (1 - q2 / mfit2)


def f_eq60(q2, pars):
    r1 = pars.get('r1')
    r2 = pars.get('r2')
    mfit2 = pars.get('mfit^2')
    return r1 / (1 - q2 / mfit2) + r2 / (1 - q2 / mfit2) ** 2


def f_eq61(q2, pars):
    r2 = pars.get('r2')
    mfit2 = pars.get('mfit^2')
    return r2 / (1 - q2 / mfit2)


def ff(process, q2):
    _processes = ['B->rho', 'Bs->K*', 'B->K*', 'B->omega', 'Bs->phi']
    if process in _processes:
        _ff = {'V': f_eq59(q2, pars=ff_pars[process + ' form factor']['V']),
               'A0': f_eq59(q2, pars=ff_pars[process + ' form factor']['A0']),
               'A1': f_eq61(q2, pars=ff_pars[process + ' form factor']['A1']),
               'A2': f_eq60(q2, pars=ff_pars[process + ' form factor']['A2']),
               'T1': f_eq59(q2, pars=ff_pars[process + ' form factor']['T1']),
               'T2': f_eq61(q2, pars=ff_pars[process + ' form factor']['T2']),
               'T3': f_eq60(q2, pars=ff_pars[process + ' form factor']['T3'])}
        # _ff['T3_notilde_nomass'] = (_ff['T3'] - _ff['T2']) / q2
        return _ff
    else:
        return {}
