#!/usr/bin/env python
# encoding: utf-8
"""
eg5.py - Code for "y Has No Errors; Marginalization"

Created by Peter Lepage on 2016-12.
Copyright (c) 2016 Cornell University. All rights reserved.
"""
DO_PLOT = True
DO_BOOTSTRAP = True

import sys
import tee

import numpy as np
import gvar as gv
import lsqfit

sys_stdout = sys.stdout

def main():
    do_fit(do_plot=DO_PLOT)
    do_fit(svdcut=1e-19, do_plot=False)

def do_fit(svdcut=None, do_plot=False):
    if svdcut is None:
        svdcut = lsqfit.nonlinear_fit.set()['svdcut']
        sys.stdout = tee.tee(sys_stdout, open('eg5a.out', 'w'))
        default_svd = True
    else:
        default_svd = False
    x, y = make_data()
    prior = make_prior(100)              # 20 exponential terms in all (10 gives same result)
    p0 = None
    for nexp in range(1, 6):
        # marginalize the last 100 - nexp terms
        fit_prior = gv.BufferDict()     # part of prior used in fit
        ymod_prior = gv.BufferDict()    # part of prior absorbed in ymod
        for k in prior:
            fit_prior[k] = prior[k][:nexp]
            ymod_prior[k] = prior[k][nexp:]
        ymod = y - fcn(x, ymod_prior)
        # fit modified data with just nexp terms
        fit = lsqfit.nonlinear_fit(
            data=(x, ymod), prior=fit_prior, fcn=fcn, p0=p0, tol=1e-10,
            svdcut=svdcut
            )
        if not default_svd and nexp == 5:
            sys.stdout = tee.tee(sys_stdout, open('eg5b.out', 'w'))
        print '************************************* nexp =',nexp
        print fit.format(True)
        p0 = fit.pmean
        if do_plot:
            import matplotlib.pyplot as plt
            if nexp > 4:
                continue
            plt.subplot(2, 2, nexp)
            if nexp not in  [1, 3]:
                plt.yticks([0.05, 0.10, 0.15, 0.20, 0.25], [])
            else:
                plt.ylabel('y')
            if nexp not in [3, 4]:
                plt.xticks([1.0, 1.5, 2.0, 2.5], [])
            else:
                plt.xlabel('x')
            plt.errorbar(x=x, y=gv.mean(ymod), yerr=gv.sdev(ymod), fmt='bo')
            plt.plot(x, y, '-r')
            plt.plot(x, fcn(x, fit.pmean), ':k')
            plt.text(1.75, 0.22,'nexp = {}'.format(nexp))
            if nexp == 4:
                plt.savefig('eg5.png', bbox_inches='tight')
                plt.show()
    # print summary information and error budget
    E = fit.p['E']              # best-fit parameters
    a = fit.p['a']
    outputs = {
        'E1/E0':E[1] / E[0], 'E2/E0':E[2] / E[0],
        'a1/a0':a[1] / a[0], 'a2/a0':a[2] / a[0]
        }
    inputs = {
        'E prior':prior['E'], 'a prior':prior['a'],
        'svd cut':fit.svdcorrection,
        }
    print(fit.fmt_values(outputs))
    print(fit.fmt_errorbudget(outputs, inputs))
    sys.stdout = sys_stdout

def fcn(x,p):
    a = p['a']       # array of a[i]s
    E = p['E']       # array of E[i]s
    return np.sum(ai * np.exp(-Ei*x) for ai, Ei in zip(a, E))

def make_prior(nexp):
    prior = gv.BufferDict()
    prior['a'] = gv.gvar(nexp * ['0.5(5)'])
    dE = gv.gvar(nexp * ['1.0(1)'])
    prior['E'] = np.cumsum(dE)
    return prior

def make_data():
    x = np.array([ 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6])
    y = np.array([
        0.2740471001620033,  0.2056894154005132,  0.158389402324004 ,
        0.1241967645280511,  0.0986901274726867,  0.0792134506060024,
        0.0640743982173861,  0.052143504367789 ,  0.0426383022456816,
        ])
    return x, y

if __name__ == '__main__':
    main()