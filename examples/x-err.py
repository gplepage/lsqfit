#!/usr/bin/env python
# encoding: utf-8
"""
x-err.py - Code for "x has Error Bars"
"""
# Copyright (c) 2012 G. Peter Lepage. 
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from __future__ import print_function   # makes this work for python2 and 3

DO_PLOT = False
DO_BOOTSTRAP = False
DO_SVD = True

SVDCUT = 1e-10 if DO_SVD else None

import lsqfit
import numpy as np
import gvar as gv

def f_exact(x):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

def f(p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    x = p['x']                      # x is a parameter
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data():                    # make x,y fit data
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
    cr = gv.gvar(0.0,0.01)
    c = [gv.gvar(cr(),0.01) for n in range(100)]
    x_xmax = x/max(x)
    noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
    y = f_exact(x)*noise            # noisy y[i]s
    xfac = gv.gvar(1.0,0.00001)     # gaussian distrib'n: 1 +- 0.001%
    x = np.array([xi*gv.gvar(xfac(),xfac.sdev) for xi in x]) # noisy x[i]s
    return x,y

def make_prior(nexp,x):             # make priors for fit parameters
    prior = gv.BufferDict()         # dictionary-like
    prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
    prior['E'] = [gv.gvar(i+1,0.5) for i in range(nexp)]
    prior['x'] = x                  # x now an array of parameters
    return prior

def main():
    gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    x,y = make_data()               # make fit data
    p0 = None                       # make larger fits go faster (opt.)
    for nexp in range(3,8):
        print('************************************* nexp =',nexp)
        prior = make_prior(nexp,x)
        fit = lsqfit.nonlinear_fit(data=y,fcn=f,prior=prior,p0=p0,svdcut=SVDCUT)
        fit.check_roundoff()
        print(fit)                  # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print('E1/E0 =',(E[1]/E[0]).fmt(),'  E2/E0 =',(E[2]/E[0]).fmt())
        print('a1/a0 =',(a[1]/a[0]).fmt(),'  a2/a0 =',(a[2]/a[0]).fmt())
        print()
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
    
    #
    if DO_BOOTSTRAP:
        Nbs = 10                                     # number of bootstrap copies
        outputs = {'E1/E0':[], 'E2/E0':[], 'a1/a0':[],'a2/a0':[],'E1':[],'a1':[]}   # results
        for bsfit in fit.bootstrap_iter(n=Nbs):
            E = bsfit.pmean['E']                     # best-fit parameters
            a = bsfit.pmean['a']
            outputs['E1/E0'].append(E[1]/E[0])       # accumulate results
            outputs['E2/E0'].append(E[2]/E[0])
            outputs['a1/a0'].append(a[1]/a[0])
            outputs['a2/a0'].append(a[2]/a[0])
            outputs['E1'].append(E[1])
            outputs['a1'].append(a[1])
            # print E[:2]
            # print a[:2]
            # print bsfit.chi2/bsfit.dof

        # extract means and standard deviations from the bootstrap output
        for k in outputs:
            outputs[k] = gv.gvar(np.mean(outputs[k]),
                                 np.std(outputs[k])).fmt()
        print('Bootstrap results:')
        print('E1/E0 =',outputs['E1/E0'],'  E2/E1 =',outputs['E2/E0'])
        print('a1/a0 =',outputs['a1/a0'],'  a2/a0 =',outputs['a2/a0'])
        print('E1 =',outputs['E1'],'  a1 =',outputs['a1'])
    
    if DO_PLOT:
        print(fit.format(100))                   # print the fit results
        import pylab as pp   
        from gvar import mean,sdev     
        fity = f(x,fit.pmean)
        ratio = y/fity
        pp.xlim(0,21)
        pp.xlabel('x')
        pp.ylabel('y/f(x,p)')
        pp.errorbar(x=gv.mean(x),y=gv.mean(ratio),yerr=gv.sdev(ratio),fmt='ob')
        pp.plot([0.0,21.0],[1.0,1.0])
        pp.show()

if __name__ == '__main__':
    main()
