#!/usr/bin/env python
# encoding: utf-8
"""
err-budget.py - Code for "Tuning Priors and the Empirical Bayes Criterion" and
              for "Partial Errors and Error Budgets"
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

DO_EMPBAYES = False
DO_ERRORBUDGET = True
DO_PLOT = False
DO_SVD = True

SVDCUT = 1e-10 if DO_SVD else None

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
import lsqfit
import numpy as np
import gvar as gv


def f_exact(x):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

def f(x,p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data():                    # make x,y fit data
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
    cr = gv.gvar(0.0,0.01)
    c = [gv.gvar(cr(),cr.sdev) for n in range(100)]
    x_xmax = x/max(x)
    noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
    y = f_exact(x)*noise            # noisy y[i]s
    return x,y

def make_prior(nexp):               # make priors for fit parameters
    prior = gv.BufferDict()         # dictionary-like
    prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
    de = [gv.gvar(0.9,0.01) for i in range(nexp)]
    de[0] = gv.gvar(1,0.5)     
    prior['E'] = [sum(de[:i+1]) for i in range(nexp)]
    return prior

def main():
    gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    x,y = make_data()               # make fit data
    p0 = None                       # make larger fits go faster (opt.)
    for nexp in range(3,8):
        print('************************************* nexp =',nexp)
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior,p0=p0,svdcut=SVDCUT)
        print(fit)                  # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print('E1/E0 =',(E[1]/E[0]).fmt(),'  E2/E0 =',(E[2]/E[0]).fmt())
        print('a1/a0 =',(a[1]/a[0]).fmt(),'  a2/a0 =',(a[2]/a[0]).fmt())
        print()
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
    
    if DO_ERRORBUDGET:
        outputs = OrderedDict([
            ('E1/E0', E[1]/E[0]), ('E2/E0', E[2]/E[0]),         
            ('a1/a0', a[1]/a[0]), ('a2/a0', a[2]/a[0])
            ])
        inputs = OrderedDict([
            ('E', fit.prior['E']), ('a', fit.prior['a']),
            ('y', y), ('svd', fit.svdcorrection)
            ])
        print(fit.fmt_values(outputs))
        print(fit.fmt_errorbudget(outputs,inputs))
        
    if DO_EMPBAYES:
        def fitargs(z,nexp=nexp,prior=prior,f=f,data=(x,y),p0=p0):
            z = gv.exp(z)
            prior['a'] = [gv.gvar(0.5,0.5*z[0]) for i in range(nexp)]
            return dict(prior=prior,data=data,fcn=f,p0=p0)
        ##
        z0 = [0.0]
        fit,z = lsqfit.empbayes_fit(z0,fitargs,tol=1e-3)
        print(fit)                  # print the optimized fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print('E1/E0 =',(E[1]/E[0]).fmt(),'  E2/E0 =',(E[2]/E[0]).fmt())
        print('a1/a0 =',(a[1]/a[0]).fmt(),'  a2/a0 =',(a[2]/a[0]).fmt())
        print("prior['a'] =",fit.prior['a'][0].fmt())
        print()
    
    if DO_PLOT:
        import pylab as pp   
        from gvar import mean,sdev     
        fity = f(x,fit.pmean)
        ratio = y/fity
        pp.xlim(0,21)
        pp.xlabel('x')
        pp.ylabel('y/f(x,p)')
        pp.errorbar(x=x,y=mean(ratio),yerr=sdev(ratio),fmt='ob')
        pp.plot([0.0,21.0],[1.0,1.0])
        pp.show()

if __name__ == '__main__':
    main()
