#!/usr/bin/env python
# encoding: utf-8
"""
eg3.py - Code for "Correlated Parameters"

Created by Peter Lepage on 2010-01-04.
Copyright (c) 2010 Cornell University. All rights reserved.
"""
DO_PLOT = False
DO_BOOTSTRAP = False
DO_SVD = False

SVDCUT = 1e-10 if DO_SVD else None

import sys
import lsqfit
import numpy as np
import gvar as gd
import tee

def f_exact(x):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

def f(x,p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data():                    # make x,y fit data
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
    cr = gd.gvar(0.0,0.01)
    c = [gd.gvar(cr(),0.01) for n in range(100)]
    x_xmax = x/max(x)
    noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
    y = f_exact(x)*noise            # noisy y[i]s
    return x,y

def make_prior(nexp):               # make priors for fit parameters
    prior = lsqfit.GPrior()         # Gaussian prior -- dictionary-like
    prior['a'] = [gd.gvar(0.5,0.5) for i in range(nexp)]
    de = [gd.gvar(0.9,0.01) for i in range(nexp)]
    de[0] = gd.gvar(1,0.5)     
    prior['E'] = [sum(de[:i+1]) for i in range(nexp)]
    return prior

def main():
    gd.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    x,y = make_data()               # make fit data
    p0 = None                       # make larger fits go faster (opt.)
    sys_stdout = sys.stdout
    for nexp in range(3,9):
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior,p0=p0,svdcut=SVDCUT)
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
        if nexp == 7:
            sys.stdout = tee.tee(sys_stdout, open("eg3.out","w"))
        print '************************************* nexp =',nexp
        print fit                   # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
        print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
        # print (E[1]/E[0]).partialsdev(fit.prior['E'])
        # print (E[1]/E[0]).partialsdev(fit.prior['a'])
        # print (E[1]/E[0]).partialsdev(fit.y)
        sys.stdout = sys_stdout
        print
    
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
            outputs[k] = gd.gvar(np.mean(outputs[k]),np.std(outputs[k]))
        print 'Bootstrap results:'
        print 'E1/E0 =',outputs['E1/E0'],'  E2/E1 =',outputs['E2/E0']
        print 'a1/a0 =',outputs['a1/a0'],'  a2/a0 =',outputs['a2/a0']
        print 'E1 =',outputs['E1'],'  a1 =',outputs['a1']

    if DO_PLOT:
        print fit.format(100)                   # print the fit results
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
