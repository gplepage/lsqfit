#!/usr/bin/env python
# encoding: utf-8
"""
eg5.py - Code for "y has No Error Bars" and "SVD Cuts and Roundoff Error"

Created by Peter Lepage on 2010-01-04.
Copyright (c) 2010-13 G. Peter Lepage. All rights reserved.
"""

#### run with USE_SVD = True and also with USE_SVD = False

USE_SVD = True
DO_BOOTSTRAP = True
DO_ERRORBUDGET = True

SVDCUT = 1e-15 if USE_SVD else 1e-19

import sys
import tee
import lsqfit
import numpy as np
import gvar as gd

def f_exact(x):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

def f(x,p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data(p):                   # make x,y fit data
    x = np.arange(1.,10*0.2+1.,0.2)
    ymod = f_exact(x)-f(x,p)
    # ymod = gd.rebuild(ymod)
    return x,ymod

def make_prior(nexp):               # make priors for fit parameters
    prior = lsqfit.GPrior()         # Gaussian prior -- dictionary-like
    prior['a'] = [gd.gvar(0.5,0.5) for i in range(nexp)]
    de = [gd.gvar(0.9,0.2) for i in range(nexp)]
    de[0] = gd.gvar(1,0.5)
    prior['E'] = [sum(de[:i+1]) for i in range(nexp)]
    return prior

def main():
    gd.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    max_prior = make_prior(20)      # maximum sized prior
    p0 = None                       # make larger fits go faster (opt.)
    sys_stdout = sys.stdout
    if USE_SVD:
        sys.stdout = tee.tee(sys_stdout,open("eg5a.out","w"))
    for nexp in range(1,5):
        print '************************************* nexp =',nexp
        fit_prior = lsqfit.GPrior()     # prior used in fit
        ymod_prior = lsqfit.GPrior()    # part of max_prior absorbed in ymod
        for k in max_prior:
            fit_prior[k] = max_prior[k][:nexp]
            ymod_prior[k] = max_prior[k][nexp:]
        x,y = make_data(ymod_prior)     # make fit data
        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=fit_prior,p0=p0,svdcut=SVDCUT,maxit=10000)
        if nexp==4 and not USE_SVD:
            sys.stdout = tee.tee(sys_stdout, open("eg5b.out", "w"))
        print fit.format(100)                   # print the fit results
        # if nexp>3:
        #     E = fit.p['E']              # best-fit parameters
        #     a = fit.p['a']
        #     print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
        #     print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
            # E = fit.palt['E']              # best-fit parameters
            # a = fit.palt['a']
            # print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
            # print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
        print
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
    E = fit.p['E']              # best-fit parameters
    a = fit.p['a']
    print 'E1/E0 =',(E[1]/E[0]).fmt(),'  E2/E0 =',(E[2]/E[0]).fmt()
    print 'a1/a0 =',(a[1]/a[0]).fmt(),'  a2/a0 =',(a[2]/a[0]).fmt()
    sys.stdout = sys_stdout

    if DO_ERRORBUDGET:
        if USE_SVD:
            sys.stdout = tee.tee(sys_stdout,open("eg5d.out","w"))
        outputs = {'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],
                 'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]}
        inputs = {'E':max_prior['E'],'a':max_prior['a'],'svd':fit.svdcorrection}
        print fit.fmt_values(outputs)
        print fit.fmt_errorbudget(outputs,inputs)
        sys.stdout = sys_stdout
        outputs = {''}

    if DO_BOOTSTRAP:
        Nbs = 40                                     # number of bootstrap copies
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
            # print bsfit
        # extract means and "standard deviations" from the bootstrap output
        outputs = gd.dataset.avg_data(outputs,bstrap=True)
        # for k in outputs:
        #     outputs[k] = gd.gvar(np.mean(outputs[k]),np.std(outputs[k]))
        if USE_SVD:
            sys.stdout = tee.tee(sys_stdout,open("eg5e.out","w"))
        print 'Bootstrap results:'
        print 'E1/E0 =',outputs['E1/E0'].fmt(),'  E2/E1 =',outputs['E2/E0'].fmt()
        print 'a1/a0 =',outputs['a1/a0'].fmt(),'  a2/a0 =',outputs['a2/a0'].fmt()
        print 'E1 =',outputs['E1'].fmt(),'  a1 =',outputs['a1'].fmt()

    # print fit.format(100)                   # print the fit results

    # import pylab as pp
    # from gvar import mean,sdev
    # fity = f(x,fit.pmean)
    # ratio = y/fity
    # pp.xlim(0,21)
    # pp.xlabel('x')
    # pp.ylabel('y/f(x,p)')
    # pp.errorbar(x=x,y=mean(ratio),yerr=sdev(ratio),fmt='ob')
    # pp.plot([0.0,21.0],[1.0,1.0])
    # pp.show()

if __name__ == '__main__':
    main()
