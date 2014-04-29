#!/usr/bin/env python
# encoding: utf-8
"""
eg1.py - Code for "Making Fake Data" and "Basic Fits"

Created by Peter Lepage on 2010-01-04.
Copyright (c) 2010 Cornell University. All rights reserved.
"""
DO_PLOT = False
DO_BOOTSTRAP = False
SVDCUT = (1e-14, 1e-14)

import sys
import tee
import lsqfit
import numpy as np
import gvar as gv

def f_exact(x, nterm=100):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(nterm))

def f(x,p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def f1(x,p):                         # function used to fit x,y data
    a = p['a'][:1]                      # array of a[i]s
    E = p['E'][:1]                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data(nterm=100, eps=0.01):                    # make x,y fit data
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
    cr = gv.gvar(0.0, eps)
    c = [gv.gvar(cr(), eps) for n in range(100)]
    x_xmax = x / max(x)
    noise = 1 + sum(c[n] * x_xmax ** n for n in range(100))
    y = f_exact(x, nterm)*noise            # noisy y[i]s
    return x,y

def make_prior(nexp):               # make priors for fit parameters
    prior = gv.BufferDict()         # Gaussian prior -- dictionary-like
    prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
    prior['E'] = [gv.gvar(i+1,0.5) for i in range(nexp)]
    return prior

def main():
    gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    x,y = make_data()               # make fit data
    p0 = None                       # make larger fits go faster (opt.)
    sys_stdout = sys.stdout
    sys.stdout = tee.tee(sys.stdout, open("eg1.out","w"))
    for nexp in range(3,20):
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior,p0=p0) #, svdcut=SVDCUT)
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
        if nexp in [8, 9, 10]:
            print(".".center(73))
        if nexp > 7 and nexp < 19:
            continue
        elif nexp not in [3]:
            print("")
        print '************************************* nexp =',nexp
        print fit.format()                   # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
        print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]

    # extra data 1
    print '\n--------------------- fit with extra information'
    sys.stdout = tee.tee(sys_stdout, open("eg1a.out", "w"))
    def ratio(p):
        return p['a'][1] / p['a'][0]
    newfit = lsqfit.nonlinear_fit(data=gv.gvar(1,1e-5), fcn=ratio, prior=fit.p)
    print (newfit)
    # print(newfit.p['a'][1] / newfit.p['a'][0])
    # print(fit.p['a'][1] / fit.p['a'][0])

    # alternate method for extra data
    sys.stdout = tee.tee(sys_stdout, open("eg1b.out", "w"))
    fit.p['a1/a0'] = fit.p['a'][1] / fit.p['a'][0]
    new_data = {'a1/a0' : gv.gvar(1,1e-5)}
    new_p = lsqfit.wavg([fit.p, new_data])
    print 'chi2/dof = %.2f\n' % (new_p.chi2 / new_p.dof)
    print 'E:', new_p['E'][:4]
    print 'a:', new_p['a'][:4]
    print 'a1/a0:', new_p['a1/a0']

    # # extra data 2
    # sys.stdout = tee.tee(sys_stdout, open("eg1b.out", "w"))
    # newfit = fit
    # for i in range(1):
    #     print '\n--------------------- fit with %d extra data sets' % (i+1)
    #     x, ynew = make_data()
    #     prior = newfit.p
    #     newfit = lsqfit.nonlinear_fit(data=(x,ynew), fcn=f, prior=prior) # , svdcut=SVDCUT)
    #     print newfit
    sys.stdout = sys_stdout
    # def fcn(x, p):
    #     return f(x, p), f(x, p)
    # prior = make_prior(nexp)
    # fit = lsqfit.nonlinear_fit(data=(x, [y, ynew]), fcn=fcn, prior=prior, p0=newfit.pmean) # , svdcut=SVDCUT)
    # print(fit)


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
            # print bsfit.chi2/bsfit.dof

        # extract means and standard deviations from the bootstrap output
        for k in outputs:
            outputs[k] = gv.gvar(np.mean(outputs[k]),np.std(outputs[k]))
        print 'Bootstrap results:'
        print 'E1/E0 =',outputs['E1/E0'],'  E2/E1 =',outputs['E2/E0']
        print 'a1/a0 =',outputs['a1/a0'],'  a2/a0 =',outputs['a2/a0']
        print 'E1 =',outputs['E1'],'  a1 =',outputs['a1']
        
    if DO_PLOT:
        import pylab as plt   
        ratio = y/fit.fcn(x,fit.pmean)
        plt.xlim(0,21)
        plt.xlabel('x')
        plt.ylabel('y/f(x,p)')
        plt.errorbar(x=x,y=gv.mean(ratio),yerr=gv.sdev(ratio),fmt='ob')
        plt.plot([0.0,21.0],[1.0,1.0])
        plt.show()

if __name__ == '__main__':
    main()
