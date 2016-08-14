#!/usr/bin/env python
# encoding: utf-8
"""
eg4.py - Code for "Tuning Priors and the Empirical Bayes Criterion" and
              for "Partial Errors and Error Budgets"

Created by Peter Lepage on 2010-01-04.
Copyright (c) 2010 Cornell University. All rights reserved.
"""
DO_EMPBAYES = True
DO_ERRORBUDGET = True
DO_SIMULATIONS = True
DO_PLOT = False

import sys
import lsqfit
import numpy as np
import gvar as gv
import tee

def f_exact(x):                     # exact f(x)
    return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

def f(x,p):                         # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data(N=100, eps=0.01):                    # make x,y fit data
    x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])[4:]
    cr = gv.gvar(0.0,eps)
    c = [gv.gvar(cr(),cr.sdev) for n in range(N)]
    x_xmax = x/max(x)
    noise = 1+ sum(c[n]*x_xmax**n for n in range(N))
    y = f_exact(x)*noise            # noisy y[i]s
    return x,y

def make_prior(nexp):               # make priors for fit parameters
    prior = dict()         # Gaussian prior -- dictionary-like
    prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
    de = [gv.gvar(0.9,0.01) for i in range(nexp)]
    de[0] = gv.gvar(1,0.5)
    prior['E'] = [sum(de[:i+1]) for i in range(nexp)]
    return prior

def main():
    gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
    x,y = make_data()               # make fit data
    p0 = None                       # make larger fits go faster (opt.)
    for nexp in range(3,5):
        print '************************************* nexp =',nexp
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior,p0=p0)
        print fit                   # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
        print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
        print
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
    sys_stdout = sys.stdout
    if DO_ERRORBUDGET:

        lines = [
            "E = fit.p['E']",
            "a = fit.p['a']",
            "print(E[1] / E[0])",
            "print((E[1] / E[0]).partialsdev(fit.prior['E']))",
            "print((E[1] / E[0]).partialsdev(fit.prior['a']))",
            "print((E[1] / E[0]).partialsdev(y))"
            ]
        sys.stdout = tee.tee(sys_stdout, open("eg4c.out","w"))
        for line in lines:
            print ">>>", line
            if line[:5] == "print":
                print(eval(line[5:]))
        # print E[1]/E[0]
        # print (E[1]/E[0]).partialsdev(fit.prior['E'])
        # print (E[1]/E[0]).partialsdev(fit.prior['a'])
        # print (E[1]/E[0]).partialsdev(y)
        outputs = {'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],
                 'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]}
        inputs = {'E':fit.prior['E'],'a':fit.prior['a'],'y':y}

        sys.stdout = tee.tee(sys_stdout, open("eg4b.out","w"))
        print fit.fmt_values(outputs)
        print fit.fmt_errorbudget(outputs,inputs)
        sys.stdout = sys_stdout

    if DO_SIMULATIONS:
        # fit simulations
        sys.stdout = tee.tee(sys_stdout, open("eg4d.out","w"))

        for sfit in fit.simulated_fit_iter(3):
            print '************************************* simulation'
            print(sfit)
            sE = sfit.p['E']             # best-fit parameters
            sa = sfit.p['a']
            E = sfit.pexact['E']
            a = sfit.pexact['a']
            print 'E1/E0 =', sE[1] / sE[0], '  E2/E0 =', sE[2] / sE[0]
            print 'a1/a0 =', sa[1] / sa[0], '  a2/a0 =', sa[2] / sa[0]
            print '\nSimulated Fit Values - Exact Values:'
            print 'E1/E0:', (sE[1] / sE[0]) - (E[1] / E[0]),\
               '  E2/E0:', (sE[2] / sE[0]) - (E[2] / E[0])
            print 'a1/a0:', (sa[1] / sa[0]) - (a[1] / a[0]),\
               '  a2/a0:', (sa[2] / sa[0]) - (a[2] / a[0])

            # compute chi**2 comparing fit results to exact results
            sim_results = [sE[0], sE[1], sa[0], sa[1]]
            exact_results = [E[0], E[1], a[0], a[1]]
            chi2 = gv.chi2(sim_results, exact_results, svdcut=1e-8)
            print '\nParameter chi2/dof [dof] = %.2f' % (chi2/chi2.dof), '[%d]' % chi2.dof, '  Q = %.1f' % chi2.Q
            print
        sys.stdout = sys_stdout

    if DO_EMPBAYES:
        def fitargs(z,nexp=nexp,prior=prior,f=f,data=(x,y),p0=p0):
            z = gv.exp(z)
            prior['a'] = [gv.gvar(0.5,0.5*z[0]) for i in range(nexp)]
            return dict(prior=prior,data=data,fcn=f,p0=p0)
        ##
        z0 = [0.0]
        fit,z = lsqfit.empbayes_fit(z0,fitargs,tol=1e-3)
        sys.stdout = tee.tee(sys_stdout, open("eg4a.out","w"))
        print fit                   # print the optimized fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
        print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
        # print "prior['a'] =",fit.prior['a'][0]
        sys.stdout = sys_stdout
        print

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
