#!/usr/bin/env python
# encoding: utf-8
"""
eg1.py - Code for "Basic Fits"

Created by Peter Lepage on 2016-12.
Copyright (c) 2016 Cornell University. All rights reserved.
"""
DO_PLOT = True
DO_BAYES = False        # should be False
DO_BOOTSTRAP = False    # should be False

import collections
import sys
import tee
import lsqfit
import numpy as np
import gvar as gv

def fcn(x,p):                       # function used to fit x,y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def f1(x,p):                         # function used to fit x,y data
    a = p['a'][:1]                      # array of a[i]s
    E = p['E'][:1]                      # array of E[i]s
    return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

def make_data():
    x = np.array([  5.,   6.,   7.,   8.,   9.,  10.,  12.,  14.])
    ymean = np.array(
        [  4.5022829417e-03,   1.8170543788e-03,   7.3618847843e-04,
           2.9872730036e-04,   1.2128831367e-04,   4.9256559129e-05,
           8.1263644483e-06,   1.3415253536e-06]
        )
    ycov = np.array([
        [  2.1537808808e-09,   8.8161794696e-10,   3.6237356558e-10,
           1.4921344875e-10,   6.1492842463e-11,   2.5353714617e-11,
           4.3137593878e-12,   7.3465498888e-13],
        [  8.8161794696e-10,   3.6193461816e-10,   1.4921610813e-10,
           6.1633547703e-11,   2.5481570082e-11,   1.0540958082e-11,
           1.8059692534e-12,   3.0985581496e-13],
        [  3.6237356558e-10,   1.4921610813e-10,   6.1710468826e-11,
           2.5572230776e-11,   1.0608148954e-11,   4.4036448945e-12,
           7.6008881270e-13,   1.3146405310e-13],
        [  1.4921344875e-10,   6.1633547703e-11,   2.5572230776e-11,
           1.0632830128e-11,   4.4264622187e-12,   1.8443245513e-12,
           3.2087725578e-13,   5.5986403288e-14],
        [  6.1492842463e-11,   2.5481570082e-11,   1.0608148954e-11,
           4.4264622187e-12,   1.8496194125e-12,   7.7369196122e-13,
           1.3576009069e-13,   2.3914810594e-14],
        [  2.5353714617e-11,   1.0540958082e-11,   4.4036448945e-12,
           1.8443245513e-12,   7.7369196122e-13,   3.2498644263e-13,
           5.7551104112e-14,   1.0244738582e-14],
        [  4.3137593878e-12,   1.8059692534e-12,   7.6008881270e-13,
           3.2087725578e-13,   1.3576009069e-13,   5.7551104112e-14,
           1.0403917951e-14,   1.8976295583e-15],
        [  7.3465498888e-13,   3.0985581496e-13,   1.3146405310e-13,
           5.5986403288e-14,   2.3914810594e-14,   1.0244738582e-14,
           1.8976295583e-15,   3.5672355835e-16]
        ])
    return x, gv.gvar(ymean, ycov)

def make_prior(nexp):               # make priors for fit parameters
    prior = gv.BufferDict()         # Gaussian prior -- dictionary-like
    prior['a'] = [gv.gvar(0.5,0.4) for i in range(nexp)]
    prior['E'] = [gv.gvar(i+1,0.4) for i in range(nexp)]
    return prior

def main():
    x,y = make_data()               # make fit data
    # y = gv.gvar(gv.mean(y), 0.75**2 * gv.evalcov(y))
    p0 = None                       # make larger fits go faster (opt.)
    sys_stdout = sys.stdout
    sys.stdout = tee.tee(sys.stdout, open("eg1.out","w"))
    for nexp in range(1, 7):
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x,y), fcn=fcn, prior=prior, p0=p0)
        if fit.chi2/fit.dof<1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
        print '************************************* nexp =',nexp
        print fit.format()                   # print the fit results
        E = fit.p['E']              # best-fit parameters
        a = fit.p['a']
        if nexp > 2:
            print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
            print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
            print

    # error budget
    outputs = {
        'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],
        'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]
        }
    inputs = {'E':fit.prior['E'], 'a':fit.prior['a'], 'y':y}
    inputs = collections.OrderedDict()
    inputs['a'] = fit.prior['a']
    inputs['E'] = fit.prior['E']
    inputs['y'] = fit.data[1]
    print '================= Error Budget Analysis'
    print fit.fmt_values(outputs)
    print fit.fmt_errorbudget(outputs,inputs)


    sys.stdout = sys_stdout
    # print(gv.gvar(str(a[1])) / gv.gvar(str(a[0])) )
    # print(gv.evalcorr([fit.p['a'][1], fit.p['E'][1]]))
    # print(fit.format(True))

    # redo fit with 4 parameters since that is enough
    prior = make_prior(4)
    fit = lsqfit.nonlinear_fit(data=(x,y), fcn=fcn, prior=prior, p0=fit.pmean)
    sys.stdout = tee.tee(sys_stdout, open("eg1a.out", "w"))
    print '--------------------- original fit'
    print fit.format()
    E = fit.p['E']              # best-fit parameters
    a = fit.p['a']
    print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
    print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]
    print
    # extra data 1
    print '\n--------------------- new fit to extra information'
    def ratio(p):
        return p['a'][1] / p['a'][0]
    newfit = lsqfit.nonlinear_fit(data=gv.gvar(1,1e-5), fcn=ratio, prior=fit.p)
    print (newfit.format())
    E = newfit.p['E']
    a = newfit.p['a']
    print 'E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0]
    print 'a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0]

    if DO_PLOT:
        import matplotlib.pyplot as plt
        ratio = y / fit.fcn(x,fit.pmean)
        plt.xlim(4, 15)
        plt.ylim(0.95, 1.05)
        plt.xlabel('x')
        plt.ylabel('y / f(x,p)')
        plt.yticks([0.96, 0.98, 1.00, 1.02, 1.04], ['0.96', '0.98', '1.00', '1.02', '1.04'])
        plt.errorbar(x=x, y=gv.mean(ratio), yerr=gv.sdev(ratio), fmt='ob')
        plt.plot([4.0, 21.0], [1.0, 1.0], 'b:')
        plt.savefig('eg1.png', bbox_inches='tight')
        plt.show()


    # alternate method for extra data
    sys.stdout = tee.tee(sys_stdout, open("eg1b.out", "w"))
    fit.p['a1/a0'] = fit.p['a'][1] / fit.p['a'][0]
    new_data = {'a1/a0' : gv.gvar(1,1e-5)}
    new_p = lsqfit.wavg([fit.p, new_data])
    print 'chi2/dof = %.2f\n' % (new_p.chi2 / new_p.dof)
    print 'E:', new_p['E'][:4]
    print 'a:', new_p['a'][:4]
    print 'a1/a0:', new_p['a1/a0']

    if DO_BAYES:
        # Bayesian Fit
        gv.ranseed([123])
        prior = make_prior(4)
        fit = lsqfit.nonlinear_fit(data=(x,y), fcn=f, prior=prior, p0=fit.pmean)
        sys.stdout = tee.tee(sys_stdout, open("eg1c.out", "w"))
        # print fit

        expval = lsqfit.BayesIntegrator(fit, limit=10.)
        # adapt integrator to PDF
        expval(neval=40000, nitn=10)

        # calculate expectation value of function g(p)
        fit_hist = gv.PDFHistogram(fit.p['E'][0])
        def g(p):
            parameters = [p['a'][0], p['E'][0]]
            return dict(
                mean=parameters,
                outer=np.outer(parameters, parameters),
                hist=fit_hist.count(p['E'][0]),
                )
        r = expval(g, neval=40000, nitn=10, adapt=False)

        # print results
        print r.summary()
        means = r['mean']
        cov = r['outer'] - np.outer(r['mean'], r['mean'])
        print 'Results from Bayesian Integration:'
        print 'a0: mean =', means[0], '  sdev =', cov[0,0]**0.5
        print 'E0: mean =', means[1], '  sdev =', cov[1,1]**0.5
        print 'covariance from Bayesian integral =', np.array2string(cov, prefix=36 * ' ')
        print

        print 'Results from Least-Squares Fit:'
        print 'a0: mean =', fit.p['a'][0].mean, '  sdev =', fit.p['a'][0].sdev
        print 'E0: mean =', fit.p['E'][0].mean, '  sdev =', fit.p['E'][0].sdev
        print 'covariance from least-squares fit =', np.array2string(gv.evalcov([fit.p['a'][0], fit.p['E'][0]]), prefix=36*' ',precision=3)
        sys.stdout = sys_stdout

        # make histogram of E[0] probabilty
        plt = fit_hist.make_plot(r['hist'])
        plt.xlabel('$E_0$')
        plt.ylabel('probability')
        plt.savefig('eg1c.png', bbox_inches='tight')
        # plt.show()


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

if __name__ == '__main__':
    main()
