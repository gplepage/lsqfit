#!/usr/bin/env python
# encoding: utf-8
"""
eg3.py - Code for "Correlated Parameters"

Created by Peter Lepage in 2016.
Copyright (c) 2016-2021 Cornell University. All rights reserved.
"""
DO_PLOT = True
DO_BOOTSTRAP = True
DO_BAYESIAN = True
DO_SIMULATION = True

import collections
import sys
import tee

import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import vegas

# lsqfit.nonlinear_fit.set_defaults(alg='subspace2D', solver='cholesky')

# generates warning which I ignore
plt.rcParams['figure.autolayout'] = True

def main():
    sys_stdout = sys.stdout
    sys.stdout = tee.tee(sys.stdout, open("eg3a.out","w"))
    x, y = make_data()
    prior = make_prior()
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(fit)
    print('p1/p0 =', fit.p[1] / fit.p[0], '    p3/p2 =', fit.p[3] / fit.p[2])
    print('corr(p0,p1) =', gv.evalcorr(fit.p[:2])[1,0])

    if DO_PLOT:
        plt.semilogx()
        plt.errorbar(
            x=gv.mean(x), xerr=gv.sdev(x), y=gv.mean(y), yerr=gv.sdev(y),
            fmt='ob'
            )
        # plot fit line
        xx = np.linspace(0.99 * gv.mean(min(x)), 1.01 * gv.mean(max(x)), 100)
        yy = fcn(xx, fit.pmean)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(xx, yy, ':r')
        plt.savefig('eg3.png', bbox_inches='tight')
        plt.show()

    sys.stdout = sys_stdout
    if DO_BOOTSTRAP:
        gv.ranseed(123)
        sys.stdout = tee.tee(sys_stdout, open('eg3c.out', 'w'))
        print(fit)
        print('p1/p0 =', fit.p[1] / fit.p[0], '    p3/p2 =', fit.p[3] / fit.p[2])
        print('corr(p0,p1) =', gv.evalcorr(fit.p[:2])[1,0])
        Nbs = 40
        outputs = {'p':[], 'p1/p0':[], 'p3/p2':[]}
        for bsfit in fit.bootstrap_iter(n=Nbs):
            p = bsfit.pmean
            outputs['p'].append(p)
            outputs['p1/p0'].append(p[1] / p[0])
            outputs['p3/p2'].append(p[3] / p[2])
        print('\nBootstrap Averages:')
        outputs = gv.dataset.avg_data(outputs, bstrap=True)
        print(gv.tabulate(outputs))
        print('corr(p0,p1) =', gv.evalcorr(outputs['p'][:2])[1,0])

        # make histograms of p1/p0 and p3/p2
        sys.stdout = sys_stdout
        print
        sys.stdout = tee.tee(sys_stdout, open('eg3d.out', 'w'))
        print('Histogram Analysis:')
        count = {'p1/p0':[], 'p3/p2':[]}
        hist = {
            'p1/p0':gv.PDFHistogram(fit.p[1] / fit.p[0]),
            'p3/p2':gv.PDFHistogram(fit.p[3] / fit.p[2]),
            }
        for bsfit in fit.bootstrap_iter(n=1000):
            p = bsfit.pmean
            count['p1/p0'].append(hist['p1/p0'].count(p[1] / p[0]))
            count['p3/p2'].append(hist['p3/p2'].count(p[3] / p[2]))
        count = gv.dataset.avg_data(count)
        plt.rcParams['figure.figsize'] = [6.4, 2.4]
        pltnum = 1
        for k in count:
            print(k + ':')
            print(hist[k].analyze(count[k]).stats)
            plt.subplot(1, 2, pltnum)
            plt.xlabel(k)
            hist[k].make_plot(count[k], plot=plt)
            if pltnum == 2:
                plt.ylabel('')
            pltnum += 1
        plt.rcParams['figure.figsize'] = [6.4, 4.8]
        plt.savefig('eg3d.png', bbox_inches='tight')
        plt.show()

    if DO_BAYESIAN:
        gv.ranseed(123)
        sys.stdout = tee.tee(sys_stdout, open('eg3e.out', 'w'))
        print(fit)
        expval = vegas.PDFIntegrator(fit.p, pdf=fit.pdf)

        # adapt integrator to PDF from fit
        neval = 1000
        nitn = 10
        expval(neval=neval, nitn=nitn)

        # <g(p)> gives mean and covariance matrix, and histograms
        hist = [
            gv.PDFHistogram(fit.p[0]), gv.PDFHistogram(fit.p[1]),
            gv.PDFHistogram(fit.p[2]), gv.PDFHistogram(fit.p[3]),
            ]
        def g(p):
            return dict(
                mean=p,
                outer=np.outer(p, p),
                count=[
                    hist[0].count(p[0]), hist[1].count(p[1]),
                    hist[2].count(p[2]), hist[3].count(p[3]),
                    ],
                )

        # evaluate expectation value of g(p)
        results = expval(g, neval=neval, nitn=nitn, adapt=False)

        # analyze results
        print('\nIterations:')
        print(results.summary())
        print('Integration Results:')
        pmean = results['mean']
        pcov =  results['outer'] - np.outer(pmean, pmean)
        print('    mean(p) =', pmean)
        print('    cov(p) =\n', pcov)

        # create GVars from results
        p = gv.gvar(gv.mean(pmean), gv.mean(pcov))
        print('\nBayesian Parameters:')
        print(gv.tabulate(p))
        print('\nlogBF =', np.log(results.pdfnorm) - fit.pdf.lognorm)


        # show histograms
        print('\nHistogram Statistics:')
        count = results['count']
        for i in range(4):
            print('p[{}] -'.format(i))
            print(hist[i].analyze(count[i]).stats)
            plt.subplot(2, 2, i + 1)
            plt.xlabel('p[{}]'.format(i))
            hist[i].make_plot(count[i], plot=plt)
            if i % 2 != 0:
                plt.ylabel('')
        plt.savefig('eg3e.png', bbox_inches='tight')
        plt.show()

    if DO_SIMULATION:
        gv.ranseed(1234)
        sys.stdout = tee.tee(sys_stdout, open('eg3f.out', 'w'))
        print(40 * '*' + ' real fit')
        print(fit.format(True))

        Q = []
        p = []
        for sfit in fit.simulated_fit_iter(n=3, add_priornoise=False):
            print(40 * '=' + ' simulation')
            print(sfit.format(True))
            diff = sfit.p - sfit.pexact
            print('\nsfit.p - pexact =', diff)
            print(gv.fmt_chi2(gv.chi2(diff)))
            print

    # omit constraint
    sys.stdout = tee.tee(sys_stdout, open("eg3b.out", "w"))
    prior = gv.gvar(4 * ['0(1)'])
    prior[1] = gv.gvar('0(20)')
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(fit)
    print('p1/p0 =', fit.p[1] / fit.p[0], '    p3/p2 =', fit.p[3] / fit.p[2])
    print('corr(p0,p1) =', gv.evalcorr(fit.p[:2])[1,0])


def make_data():
    x = np.array([
        4.    ,  2.    ,  1.    ,  0.5   ,  0.25  ,  0.167 ,  0.125 ,
        0.1   ,  0.0833,  0.0714,  0.0625
        ])
    y = gv.gvar([
        '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
        '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
        '0.044(16)'
        ])
    return x, y

def make_prior():
    b = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
    # b[1] is almost equal to 20 * b[0]
    b[1] = 20 * b[0] + gv.gvar('0.0(1)')
    return b

def fcn(x, b):
    return (b[0] * (x**2 + b[1] * x)) / (x**2 + x * b[2] + b[3])

if __name__ == '__main__':
    main()
