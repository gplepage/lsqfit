#!/usr/bin/env python
"""
eg6.py - Code for "Positive Parameters; Non-Gaussian Priors"

Created by Peter Lepage on 2013-02-12.
Copyright (c) 2013-2021 G. Peter Lepage. All rights reserved.
"""

from __future__ import print_function   # makes this work for python2 and 3
from gvar import *
from lsqfit import nonlinear_fit
import lsqfit
import functools
import inspect
import sys
import vegas
from outputsplitter import log_stdout, unlog_stdout

DO_PLOT =  True

if False:
  ranseed(12345)
  ygen = gvar(0.015, 0.2)
  print(ygen)

  # y = [gi for gi in bootstrap_iter(ygen, 20)]
  y = [gvar(ygen(), ygen.sdev) for i in range(16000)]
  ystr = [yi.fmt(2) for yi in y]
  y = gvar(fmt(y, 2))
else:
  y = gvar([
   '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
   '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
   '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
   '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
   ])

# print (y)

print()
log_prior = BufferDict()
log_prior['log(a)'] = log(gvar(0.02, 0.02))
sqrt_prior = BufferDict()
sqrt_prior['sqrt(a)'] = sqrt(gvar(0.02, 0.02))
prior = BufferDict(a = gvar(0.02, 0.02))
unif_prior = BufferDict()
unif_prior['ga(a)'] = BufferDict.uniform('ga', 0, 0.04)

for p in [prior, log_prior, sqrt_prior, unif_prior]:
    key = list(p.keys())[0].replace('(a)','_a')
    log_stdout("eg6-{}.out".format(key))
    def fcn(p, N=len(y)):
	    return N*[p['a']]
    fit = nonlinear_fit(prior=p, fcn=fcn, data=(y))
    print (fit)
    print ("a =", fit.p['a'])
    unlog_stdout()


if DO_PLOT:
    log_stdout('eg6-hist.out')
    print(fit)
    a = fit.p['a']
    print('a =', a)
    import matplotlib.pyplot as plt
    fs = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = [fs[0] * 0.7, fs[1] * 0.7]
    plt.rcParams['figure.autolayout'] = True
    a = fit.p['a']

    hist = PDFHistogram(a, nbin=16, binwidth=0.5)

    def g(p):
        return hist.count(p['a'])

    expval = vegas.PDFIntegrator(fit.p, pdf=fit.pdf)
    expval(neval=1009, nitn=10)

    count = expval(g, neval=1000, nitn=10, adapt=False)
    print('\nHistogram Analysis:')
    print (hist.analyze(count).stats)
    hist.make_plot(count, plot=plt)
    plt.xlabel('a')
    plt.savefig('eg6.png', bbox_inches='tight')
    plt.show()

