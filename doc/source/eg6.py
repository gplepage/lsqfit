#!/usr/bin/env python
# encoding: utf-8
"""
eg6.py - Code for "Positive Parameters; Non-Gaussian Priors"

Created by Peter Lepage on 2013-02-12.
Copyright (c) 2013 G. Peter Lepage. All rights reserved.
"""

from __future__ import print_function   # makes this work for python2 and 3
from gvar import *
from lsqfit import nonlinear_fit
import lsqfit
import functools
import inspect
import sys
import tee

DO_PLOT = True
sys_stdout = sys.stdout

# ranseed(12345)
# ygen = gvar(0.015, 0.2)
# print(ygen)

# y = [gi for gi in bootstrap_iter(ygen, 20)]
# y = [gvar(ygen(), ygen.sdev) for i in range(20)]
# ystr = [yi.fmt(2) for yi in y]
# y = gvar(fmt(y, 2))
y = gvar([
   '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
   '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
   '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
   '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
   ])

# print (y)

print
log_prior = BufferDict()
log_prior['log(a)'] = log(gvar(0.02, 0.02))
sqrt_prior = BufferDict()
sqrt_prior['sqrt(a)'] = sqrt(gvar(0.02, 0.02))
prior = BufferDict(a = gvar(0.02, 0.02))

stdout = sys.stdout
for p in [prior, log_prior, sqrt_prior]:
	key = list(p.keys())[0].replace('(a)','_a')
	sys.stdout = tee.tee(sys_stdout, open("eg6-{}.out".format(key), "w"))
	def fcn(p, N=len(y)):
		return N*[p['a']]
	f = nonlinear_fit(prior=p, fcn=fcn, data=(y))
	print (f)
	print ("a =", f.p['a'])

sys.stdout = tee.tee(sys_stdout, open("eg6-erfinv.out", "w"))
prior = BufferDict()
prior['erfinv(50a-1)'] = gvar('0(1)') / sqrt(2)

def fcn(p, N=len(y)):
  a = 0.02 + 0.02 * p['50a-1']
  return N * [a]

fit = nonlinear_fit(prior=prior, data=y, fcn=fcn)
print(fit)
print('a =', (0.02 + 0.02 * fit.p['50a-1']))                 # exp(log(a))

if DO_PLOT:
    sys.stdout = tee.tee(sys_stdout, open('eg6-hist.out', 'w'))
    print(fit)
    a = (1+fit.p['50a-1']) / 50
    print('a =', a)
    import matplotlib.pyplot as plt
    fs = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = [fs[0] * 0.7, fs[1] * 0.7]
    plt.rcParams['figure.autolayout'] = True
    a = (1+fit.p['50a-1']) / 50

    hist = PDFHistogram(a, nbin=16, binwidth=0.5)

    def g(p):
        return hist.count((1+p['50a-1']) / 50)

    expval = lsqfit.BayesIntegrator(fit)
    expval(neval=1009, nitn=10)

    count = expval(g, neval=1000, nitn=10, adapt=False)
    print('\nHistogram Analysis:')
    print (hist.analyze(count).stats)
    hist.make_plot(count, plot=plt)
    plt.xlabel('a')
    plt.savefig('eg6.png', bbox_inches='tight')
    plt.show()

sys.stdout = stdout



# import lsqfit

# def invf(x):
#     return 0.02 + 0.02 * tanh(x)
# def f(x):
#     return arctanh((x - 0.02) / 0.02)

# lsqfit.add_parameter_distribution('f', invf)
# intv_prior = BufferDict()
# intv_prior['f(a)'] = gvar(0,0.75)
# fit = nonlinear_fit(prior=intv_prior, fcn=fcn, data=y)
# a = fit.p['a']
# fa = fit.p['f(a)']
# print (fit.format())
# print (invf(fa), fit.p['a'])
# print (f(a), fit.p['f(a)'])
