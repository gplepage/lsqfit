#!/usr/bin/env python
# encoding: utf-8
"""
eg5.py - Code for "y has No Error Bars" and "SVD Cuts and Roundoff Error"

Created by Peter Lepage on 2013-02-12.
Copyright (c) 2013 G. Peter Lepage. All rights reserved.
"""

from __future__ import print_function   # makes this work for python2 and 3
from gvar import *
from lsqfit import nonlinear_fit
import functools
import inspect
import sys

ranseed([123])
ygen = gvar(0.02, 0.2) - 0.005
print (ygen)

y = [gvar(ygen(), ygen.sdev) for i in range(20)]
ystr = [yi.fmt(2) for yi in y]
y = gvar(ystr)
print (ystr[:])

print
log_prior = BufferDict()
log_prior['log(a)'] = log(gvar(0.02, 0.02))
sqrt_prior = BufferDict()
sqrt_prior['sqrt(a)'] = sqrt(gvar(0.02, 0.02))
prior = BufferDict(a = gvar(0.02, 0.02))

stdout = sys.stdout
for p in [prior, log_prior, sqrt_prior]:
	key = list(p.keys())[0]
	sys.stdout = open("eg6-{}.out".format(key), "w")
	def fcn(p, N=len(y)):
		return N*[p['a']]
	f = nonlinear_fit(prior=p, fcn=fcn, data=(y), extend=True)
	print (f)
	print ("a =", f.p['a'].fmt())

sys.stdout = open("eg6-erfinv.out", "w")
prior = BufferDict()
prior['erfinv(50a-1)'] = gvar('0(1)') / sqrt(2)

def fcn(p, N=len(y)):
  a = 0.02 + 0.02 * p['50a-1']
  return N * [a]

fit = nonlinear_fit(prior=prior, data=y, fcn=fcn, extend=True)
print(fit)
print('a =', (0.02 + 0.02 * fit.p['50a-1']).fmt())                 # exp(log(a))


sys.stdout = stdout



# import lsqfit

# def invf(x):
#     return 0.02 + 0.02 * tanh(x)
# def f(x):
#     return arctanh((x - 0.02) / 0.02)

# lsqfit.add_parameter_distribution('f', invf)
# intv_prior = BufferDict()
# intv_prior['f(a)'] = gvar(0,0.75)
# fit = nonlinear_fit(prior=intv_prior, fcn=fcn, data=y, extend=True)
# a = fit.p['a']
# fa = fit.p['f(a)']
# print (fit.format())
# print (invf(fa), fit.p['a'])
# print (f(a), fit.p['f(a)'])
