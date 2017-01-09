#!/usr/bin/env python
# encoding: utf-8
"""
eg9.py - More code for "chained fits"

Created by Peter Lepage on 2017-01
Copyright (c) 2017 G. Peter Lepage. All rights reserved.
"""
from __future__ import print_function

import sys

sys_stdout = sys.stdout

import numpy as np
import gvar as gv
import lsqfit
import tee

def make_fake_data(x, p, f):
    f = f(x, p)
    ans = np.empty(len(f), object)
    df = gv.gvar(len(f) * ['0.000(1)'])
    eps = gv.gvar(0,0.01)
    ans = np.array([fi + dfi() + dfi for fi,dfi in zip(f,df)]) +  eps + eps()
    # following line makes it run a lot slower, because underlying cov much bigger
    # ans = gv.gvar(gv.gvar(ans), gv.evalcov(ans))
    return ans

gv.ranseed(123456)

def f(x,p):
    return p[0] + p[1] * np.exp(- p[2] * x)

p0 =[0.5, 0.4, 0.7 ]

N = 10000
x = np.linspace(0.2, 1.0, N)
y = make_fake_data(x, p0, f)

sys.stdout = tee.tee(sys_stdout, open('eg9a.out', 'w'))
print('x = [{}  {} ... {}]'.format(x[0], x[1], x[-1]))
print('y = [{}  {} ... {}]'.format(y[0], y[1], y[-1]))
print('corr(y[0],y[9999]) =', gv.evalcorr([y[0], y[-1]])[1,0])
print()

# fit function and prior
def fcn(x, p):
    return p[0] + p[1] * np.exp(- p[2] * x)
prior = gv.gvar(['0(1)', '0(1)', '0(1)'])

# Nstride fits, each to nfit data points
nfit = 100
Nstride = len(y) // nfit
fit_time = 0.0
for n in range(0, Nstride):
    fit = lsqfit.nonlinear_fit(
        data=(x[n::Nstride], y[n::Nstride]), prior=prior, fcn=fcn
        )
    prior = fit.p
    fit_time += fit.time
    if n in [0, 9]:
        print('******* Results from ', (n+1) * nfit, 'data points')
        print(fit)
print('******* Results from ', Nstride * nfit, 'data points (final)')
print(fit)

sys.stdout = sys_stdout

print('fit time =', fit_time, '   # of points =', Nstride * nfit)

if False:
    print()
    sys.stdout = tee.tee(sys_stdout, open('eg9b.out', 'w'))

    prior = gv.gvar(['0(1)', '0(1)', '0(1)'])
    y = gv.gvar(gv.mean(y), gv.sdev(y))

    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=fcn)
    print('******* Results from uncorrelated fit')
    print(fit)

"""
******* Results from  2000 data points (final)
Least Square Fit:
  chi2/dof [dof] = 1 [2000]    Q = 0.079    logGBF = 10908

Parameters:
              0    0.519 (10)     [  0.0 (1.0) ]
              1   0.4005 (18)     [  0.0 (1.0) ]
              2   0.6978 (51)     [  0.0 (1.0) ]

Settings:
  svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 12/11.4)

fit time = 0.0    # of points = 2000

"""
