#!/usr/bin/env python
# encoding: utf-8
"""
eg4.py - Code for "Tuning Priors with the Empirical Bayes Criterion"

Created by Peter Lepage on 2016-12.
Copyright (c) 2016 Cornell University. All rights reserved.
"""

import sys
import tee
sys_stdout = sys.stdout

import numpy as np
import gvar as gv
import lsqfit

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
y = np.array([
    '0.133426(95)', '0.20525(15)', '0.27491(20)', '0.32521(25)',
    '0.34223(28)', '0.32394(28)', '0.27857(27)'
    ])

def fcn(x, p):
    return gv.exp(-p[0] - p[1] * x - p[2] * x**2 - p[3] * x**3)

def fitterargs(z):
    prior = gv.gvar([gv.gvar(0, z[0]**2) for i in range(4)])
    return dict(prior=prior, fcn=fcn, data=(x,y))

fit,z = lsqfit.empbayes_fit([0.01, 0.01, 0.01, 0.01], fitterargs)

sys.stdout = tee.tee(sys_stdout, open('eg4a.out', 'w'))
print fit.format(True)


# 3 vs 4 terms
sys.stdout = tee.tee(sys_stdout, open('eg4b.out', 'w'))
y = np.array([
    '0.133213(95)', '0.20245(15)', '0.26282(19)', '0.29099(22)',
    '0.27589(22)', '0.22328(19)', '0.15436(14)'
    ])

prior = gv.gvar(4 * ['0(5.3)'])
def fcn3(x, p):
    return gv.exp(-p[0] - p[1] * x - p[2] * x**2)

fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior[:-1], fcn=fcn3)
print 10 * '=', 'fcn(x,p) = exp(-p[0] - p[1] * x - p[2] * x**2)'
print fit
print

fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=fcn)
print 10 * '=', 'fcn(x,p) = exp(-p[0] - p[1] * x - p[2] * x**2 - p[3] * x**3)'
print fit
