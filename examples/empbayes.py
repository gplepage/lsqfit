#!/usr/bin/env python
# encoding: utf-8
"""
err-budget.py - Code for "Tuning Priors and the Empirical Bayes Criterion"
"""
# Copyright (c) 2017 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from __future__ import print_function   # makes this work for python2 and 3

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

def fitargs(z):
    dp = z
    prior = gv.gvar([gv.gvar(0, dp) for i in range(4)])
    return dict(prior=prior, fcn=fcn, data=(x,y))

fit,z = lsqfit.empbayes_fit(1.0, fitargs)
print(fit.format(True))
