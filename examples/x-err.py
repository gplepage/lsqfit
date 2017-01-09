#!/usr/bin/env python
# encoding: utf-8
"""
x-err.py - Code for "x has Error Bars"
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

import gvar as gv
import lsqfit

def make_data():
    x = gv.gvar([
        '0.73(50)',   '2.25(50)',  '3.07(50)',  '3.62(50)',  '4.86(50)',
        '6.41(50)',   '6.39(50)',  '7.89(50)',  '9.32(50)',  '9.78(50)',
        '10.83(50)', '11.98(50)', '13.37(50)', '13.84(50)', '14.89(50)'
        ])
    y = gv.gvar([
         '3.85(70)',  '5.5(1.7)',  '14.0(2.6)',   '21.8(3.4)',   '47.0(5.2)',
        '79.8(4.6)', '84.9(4.6)',  '95.2(2.2)',   '97.65(79)',   '98.78(55)',
        '99.41(25)', '99.80(12)', '100.127(77)', '100.202(73)', '100.203(71)'
        ])
    return x,y

def make_prior(x):
    prior = gv.BufferDict()
    prior['b'] = gv.gvar(['0(500)', '0(5)', '0(5)', '0(5)'])
    prior['x'] = x
    return prior

def fcn(p):
    b0, b1, b2, b3 = p['b']
    x = p['x']
    return b0 / ((1. + gv.exp(b1 - b2 * x)) ** (1. / b3))

x, y = make_data()
prior = make_prior(x)
fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=fcn)
print(fit)

