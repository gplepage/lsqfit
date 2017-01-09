#!/usr/bin/env python
# encoding: utf-8
"""
p-corr.py - Code for "Correlated Parameters"
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

def main():
    x, y = make_data()
    prior = make_prior()
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(fit)
    print('p1/p0 =', fit.p[1] / fit.p[0], 'p3/p2 =', fit.p[3] / fit.p[2])
    print('corr(p0,p1) = {:.4f}'.format(gv.evalcorr(fit.p[:2])[1,0]))

def make_data():
    x = np.array([
        4., 2., 1., 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625
        ])
    y = gv.gvar([
        '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
        '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
        '0.044(16)'
        ])
    return x, y

def make_prior():
    p = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
    p[1] = 20 * p[0] + gv.gvar('0.0(1)')        # p[1] correlated with p[0]
    return p

def fcn(x, p):
    return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

if __name__ == '__main__':
    main()

