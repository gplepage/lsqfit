#!/usr/bin/env python
# encoding: utf-8
"""
uncorrelated.py -- large amounts of uncorrelated fit data

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

from __future__ import print_function

import numpy as np
import gvar as gv
import lsqfit

def make_fake_data(x, p, f):
    f = f(x, p)
    df = gv.gvar(len(f) * ['0.000(1)'])
    return np.array([fi + dfi()/2. + dfi for fi, dfi in zip(f, df)])

gv.ranseed(12)

def f(x,p):
    return p[0] + p[1] * np.exp(- p[2] * x)

def main():
    p0 =[0.5, 0.4, 0.7 ]

    N = 50000   # takes 2min to do 2000000; scales linearly
    x = np.linspace(0.2, 2., N)
    y = make_fake_data(x, p0, f)
    print('y = [{} {} ... {}]\n'.format(y[0], y[1], y[-1]))
    prior = gv.gvar(['0(1)', '0(1)', '0(1)'])
    fit = lsqfit.nonlinear_fit(udata=(x,y), prior=prior, fcn=f)
    print(fit)

if __name__ == '__main__':
  main()