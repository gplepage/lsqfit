#!/usr/bin/env python
# encoding: utf-8
"""
y-vs-x.py - Code for "Basic Fits"
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

import lsqfit
import numpy as np
import gvar as gv

def main():
    x, y = make_data()              # collect fit data
    p0 = None                       # make larger fits go faster (opt.)
    for nexp in range(1, 7):
        print('************************************* nexp =', nexp)
        prior = make_prior(nexp)
        fit = lsqfit.nonlinear_fit(data=(x, y), fcn=fcn, prior=prior, p0=p0)
        print(fit)                  # print the fit results
        if nexp > 2:
            E = fit.p['E']          # best-fit parameters
            a = fit.p['a']
            print('E1/E0 =', E[1] / E[0], '  E2/E0 =', E[2] / E[0])
            print('a1/a0 =', a[1] / a[0], '  a2/a0 =', a[2] / a[0])
        if fit.chi2 / fit.dof < 1.:
            p0 = fit.pmean          # starting point for next fit (opt.)
        print()

    # error budget analysis
    # outputs = {
    #     'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],
    #     'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]
    #     }
    # inputs = {'E':fit.prior['E'], 'a':fit.prior['a'], 'y':y}
    outputs = gv.BufferDict()
    outputs['E2/E0'] = E[2] / E[0]
    outputs['E1/E0'] = E[1] / E[0]
    outputs['a2/a0'] = a[2] / a[0]
    outputs['a1/a0'] = a[1] / a[0]
    inputs = gv.BufferDict()
    inputs['a'] = fit.prior['a']
    inputs['y'] = y
    inputs['E'] = fit.prior['E']
    print('================= Error Budget Analysis')
    print(fit.fmt_values(outputs))
    print(fit.fmt_errorbudget(outputs,inputs))

def fcn(x, p):                      # function used to fit x, y data
    a = p['a']                      # array of a[i]s
    E = p['E']                      # array of E[i]s
    return sum(ai * np.exp(-Ei * x) for ai, Ei in zip(a, E))

def make_prior(nexp):               # make priors for fit parameters
    prior = gv.BufferDict()         # any dictionary works
    prior['a'] = [gv.gvar(0.5, 0.4) for i in range(nexp)]
    prior['E'] = [gv.gvar(i+1, 0.4) for i in range(nexp)]
    return prior

def make_data():                     # assemble fit data
   x = np.array([  5.,   6.,   7.,   8.,   9.,  10.,  12.,  14.])
   ymean = np.array(
       [  4.5022829417e-03,   1.8170543788e-03,   7.3618847843e-04,
          2.9872730036e-04,   1.2128831367e-04,   4.9256559129e-05,
          8.1263644483e-06,   1.3415253536e-06]
       )
   ycov = np.array(
       [[ 2.1537808808e-09,   8.8161794696e-10,   3.6237356558e-10,
          1.4921344875e-10,   6.1492842463e-11,   2.5353714617e-11,
          4.3137593878e-12,   7.3465498888e-13],
       [  8.8161794696e-10,   3.6193461816e-10,   1.4921610813e-10,
          6.1633547703e-11,   2.5481570082e-11,   1.0540958082e-11,
          1.8059692534e-12,   3.0985581496e-13],
       [  3.6237356558e-10,   1.4921610813e-10,   6.1710468826e-11,
          2.5572230776e-11,   1.0608148954e-11,   4.4036448945e-12,
          7.6008881270e-13,   1.3146405310e-13],
       [  1.4921344875e-10,   6.1633547703e-11,   2.5572230776e-11,
          1.0632830128e-11,   4.4264622187e-12,   1.8443245513e-12,
          3.2087725578e-13,   5.5986403288e-14],
       [  6.1492842463e-11,   2.5481570082e-11,   1.0608148954e-11,
          4.4264622187e-12,   1.8496194125e-12,   7.7369196122e-13,
          1.3576009069e-13,   2.3914810594e-14],
       [  2.5353714617e-11,   1.0540958082e-11,   4.4036448945e-12,
          1.8443245513e-12,   7.7369196122e-13,   3.2498644263e-13,
          5.7551104112e-14,   1.0244738582e-14],
       [  4.3137593878e-12,   1.8059692534e-12,   7.6008881270e-13,
          3.2087725578e-13,   1.3576009069e-13,   5.7551104112e-14,
          1.0403917951e-14,   1.8976295583e-15],
       [  7.3465498888e-13,   3.0985581496e-13,   1.3146405310e-13,
          5.5986403288e-14,   2.3914810594e-14,   1.0244738582e-14,
          1.8976295583e-15,   3.5672355835e-16]]
       )
   return x, gv.gvar(ymean, ycov)

if __name__ == '__main__':
    main()
