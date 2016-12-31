#!/usr/bin/env python
# encoding: utf-8
"""
eg7.py - Code for "y has Unknown Errors"

Created by Peter Lepage on 2010-01-04.
Copyright (c) 2016 G. Peter Lepage. All rights reserved.
"""

import sys
import tee
import lsqfit
import numpy as np
import gvar as gv

MAKE_PLOT = False

if MAKE_PLOT:
    import matplotlib.pyplot as plt

def main():
    sys_stdout = sys.stdout

    # version 1 - relative errors
    sys.stdout = tee.tee(sys_stdout, open("eg7a.out","w"))

    # fit data and prior
    x = np.array([1., 2., 3., 4.])
    y = np.array([3.4422, 1.2929, 0.4798, 0.1725])
    prior = gv.gvar(['10(1)', '1.0(1)'])

    # fit function
    def fcn(x, p):
        return p[0] * gv.exp( - p[1] * x)

    # find optimal dy
    def fitargs(z):
        dy = y * z
        newy = gv.gvar(y, dy)
        return dict(data=(x, newy), fcn=fcn, prior=prior)

    fit, z = lsqfit.empbayes_fit(0.001, fitargs)
    print fit.format(True)
    if MAKE_PLOT:
        ratio = fit.y / fcn(x, fit.pmean)
        plt.errorbar(x=fit.x, y=gv.mean(ratio), yerr=gv.sdev(ratio), c='b')
        plt.plot([0.5, 4.5], [1.0, 1.0], c='r')

    # version 2 - additive errors
    sys.stdout = tee.tee(sys_stdout, open("eg7b.out","w"))

    def fitargs(z):
        dy =  np.ones_like(y) * z
        newy = gv.gvar(y, dy)
        return dict(data=(x, newy), fcn=fcn, prior=prior)

    fit, z = lsqfit.empbayes_fit(0.001, fitargs)
    print fit.format(True)

    if MAKE_PLOT:
        ratio = fit.y / fcn(x, fit.pmean)
        plt.errorbar(x=fit.x + 0.1, y=gv.mean(ratio), yerr=gv.sdev(ratio), c='g')
        plt.show()

    # version 3 - tuning a prior

if __name__ == '__main__':
    main()

