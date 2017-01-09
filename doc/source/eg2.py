#!/usr/bin/env python
# encoding: utf-8
"""
eg2.py - Code for "x has Error Bars"

Created by Peter Lepage.
Copyright (c) 2016 Cornell University. All rights reserved.
"""
import sys

import numpy as np
import gvar as gv
import lsqfit
import tee
import matplotlib.pyplot as plt

MAKE_PLOT = True


def main():
    sys_stdout = sys.stdout
    sys.stdout = tee.tee(sys.stdout, open("eg2.out","w"))
    x, y = make_data()
    prior = make_prior(x)
    fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=fcn)
    print(fit.format())

    if MAKE_PLOT:
        import matplotlib.pyplot as plt
        plt.errorbar(
            x=gv.mean(x), xerr=gv.sdev(x), y=gv.mean(y), yerr=gv.sdev(y),
            fmt='ob'
            )
        # plot fit line
        x = np.linspace(0.99 * min(x).mean, 1.01 * max(x).mean, 100)
        p = dict(b=fit.pmean['b'], x=x)
        y = fcn(p)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x,y, ':r')
        plt.savefig('eg2.png', bbox_inches='tight')
        plt.show()

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
    return b0 / ((1 + gv.exp(b1 - b2 * x)) ** (1. / b3))


if __name__ == '__main__':
    main()
