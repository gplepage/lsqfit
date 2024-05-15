# Copyright (c) 2017-24 G. Peter Lepage.
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

# import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
import lsqfit
import sys 

try:
    import vegas
except:
    # fake the run so that `make run` still works
    outfile = open('bayes.out', 'r').read()
    print(outfile[:-1])
    exit(0)

if sys.argv[1:]:
    SHOW_PLOT = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_PLOT = True

if SHOW_PLOT:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        SHOW_PLOT = False

gv.ranseed(1)

def main():
    # least squares fit
    x, y = make_data()
    prior = make_prior()
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(fit)

    # check whether Gaussian using Bayesian integrator
    vfit = lsqfit.vegas_fit(fit=fit, neval=1_000)
    print(vfit)
    r = vfit.stats(histograms=True)
    for i in range(len(r)):
        print(f'\n---- p[{i}]')
        print(r.stats[i])
        if SHOW_PLOT:
            plt.subplot(2, 2, i + 1)
            r.stats[i].plot_histogram(plot=plt)
            plt.xlabel(f'p[{i}]')
    if SHOW_PLOT:
        plt.show()
    return 

def make_data():
    x = np.array([
        4.    ,  2.    ,  1.    ,  0.5   ,  0.25  ,  0.167 ,  0.125 ,
        0.1   ,  0.0833,  0.0714,  0.0625
        ])
    y = gv.gvar([
        '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
        '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
        '0.044(16)'
        ])
    return x, y

def make_prior():
    p = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
    p[1] = 20 * p[0] + gv.gvar('0.0(1)')     # p[1] correlated with p[0]
    return p

# @vegas.rbatchintegrand
def fcn(x, p):
    if len(p.shape) == 2:
        # add batch index
        x = x[:, None]
    return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

if __name__ == '__main__':
    gv.ranseed(1234)
    main()
