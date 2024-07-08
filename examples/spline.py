"""
spline.py --- fitting a spline to data in file spline.p (or json)
"""

# Created by G. Peter Lepage (Cornell University) on 2014-04-28.
# Copyright (c) 2020-24 G. Peter Lepage.
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
import numpy as np
import sys 

if sys.argv[1:]:
    SHOW_PLOTS = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_PLOTS = True

def main():
    # read and fit data
    param, data = collect_data('spline.p')  # works with spline.json as well
    F, prior = make_fcn_prior(param)
    fit = lsqfit.nonlinear_fit(data=data, prior=prior, fcn=F)
    print(fit)

    # create f(m)
    f = gv.cspline.CSpline(fit.p['mknot'], fit.p['fknot'])

    # create error budget
    outputs = {'f(1)':f(1), 'f(5)':f(5), 'f(9)':f(9)}
    inputs = {'data':data}
    inputs.update(prior)
    print(gv.fmt_values(outputs))
    print(gv.fmt_errorbudget(outputs=outputs, inputs=inputs))

    if SHOW_PLOTS:
        make_plot(param, data, fit)

def make_fcn_prior(param):
    def F(p):
        f = gv.cspline.CSpline(p['mknot'], p['fknot'])
        ans = {}
        for s in param:
            ainv, am = param[s]
            m  = am * ainv
            ans[s] = f(m)
            for i,ci in enumerate(p['c']):
                ans[s] += ci * am ** (2 + 2 * i)
        return ans 
    prior = gv.gvar(dict(
        mknot=['1.00(1)', '1.5(5)', '3(1)', '9.00(1)'],
        fknot=['0(1)', '1(1)', '1(1)', '1(1)'],
        # mknot=['1.00(1)', '1.5(5)', '3(1)', '6(2)', '9.00(1)'],
        # fknot=['0(1)', '1(1)', '1(1)', '1(1)', '1(1)'],
        c=['0(1)'] * 5,
        ))
    return F, prior

def collect_data(datafile):
    param = dict(
        A=(10., np.array([0.1, 0.3, 0.5, 0.7, 0.9])),
        B=(5., np.array([0.3, 0.5, 0.7, 0.9])),
        C=(2.5, np.array([0.5, 0.7, 0.9])),
        )
    data = gv.load(datafile)
    return param,data

def make_plot(param, data, fit):
    import matplotlib.pyplot as plt 
    plt.cla()
    f = gv.cspline.CSpline(
        fit.p['mknot'], fit.p['fknot'], 
        )
    coliter = iter(['r', 'b', 'g'])
    m = np.arange(1, 9, 0.1)
    fm = f(m)
    fmavg = gv.mean(fm)
    fmplus = fmavg + gv.sdev(fm)
    fmminus = fmavg - gv.sdev(fm)    
    plt.fill_between(m, fmplus, fmminus, color='k', alpha=0.20) 
    plt.plot(m, fmavg, 'k:')
    # true function
    fm = 1. - .3 / m - .3 / m**2
    plt.plot(m, fm, 'k--')
    for s in data:
        plt.plot()
        ainv, am = param[s]
        ms = ainv * am 
        d = gv.mean(data[s])
        derr = gv.sdev(data[s])
        col = next(coliter)
        plt.errorbar(x=ms, y=d, yerr=derr, fmt=col + 'o')
        plt.text(ms[-1] - 0.6, d[-1], s, color=col, fontsize='x-large')
        fs = gv.mean(fm)
        ams = m / ainv
        idx = ams < am[-1]
        ams = ams[idx]
        fs = gv.mean(fm[idx])
        for i, ci in enumerate(fit.p['c']):
            fs += ci.mean * ams ** (2 * (i + 1))
        plt.plot(m[idx], fs, col + ':')
    plt.xlabel('m')
    plt.ylabel('f')
    plt.text(8, 0.65, 'f(m)', fontsize='x-large')
    # plt.savefig('spline.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()