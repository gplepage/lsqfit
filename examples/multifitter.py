from __future__ import print_function
#!/usr/bin/env python
# encoding: utf-8
"""
Code for testing MultiFitter
"""
# Copyright (c) 2018 G. Peter Lepage.
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

import collections
import numpy as np
import lsqfit
import gvar as gv
import sys

gv.ranseed(12)   # remove randomness

if sys.argv[1:]:
    SHOW_GRID = eval(sys.argv[1])   # display picture of grid ?
else:
    SHOW_GRID = True

if SHOW_GRID:
    try:
        import matplotlib
    except ImportError:
        SHOW_GRID = False

def main():
    # initial data
    data = gv.gvar(dict(
        d1=['1.154(10)', '2.107(16)', '3.042(22)', '3.978(29)'],
        d2=['0.692(10)', '1.196(16)', '1.657(22)', '2.189(29)'],
        d3=['0.107(10)', '0.030(16)', '-0.027(22)', '-0.149(29)'],
        d4=['0.002(10)', '-0.197(16)', '-0.382(22)', '-0.627(29)'],
        ))
    models = [
       Linear('d1', x=[1,2,3,4], intercept='a', slope='s1', ncg=2),
       Linear('d2', x=[1,2,3,4], intercept='a', slope='s2'),
       Linear('d3', x=[1,2,3,4], intercept='a', slope='s3'),
       Linear('d4', x=[1,2,3,4], intercept='a', slope='s4'),
       ]
    # N.B., use log-normal prior for s1
    prior = gv.gvar(collections.OrderedDict([
        ('a','0(1)'), ('log(s1)','0(1)'), ('s2','0(1)'),
        ('s3','0(1)'), ('s4','0(1)'),
        ]))

    # reconfigure models and make fitter
    models = [[tuple(models[0:2]), models[2]], dict(mopt=True), models[3]]
    fitter = lsqfit.MultiFitter(models=models)
    # simultaneous fit
    print(30 * '-', 'lsqfit')
    fit = fitter.lsqfit(data=data, prior=prior, svdcut=1e-10)
    print(fit.formatall())
    if SHOW_GRID:
        fit.show_plots(view='diff')

    # chained fit
    print('\n' + 30 * '-' + ' chained_lsqfit')
    fit = fitter.chained_lsqfit(data=data, prior=prior)
    print(fit.format())
    print(fit.formatall())
    # if SHOW_GRID:
    #     fit.show_plots()

    # bootstrap last fit
    for bfit in fit.bootstrapped_fit_iter(n=2):
        print(20 * '*', 'bootstrap')
        print(gv.fmt(bfit.p, ndecimal=3))

class Linear(lsqfit.MultiFitterModel):
    def __init__(self, datatag, x, intercept, slope, ncg=1):
        super(Linear, self).__init__(datatag)
        # the independent variable
        self.x = np.array(x)
        # keys used to find the intercept and slope in a parameter dictionary
        self.intercept = intercept
        self.slope = slope
        self.ncg = ncg

    def fitfcn(self, p):
        try:
            return p[self.intercept] + p[self.slope] * self.x
        except KeyError:
            # slope parameter marginalized
            return len(self.x) * [p[self.intercept]]

    def buildprior(self, prior, mopt=None, extend=False):
        " Extract the model's parameters from prior. "
        newprior = {}
        # allow for log-normal, etc priors
        intercept, slope = gv.get_dictkeys(
            prior, [self.intercept, self.slope]
            )
        newprior[intercept] = prior[intercept]
        if mopt is None:
            # slope parameter marginalized if mopt is not None
            newprior[slope] = prior[slope]
        return newprior

    def builddata(self, data):
        " Extract the model's fit data from data. "
        return data[self.datatag]


if __name__ == '__main__':
    main()