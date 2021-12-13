#!/usr/bin/env python
# encoding: utf-8
"""
simple.py

"""
# Copyright (c) 2012 G. Peter Lepage. 
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
import sys

y = {                      # data for the dependent variable
   "data1" : gv.gvar([1.376,2.010],[[ 0.0047,0.01],[ 0.01,0.056]]),
   "data2" : gv.gvar([1.329,1.582],[[ 0.0047,0.0067],[0.0067,0.0136]]),
   "b/a"   : gv.gvar(2.0,0.5)
}
x = {                      # independent variable
   "data1" : np.array([0.1,1.0]),
   "data2" : np.array([0.1,0.5])
}
                           # a priori values for fit parameters
prior = dict(a=gv.gvar(0.5,0.5),b=gv.gvar(0.5,0.5))

# print(y["data1"][0].mean,"+-",y["data1"][0].sdev)
# print(gv.evalcov(y["data1"]))
def fcn(x,p):              # fit function of x and parameters p
   ans = {}
   for k in ["data1","data2"]:
      ans[k] = gv.exp(p['a'] + x[k]*p['b'])
   ans['b/a'] = p['b']/p['a']
   return ans

# do the fit
fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=fcn)
sys.stdout = open("eg0.out","w")
print(fit.format(maxline=True))     # print standard summary of fit

p = fit.p                  # best-fit values for parameters
outputs = dict(a=p['a'],b=p['b'])
outputs['b/a'] = p['b']/p['a']
inputs = dict(y=y,prior=prior)
print(gv.fmt_values(outputs))             # tabulate outputs
print(gv.fmt_errorbudget(outputs,inputs)) # print error budget for outputs

# save best-fit values in file "outputfile.p" for later use
import pickle
pickle.dump(fit.p,open("outputfile.p","wb"))
