#!/usr/bin/env python
# encoding: utf-8
"""
test-lsqfit.py

"""
# Copyright (c) 2012-2018 G. Peter Lepage.
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

import os
import sys
import pickle
import unittest
import numpy as np
import gvar as gv
from lsqfit import *
import lsqfit

## test_lsqfit should work for any of the fitters. Use DEFAULTS to choose:
#
# lsqfit.nonlinear_fit.DEFAULTS.update(dict(
#   fitter=['gsl_multifit', 'gsl_v1_multifit', 'scipy_least_squares'][1],
#   ))

FAST = False         # skips embayes and bootstrap tests

PRINT_FIT = False

mean = gv.mean
sdev = gv.sdev

class ArrayTests(object):
    def __init__(self):
        pass

    def assert_gvclose(self,x,y,rtol=1e-5,atol=1e-5,prt=False):
        """ asserts that the means and sdevs of all x and y are close """
        if hasattr(x,'keys') and hasattr(y,'keys'):
            if sorted(x.keys())==sorted(y.keys()):
                for k in x:
                    self.assert_gvclose(x[k],y[k],rtol=rtol,atol=atol)
                return
            else:
                raise ValueError("x and y have mismatched keys")
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.asarray(x).flat
        y = np.asarray(y).flat
        if prt:
            print(np.array(x))
            print(np.array(y))
        for xi,yi in zip(x,y):
            self.assertGreater(atol+rtol*abs(yi.mean),abs(xi.mean-yi.mean))
            self.assertGreater(10*(atol+rtol*abs(yi.sdev)),abs(xi.sdev-yi.sdev))

    def assert_arraysclose(self,x,y,rtol=1e-5,prt=False):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        max_val = max(np.abs(list(x)+list(y)))
        max_rdiff = max(np.abs(x-y))/max_val
        if prt:
            print(x)
            print(y)
            print(max_val,max_rdiff,rtol)
        self.assertAlmostEqual(max_rdiff,0.0,delta=rtol)

    def assert_arraysequal(self,x,y):
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = [float(xi) for xi in np.array(x).flatten()]
        y = [float(yi) for yi in np.array(y).flatten()]
        self.assertSequenceEqual(x,y)
        return 1


bool_iter = ([True,False][i%2] for i in range(1000))
def print_fit(fit,vd):
    """ print out fit """
    output = '\n'
    if fit.prior is None:
        cd = {'data':fit.y}
    else:
        cd = {'data':fit.y,'priors':fit.prior}
    output += fit.fmt_values(vd) + '\n'
    output += fit.fmt_errorbudget(vd,cd,percent=next(bool_iter))
    output += '\n'+fit.format(nline=1000)+'\n'
    if PRINT_FIT:
        print(output)
    return output

class test_lsqfit(unittest.TestCase,ArrayTests):
    def setUp(self):
        """ setup """
        global gvar
        gv.gvar = gv.gvar_factory()
        gvar = gv.gvar
        # gv.ranseed((1969,1974))   # don't use; want different rans each time
        self.label = None
        try:
            os.unlink("test-lsqfit.p")
        except OSError:
            pass

    def tearDown(self):
        global gvar
        gvar = None
        try:
            os.unlink("test-lsqfit.p")
        except OSError:
            pass
        # if self.label is not None:
        #     print self.label

    def t_basicfit(self, yfac, pfac, p0file):
        """ checks means, sdevs, fit.cov, etc in extreme cases """
        ycov = np.array([[2., .25], [.25, 4.]]) * yfac
        y = gv.gvar([1., 4.], ycov)
        pr = gv.BufferDict()
        pcov = np.array([[2., .5], [.5, 1.]]) * pfac
        pr['p'] = gv.gvar([4., 16.], pcov)
        def fcn(x,p):
            return dict(y=p['p'] ** 2)
        y = dict(y=y)
        fit = nonlinear_fit(data=(None,y), prior=pr, fcn=fcn, p0=p0file, debug=True)
        with open(p0file, 'rb') as f:
            p0 = pickle.load(f)
            self.assert_arraysclose(p0['p'], fit.pmean['p'])
        print_fit(fit,dict(y=wavg(fit.p['p']**2)))
        self.assertEqual(fit.dof, 2)
        self.assertAlmostEqual(fit.Q, 1.0)
        self.assertAlmostEqual(fit.chi2, 0.0)
        cd = {'data':fit.y, 'priors':fit.prior, 'p':[fit.prior['p']]}
        err = {}
        err['data'] = fit.p['p'][1].partialsdev(fit.y)
        err['priors'] = fit.p['p'][1].partialsdev(fit.prior)
        err['p'] = fit.p['p'][1].partialsdev(fit.prior['p'])
        if yfac>100*pfac:
            self.assert_gvclose(fit.p,pr)
            self.assert_arraysclose(fit.pmean['p'], gv.mean(pr)['p'])
            self.assert_arraysclose(fit.psdev['p'], gv.sdev(pr)['p'])
            self.assert_arraysclose(pcov,fit.cov)
            self.assert_arraysclose(pcov,gv.evalcov(fit.p['p']))
            self.assertNotAlmostEqual(fit.p['p'][1].sdev,err['data'],
                                    delta=5e-3*err['priors'])
            self.assertAlmostEqual(fit.p['p'][1].sdev,err['priors'],
                                    delta=5e-3*err['priors'])
            self.assertAlmostEqual(fit.p['p'][1].sdev,err['p'],
                                    delta=5e-3*err['p'])
        elif pfac>100*yfac:
            self.assert_gvclose(fit.p['p']**2,y['y'])
            self.assert_arraysclose(fit.pmean['p']**2,[x.mean for x in y['y']])
            self.assert_arraysclose([x.sdev for x in fit.p['p']**2],
                                [x.sdev for x in y['y']])
            self.assert_arraysclose(ycov,gv.evalcov(fit.p['p']**2))
            self.assertAlmostEqual(fit.p['p'][1].sdev,err['data'],
                                    delta=5e-3*err['data'])
            self.assertNotAlmostEqual(fit.p['p'][1].sdev,err['p'],
                            delta=5e-3*err['p'])
            self.assertNotAlmostEqual(fit.p['p'][1].sdev,err['priors'],
                            delta=5e-3*err['priors'])
        else:
            self.assertTrue(False)

    def test_basicfit(self):
        p0file = "test-lsqfit.p"
        for yf,pf in [(1e22,1),(1,1e22)]:
            self.label = ("nonlinear_fit prior-dominated"
                if yf>1 else "nonlinear_fit data-dominated")
            self.t_basicfit(yf, pf, p0file)

    def test_debug(self):
        " debug==True "
        with self.assertRaises(RuntimeError):
            y = gv.gvar(1, 1)
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return p['p'] * gv.gvar(1, 0)
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)
        with self.assertRaises(RuntimeError):
            y = [gv.gvar(1, 1)]
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return [p['p'], p['p']]
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)
        with self.assertRaises(RuntimeError):
            y = dict(a=gv.gvar(1, 1))
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return dict(a=[p['p'], p['p']])
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)
        with self.assertRaises(RuntimeError):
            y = dict(a=[gv.gvar(1, 1)])
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return dict(a=[p['p'], p['p']])
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)
        with self.assertRaises(RuntimeError):
            y = dict(a=gv.gvar(1, 1))
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return dict(a=p['p'], b=p['p'])
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)
        with self.assertRaises(RuntimeError):
            y = dict(a=gv.gvar(1, 1), b=gv.gvar(1,1))
            prior = dict(p=gv.gvar(0, 1))
            def fcn(p):
                return dict(a=p['p'])
            fit = nonlinear_fit(data=y, prior=prior, fcn=fcn, debug=True)

    def test_format(self):
        """ fit.format """
        tol = (1e-15, 1e-15, 1e-15)
        # case 1 - y and prior dictionaries
        y = gv.BufferDict([('a',gv.gvar(1.5,1.0)), ('b',gv.gvar(0.8,0.5))])
        prior = gv.BufferDict(p=gv.gvar(0,2))
        def f(p):
            return dict(a=p['p'], b=p['p'])

        fit = nonlinear_fit(
            data=y, prior=prior, fcn=f, svdcut=1e-15, tol=tol,
            )
        out = [
            'Least Square Fit:',
            '  chi2/dof [dof] = 0.3 [2]    Q = 0.74    logGBF = -2.9682',
            '',
            'Parameters:',
            '              p    0.90 (44)     [  0.0 (2.0) ]  ',
            '',
            'Fit:',
            '      key       y[key]    f(p)[key]',
            '-----------------------------------',
            '        a    1.5 (1.0)    0.90 (44)  ',
            '        b    0.80 (50)    0.90 (44)  ',
            '',
            'Settings:',
            '  svdcut/n = 1e-15/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 2*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(True).split('\n')[:len(out)])
        self.assertEqual(out, fit.format(True, pstyle='v').split('\n')[:len(out)])
        out = [
            'Least Square Fit:',
            '  chi2/dof [dof] = 0.3 [2]    Q = 0.74    logGBF = -2.9682',
            '',
            'Parameters:',
            '              p   0.895238 +- 0.436436          [     0 +- 2 ]  ',
            '',
            'Settings:',
            '  svdcut/n = 1e-15/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 2*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(pstyle="vv").split('\n')[:len(out)])
        prior['dummy'] = gv.gvar(10,1)
        fit = nonlinear_fit(
            data=y, prior=prior, fcn=f, svdcut=1e-15, tol=tol,
            )
        out = [
            'Least Square Fit:',
            '  chi2/dof [dof] = 0.3 [2]    Q = 0.74    logGBF = -2.9682',
            '',
            'Parameters:',
            '              p    0.90 (44)     [  0.0 (2.0) ]  ',
            '',
            'Settings:',
            '  svdcut/n = 1e-15/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 2*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(pstyle='m').split('\n')[:len(out)])
        self.assert_gvclose(fit.p['p'], wavg([y['a'],y['b'],prior['p']]))

        # case 2 - x and y; no prior
        x = np.array([1.,2.])
        y = np.array([gv.gvar(1.3,0.3), gv.gvar(1.9,0.5)])
        prior = dict(p=gv.gvar(0,2))
        p0 = gv.mean(prior)
        def f(x,p):
            return p['p']*x

        fit = nonlinear_fit(
            p0=p0, data=(x,y), fcn=f, svdcut=None, tol=tol,
            )
        out = [
            'Least Square Fit (no prior):',
            '  chi2/dof [dof] = 0.8 [1]    Q = 0.37    ',
            '',
            'Parameters:',
            '              p    1.09 (19)     [   0 +- inf ]  ',
            '',
            'Fit:',
            '     x[k]         y[k]    f(x[k],p)',
            '-----------------------------------',
            '        1    1.30 (30)    1.09 (19)  ',
            '        2    1.90 (50)    2.19 (38)  ',
            '',
            'Settings:',
            '  svdcut/n = None/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 2*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(100).split('\n')[:len(out)])

        # case three vectors
        y = gv.gvar([['1.0(1)', '2.0(1)'], ['0.9(2)', '2.2(1)']])
        prior = gv.gvar(dict(a=['1.0(5)', '1.0(5)']))
        def fcn(p):
            return [p['a'], p['a']]
        fit = nonlinear_fit(
            prior=prior, data=y, fcn=fcn, debug=True, tol=tol,
            )
        out = [
            'Least Square Fit:',
            '  chi2/dof [dof] = 1.7 [4]    Q = 0.14    logGBF = -2.3346',
            '',
            'Parameters:',
            '            a 0   0.981 (88)     [  1.00 (50) ]  ',
            '              1   2.078 (70)     [  1.00 (50) ]  **',
            '',
            'Fit:',
            '      key       y[key]     f(p)[key]',
            '------------------------------------',
            '      0,0    1.00 (10)    0.981 (88)  ',
            '      0,1    2.00 (10)    2.078 (70)  ',
            '      1,0    0.90 (20)    0.981 (88)  ',
            '      1,1    2.20 (10)    2.078 (70)  *',
            '',
            'Settings:',
            '  svdcut/n = 1e-15/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 2*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(True).split('\n')[:len(out)])

        # log-normal
        prior = {'log(a)':gv.log(gv.gvar(['1.0(5)', '1.0(5)']))}
        fit = nonlinear_fit(
            prior=prior, data=y, fcn=fcn, debug=True, tol=tol,
            )
        # print(fit.format(extend=True))
        out = [
            'Least Square Fit:',
            '  chi2/dof [dof] = 1.1 [4]    Q = 0.36    logGBF = -1.77',
            '',
            'Parameters:',
            '       log(a) 0   -0.020 (90)     [  0.00 (50) ]  ',
            '              1    0.739 (34)     [  0.00 (50) ]  *',
            '------------------------------------------------',
            '            a 0    0.981 (88)     [  1.00 (50) ]  ',
            '              1    2.093 (71)     [  1.00 (50) ]  **',
            '',
            'Fit:',
            '      key       y[key]     f(p)[key]',
            '------------------------------------',
            '      0,0    1.00 (10)    0.981 (88)  ',
            '      0,1    2.00 (10)    2.093 (71)  ',
            '      1,0    0.90 (20)    0.981 (88)  ',
            '      1,1    2.20 (10)    2.093 (71)  *',
            '',
            'Settings:',
            '  svdcut/n = 1e-15/0    tol = (1e-15,1e-15,1e-15)    (itns/time = 5*/0.0)',
            ][:-1]
        self.assertEqual(out, fit.format(True).split('\n')[:len(out)])

    def test_unusual_cases(self):
        """ unusual cases """
        # case 1 - y and prior are scalars
        y = gv.gvar(1.5,.1)
        prior = gv.gvar(2.,.5)
        def f(p): return p

        fit = nonlinear_fit(data=y,prior=prior,fcn=f)
        self.assertEqual(str(fit.p), str(wavg([y, prior])))

        # case 2 - no x, y is 2 element array
        y = gv.gvar([(1.5,.1),(1.7,.2)])
        prior = gv.gvar(2.,.5)
        def f(p): return [p, p]

        fit = nonlinear_fit(data=y,prior=prior,fcn=f, tol=1e-8)
        self.assertEqual(str(fit.p), str(wavg(y.tolist()+[prior])))

    def test_wavg1(self):
        """ fit vs wavg uncorrelated """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)

        c = gvar(4.,0.125)
        for ny,yf,ryf in [(1,1.,1.),(5,1.,1.),(5,2.,1.),(5,1.,2.)]:
            y = np.array([gvar(c(),yf*c.sdev) for i in range(ny)])
            rooty = np.array([gvar(c()**0.5,ryf*(c**0.5).sdev)
                                for i in range(ny)])
            pr = dict(rooty=rooty)  # prior
            def fcn(x,p):
                return p['rooty']**2

            fit = nonlinear_fit(data=(None,y),prior=pr,fcn=fcn,
                            tol=1e-10,debug=True)
            print_fit(fit,dict(y=wavg(fit.p['rooty']**2)))
            output = avg(fit.p['rooty']**2)
            # self.assertEqual(wavg.dof,ny-1)
            input = wavg([avg(y),avg(rooty**2)])
            self.assertEqual(input.dof,1)
            # print("*** wavg1",input,output)
            self.assert_gvclose(input,output,rtol=1e-2)
            if ny>1:  # cov diag
                self.assert_arraysequal(fit.cov,np.diag(np.diag(fit.cov)))

    def test_wavg2(self):
        """ fit vs wavg correlated """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)

        c = gvar(4.,0.125)
        for ny,yf,ryf in [(1,1.,1.),(5,1.,1.),(5,1.,1.),(5,2.,1.),(5,1.,2.)]:
            yo = np.array([gvar(c(),yf*c.sdev) for i in range(ny+1)])
            rootyo = np.array([gvar(c()**0.5,ryf*(c**0.5).sdev)
                                for i in range(ny+1)])
            y = (yo[:-1]+yo[1:])/2.             # introduce correlations
            rooty = (rootyo[:-1]+rootyo[1:])/2
            pr = gv.BufferDict()
            pr['rooty'] = rooty
            def fcn(x,p):
                return p['rooty']**2

            fit = nonlinear_fit(
                data=(None,y), prior=pr, fcn=fcn, tol=1e-10, debug=True
                )
            print_fit(fit,dict(y=wavg(fit.p['rooty']**2)))
            # check mean and std devn
            output = avg(fit.p['rooty']**2)
            input = wavg([avg(yo),avg(rootyo**2)])
            # print("*** wavg2",input,output)
            self.assert_gvclose(input,output,rtol=2e-2)

            # check cov matrix
            invcovy = np.linalg.inv(gv.evalcov(y))
            invcovry2 = np.linalg.inv(gv.evalcov(rooty**2))
            inv_cov_input = invcovy+invcovry2   # add prior and data in quad.
            cov_output = gv.evalcov(fit.p['rooty']**2) # output cov matrix
            io_prod = np.dot(inv_cov_input,cov_output)  # should be unit matrix
            self.assert_arraysclose(io_prod,np.diag(ny*[1.]),rtol=5e-2)
            if ny>1:        # cov not diag
                self.assertTrue(not np.all(fit.cov==np.diag(np.diag(fit.cov))))


    def test_wavg_svd(self):
        """ wavg with svd cut """
        a,b,c = gvar(["1(1)","1(1)","1(1)"])
        var = wavg([(a+b)/2,(a+c)/2.,a],svdcut=1-1e-16).var
        self.assertAlmostEqual(var,0.4561552812808828)
        var = wavg([(a+b)/2.,(a+c)/2.,a],svdcut=1e-18).var
        self.assertAlmostEqual(var,1./3.)
        var = wavg([b,c,a]).var
        self.assertAlmostEqual(var,1./3.)

    def test_wavg_vec(self):
        """ wavg of arrays """
        ans = wavg([[gvar(2.1,1.),4+gvar(2.1,1.)],
                    [gvar(1.9,10.),4+gvar(1.9,10.)]])
        self.assert_arraysclose(mean(ans),[2.09802,6.09802],rtol=1e-4)
        self.assert_arraysclose(sdev(ans),[0.995037,0.995037],rtol=1e-4)

    def test_wavg_dict(self):
        """ wavg of dicts """
        ans = wavg([
            dict(c=2+gvar(1.9,10.)),
            dict(a=gvar(2.1,1.),  b=[gvar(2.1,1.),4+gvar(2.1,1.)], c=2+gvar(2.1,1.)),
            dict(a=gvar(1.9,10.), b=[gvar(1.9,10.),4+gvar(1.9,10.)]),
            ])
        self.assert_arraysclose(mean(ans['b']),[2.09802,6.09802],rtol=1e-4)
        self.assert_arraysclose(sdev(ans['b']),[0.995037,0.995037],rtol=1e-4)
        self.assertAlmostEqual(ans['a'].mean, 2.09802, places=4)
        self.assertAlmostEqual(ans['a'].sdev, 0.995037, places=4)
        self.assertAlmostEqual(ans['c'].mean, 4.09802, places=4)
        self.assertAlmostEqual(ans['c'].sdev, 0.995037, places=4)

    def test_wavg_extend(self):
        g1 = gv.gvar({'log(a)':'1(1)', 'b':'2(1)'})
        g2 = gv.gvar({'log(a)':'2.00000(1)', 'b':'1(1)', 'c':'12(1)'})
        avg = wavg([g1, g2])
        self.assertAlmostEqual(np.exp(avg['log(a)'].mean), avg['a'].mean)
        self.assertTrue(gv.equivalent(avg['c'], g2['c']))
        self.assertAlmostEqual(avg['a'].mean, g2['a'].mean)

    def test_wavg_edge_cases(self):
        " wavg for edge cases "
        x = wavg(gv.gvar(['0(1)', '1(1)']))
        self.assertAlmostEqual(x.mean, 0.5)
        self.assertAlmostEqual(x.sdev, 0.5 ** 0.5)
        x = wavg([gv.gvar(dict(a='0(1)')), gv.gvar(dict(a='1(1)'))])
        self.assertAlmostEqual(x['a'].mean, 0.5)
        self.assertAlmostEqual(x['a'].sdev, 0.5 ** 0.5)
        x = wavg(gv.gvar(['0(1)']))
        self.assertAlmostEqual(x.mean, 0.0)
        self.assertAlmostEqual(x.sdev, 1.0)
        self.assertEqual(x.fit, None)
        x = wavg([gv.gvar(dict(a='0(1)'))])
        self.assertAlmostEqual(x['a'].mean, 0.0)
        self.assertAlmostEqual(x['a'].sdev, 1.0)
        self.assertEqual(x.fit, None)
        x = wavg([])
        self.assertEqual(x, None)

    def test_noprior(self):
        """ fit without prior """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)

        y = gvar(4.,0.25)
        ny = 8
        y = np.array([gvar(y(),y.sdev) for i in range(ny+1)])
        ynocorr = y[:-1]
        y = (y[:-1]+y[1:])/2.
        ycov = gv.evalcov(y)
        ymean = np.array([x.mean for x in y])
        ydict = dict(y=y)
        def arrayfcn(x,p):
            return p**2 # np.array([p[k]**2 for k in p])

        def dictfcn(x,p):
            return dict(y=p**2) # np.array([p[k]**2 for k in p]))

        p0 = None
        for i,data in enumerate([(None,y),(None,ydict),
                            (None,dict(y=ynocorr)),(None,ymean,ycov)]):
            p0 = np.ones(ny,float)*0.1
            if isinstance(data[1],dict):
                fcn = dictfcn
                datay = data[1]['y']
            else:
                fcn = arrayfcn
                datay = data[1]
            if len(data)<3:
                dataycov = gv.evalcov(datay)
            else:
                datay = gv.gvar(datay,data[2])
                dataycov = data[2]
            fit = nonlinear_fit(data=data,p0=p0,fcn=fcn,debug=True,tol=1e-14)
            print_fit(fit,dict(y=avg(fit.p**2)))
            self.assertIsNone(fit.logGBF)
            self.assertEqual(fit.dof,0.0)
            self.assertAlmostEqual(fit.chi2,0.0,places=4)
            self.assert_arraysclose(gv.evalcov(fit.p**2),
                                        dataycov,rtol=1e-4)
            self.assert_gvclose(fit.p**2,datay,1e-4)

    @unittest.skipIf(FAST,"for speed")
    def test_bootstrap(self):
        """ bootstrap_iter """
        # data and priors
        def bin(y): # correlates different y's
            return (y[:-1]+y[1:])/2.

        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)

        yc = gvar(4., 0.25)
        p2c = gvar(4., 0.25)
        ny = 3
        y = np.array([gvar(yc(), yc.sdev) for i in range(ny)])
        p = np.array([gvar(p2c(), p2c.sdev)**0.5 for i in range(ny)])
        eps = gvar(1., 1e-4)

        cases = [(y[:-1], p[:-1], False),
                (bin(y), bin(p), False),
                (bin(y), bin(p), True),
                (bin(y), p[:-1], False),
                (y[:-1], bin(p), False),
                (y[:-1]*eps, p[:-1]*eps**0.5, False),
                (y[:-1], None, False),
                (y[:-1], None, True),
                (bin(y), None, False)]
        for y, p, use_dlist in cases:
            # fit then bootstrap
            prior = None if p is None else np.array(p)
            p0 = gv.mean(y)**0.5 if p is None else None
            data = None, y
            def fcn(x, p):
                return p**2 # np.array([p[k]**2 for k in p])

            fit = nonlinear_fit(data=data,fcn=fcn,prior=prior,p0=p0,debug=True)
            # print(fit.format(nline=100))
            def evfcn(p):
                return {0:np.average(p**2)} # [p[k]**2 for k in p])

            bs_ans = gv.dataset.Dataset()
            nbs = 1000 / 10
            if use_dlist:
                fit_iter = fit.bootstrapped_fit_iter(
                    datalist=((None,yb) for yb in gv.bootstrap_iter(y,nbs))
                    )
            else:
                fit_iter = fit.bootstrapped_fit_iter(n=nbs)
            for bfit in fit_iter:
                if bfit.error is None:
                    bs_ans.append(evfcn(bfit.pmean))
            bs_ans = gv.dataset.avg_data(bs_ans,median=True,spread=True)[0]
            target_ans = wavg([avg(y),avg(p**2)]) if p is not None else avg(y)
            fit_ans = avg(fit.p**2)
            rtol = 10.*fit_ans.sdev/nbs**0.5  # 10 sigma
            # print(bs_ans,fit_ans,target_ans,rtol)
            self.assert_gvclose(target_ans, fit_ans, rtol=rtol)
            self.assert_gvclose(bs_ans, fit_ans, rtol=rtol)


    def test_svd(self):
        """ svd cuts """
        # data and priors
        fac = 100.
        rtol = 1/fac
        sig1 = 1./fac
        sig2 = 1e-2/fac
        y0 = gvar(1., sig1) * np.array([1,1]) + gvar(0.1, sig2) * np.array([1,-1])
        y = y0 + next(gv.raniter(y0)) - gv.mean(y0)
        p02 = gvar(1., sig1) * np.array([1,1]) + gvar(0.1, sig2) * np.array([1,-1])
        p = (p02 + next(gv.raniter(p02)) - gv.mean(p02))**0.5
        eps = gvar(1., 1.e-8)
        reps = eps**0.5

        cases = [
            (y, p, 1e-20, False),
            (y, p, 1e-2, False),
            (y*eps, p*reps, 1e-20, True),
            (y*eps,p*reps, 1e-2, True),
            ((gv.mean(y), gv.evalcov(y)), p, 1e-20, False),
            ((gv.mean(y), gv.evalcov(y)), p, 1e-2, False)
            ]
        for y, p, svdcut, correlated in cases:
            prior = np.array(p)
            if not isinstance(y,tuple):
                data = None,y
            else:
                data = (None,)+y
            def fcn(x,p):
                return p**2 # np.array([p[k]**2 for k in p])

            fit = nonlinear_fit(data=data,fcn=fcn,prior=prior,
                                svdcut=svdcut,debug=True)
            # print(fit.format(nline=100))
            y = fit.y.flatten()
            pr = fit.prior.flatten()
            p = fit.p.flatten()
            ans_y = [(y[0]+y[1])/2, (y[0]-y[1])/2]
            ans_pr = [(pr[0]**2 + pr[1]**2) / 2,(pr[0]**2 - pr[1]**2) / 2]
            ans_p = [(p[0]**2+p[1]**2)/2, (p[0]**2-p[1]**2)/2]
            target_ans = wavg([ans_y, ans_pr])
            fit_ans = np.array(ans_p)
            self.assert_gvclose(target_ans, fit_ans, rtol=rtol)
            s2 = max(fit_ans[0].sdev*sig2/sig1,svdcut**0.5*fit_ans[0].sdev)
            self.assertAlmostEqual(fit_ans[1].sdev/s2, 1., places=2)
            if np.sum(fit.svdcorrection).sdev == 0:
                self.assertEqual(fit.svdn, 0)
            else:
                self.assertEqual(fit.svdn, 2)
            if not correlated:
                self.assertEqual(fit.nblocks[2], 2)
            else:
                self.assertEqual(fit.nblocks[4], 1)

        # negative cut
        x, dx = gvar(['1(1)', '0.01(1)'])
        data = np.array([(x+dx)/2, (x-dx)/20.])
        # g, wgts = svd([(x+dx)/2, (x-dx)/20.], svdcut=-0.2 ** 2, wgts=-1)
        prior = gv.gvar(['1(10)', '0.05(50)'])
        def f(p):
            return p
        fit = nonlinear_fit(data=data, prior=prior, fcn=f, svdcut=-0.2 ** 2)
        g = fit.p
        self.assertEqual((g[0]+g[1]*10).fmt(1), '1.0(1.0)')
        self.assertEqual(fit.dof, 1)
        self.assertEqual(fit.svdn, 1)



    def test_logGBF(self):
        " fit.logGBF "
        yg = gv.gvar(['2(1)', '4(6)', '-0.62(1)', '-100(10)'])
        y = gv.gvar(next(gv.raniter(yg)), gv.sdev(yg))
        prior = gv.gvar(gv.mean(yg), 0.5 * gv.sdev(yg))
        ymean = gv.mean(yg)
        yvar = gv.var(yg) + gv.var(prior)   # variance of posterior y distn
        def prob(y, ymean=ymean, yvar=yvar):
            " posterior probability distribution "
            return np.prod(
                np.exp(- (y - ymean) ** 2 / (2 * yvar)) /
                (2 * yvar * np.pi) ** 0.5
                )
        def chi2(y, ymean=ymean, yvar=yvar):
            " chi**2 from prob(y) "
            return np.sum(
                (y - ymean) ** 2 / yvar
                )
        def fcn(p):
            " fit function "
            return p
        fit = nonlinear_fit(data=y, fcn=fcn, prior=prior)
        self.assertAlmostEqual(fit.logGBF, np.log(prob(gv.mean(y))))
        self.assertAlmostEqual(fit.chi2, chi2(gv.mean(y)))

    @unittest.skipIf(FAST,"for speed")
    def test_empbayes(self):
        """ empbayes fit """
        y = gv.gvar([
            '0.5351(54)', '0.6762(67)', '0.9227(91)',
            '1.3803(131)', '4.0145(399)'
            ])
        x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])
        def f(x, p):                  # fit function
            n = np.arange(len(p))
            return np.sum(p[:, None] * x[None, :] ** n[:, None], axis=0)
        def fitargs(z):
            prior = gv.gvar(25 * ['0 +- %g' % np.exp(z[0])])
            return dict(data=(x, y), prior=prior, fcn=f)
        if PRINT_FIT:
            def analyzer(z,f,it):
                print("%3d  %.8f  ->  %.8f" % (it, np.exp(z[0]), f))

        else:
            analyzer = None
        z0 = np.log([0.7]) # [5.])
        fit,z = empbayes_fit(z0,fitargs,analyzer=analyzer,tol=1e-3)
        self.assertAlmostEqual(np.exp(z[0]), 0.6012, places=1)
        # know correct answer from the process for creating the data

        # variation: use a number
        def fitargs(z):
            prior = gv.gvar(25 * ['0 +- %g' % np.exp(z)])
            return dict(data=(x, y), prior=prior, fcn=f)
        z0 = np.log(0.7) # 5.)
        fit,z = empbayes_fit(z0,fitargs,analyzer=analyzer,tol=1e-3)
        self.assertAlmostEqual(np.exp(z), 0.6012, places=1)

        # variation: use a dictionary
        def fitargs(z):
            prior = gv.gvar(25 * ['0 +- %g' % np.exp(z['z'])])
            return dict(data=(x, y), prior=prior, fcn=f)
        z0 = dict(z=np.log(0.7))
        fit,z = empbayes_fit(z0,fitargs,analyzer=analyzer,tol=1e-3)
        # self.assertAlmostEqual(np.exp(z['z']), 0.6012, places=1)

    def test_unpack_data(self):
        """ lsqfit._unpack_data """
        def make_mat(wlist, n):
            ans = np.zeros((n,n), float)
            i, wgts = wlist[0]
            if len(i) > 0:
                ans[i, i] = np.array(wgts) ** 2
            for i, wgts in wlist[1:]:
                for w in wgts:
                    ans[i, i[:, None]] += np.outer(w, w)
            return ans
        def test_logdet(fdata, prior, y):
            if prior is not None:
                yp = np.concatenate((y.flat, prior.flat))
                self.assertAlmostEqual(
                    fdata.logdet,
                    np.log(np.linalg.det(gv.evalcov(yp))),
                    )
            else:
                self.assertAlmostEqual(
                    fdata.logdet,
                    np.log(np.linalg.det(gv.evalcov(y.flat))),
                    )

        # case 1 - (x,y) and prior
        yo = dict(y=[gv.gvar(1, 2), gv.gvar(10,4)])
        p = gv.gvar(2,4)
        po = lsqfit._unpack_gvars({'log(p)':gv.log(p)})
        xo = 20
        x, y, prior, fdata = lsqfit._unpack_data(
            data=(xo,yo), prior=po, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x, xo)
        self.assert_gvclose(y['y'], yo['y'])
        self.assert_gvclose(gv.exp(prior['log(p)']), p)
        self.assert_gvclose(prior['p'], p)

        self.assert_arraysequal(fdata.mean, [1., 10., gv.log(2.)])
        self.assert_arraysequal(fdata.inv_wgts[0][0], [0,1,2])
        self.assert_arraysequal(fdata.inv_wgts[0][1], [0.5, 0.25, 0.5])
        sumsvd = fdata.svdcorrection
        self.assertEqual([sumsvd.mean, sumsvd.sdev], [0, 0])
        test_logdet(fdata, prior, y)

        # case 2 - no x
        po = lsqfit._unpack_gvars(dict(p=gv.gvar(2, 4)))
        x, y, prior, fdata = lsqfit._unpack_data(
            data=yo, prior=po, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x,False)
        self.assert_gvclose(y['y'], yo['y'])
        self.assert_gvclose(prior['p'], po['p'])
        self.assert_arraysequal(fdata.mean, [1., 10., 2.])
        self.assert_arraysequal(fdata.inv_wgts[0][0], [0,1,2])
        self.assert_arraysequal(fdata.inv_wgts[0][1], [0.5, 0.25, 0.25])
        sumsvd = fdata.svdcorrection
        self.assertEqual([sumsvd.mean, sumsvd.sdev], [0, 0])
        test_logdet(fdata, prior, y)

        # case 3 - no prior, x
        x, y, prior, fdata = lsqfit._unpack_data(
            data=yo, prior=None, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x,False)
        self.assert_gvclose(y['y'], yo['y'])
        self.assertEqual(prior, None)
        self.assert_arraysequal(fdata.mean, [1., 10.])
        self.assert_arraysequal(fdata.inv_wgts[0][0], [0,1])
        self.assert_arraysequal(fdata.inv_wgts[0][1], [0.5, 0.25])
        sumsvd = fdata.svdcorrection
        self.assertEqual([sumsvd.mean, sumsvd.sdev], [0, 0])
        test_logdet(fdata, prior, y)


        # case 3 - prior and data correlated
        one = gv.gvar(1,1e-4)
        yo['y'][0] *= one
        po['p'] *= one
        x, y, prior, fdata = lsqfit._unpack_data(
            data=yo, prior=po, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x,False)
        self.assert_gvclose(y['y'],yo['y'])
        self.assert_gvclose(prior['p'],po['p'])
        self.assert_arraysequal(fdata.mean, [1,10,2])
        self.assert_arraysequal(fdata.inv_wgts[0][0], [1])
        self.assert_arraysequal(fdata.inv_wgts[1][0], [0, 2])
        fd_icov = make_mat(fdata.inv_wgts, 3)
        icov = np.linalg.inv(gv.evalcov(y['y'].tolist()+[po['p']]))
        self.assert_arraysclose(fd_icov,icov)
        icov = np.linalg.inv(gv.evalcov(np.concatenate((y.flat,prior.flat))))
        self.assert_arraysclose(fd_icov,icov)
        test_logdet(fdata, prior, y)

        # case 4 - vector p, no correlation
        yo = dict(y=[gv.gvar(1, 2),gv.gvar(10,4)])
        pvo = lsqfit._unpack_gvars([gv.gvar(1,2),gv.gvar(1,4)])
        x, y, prior, fdata = lsqfit._unpack_data(
            data=yo, prior=pvo, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x,False)
        self.assert_gvclose(y['y'],yo['y'])
        self.assert_gvclose(prior,pvo)
        self.assert_arraysequal(fdata.mean,[1, 10, 1, 1])
        self.assert_arraysequal(fdata.inv_wgts[0][0], [0, 1, 2, 3])
        self.assert_arraysequal(fdata.inv_wgts[0][1],[0.5, 0.25, 0.5, 0.25])
        test_logdet(fdata, prior, y)

        # case 5 - vector p, correlation
        yo = dict(y=[gv.gvar(1, 2),gv.gvar(10,4)])
        pvo = lsqfit._unpack_gvars([gv.gvar(1,2),gv.gvar(1,4)])
        yo['y'][0] *= one
        pvo[0] *= one
        x, y, prior, fdata = lsqfit._unpack_data(
            data=yo, prior=pvo, svdcut=None,
            uncorrelated_data=False,
            )
        self.assertEqual(x,False)
        self.assert_gvclose(y['y'],yo['y'])
        self.assert_gvclose(prior,pvo)
        self.assert_arraysequal(fdata.inv_wgts[0][0], [1, 3])
        self.assertEqual(numpy.ndim(fdata.inv_wgts[0][1]), 1)
        self.assert_arraysequal(fdata.inv_wgts[1][0], [0, 2])
        self.assertEqual(numpy.ndim(fdata.inv_wgts[1][1]), 2)
        fd_icov = make_mat(fdata.inv_wgts, 4)
        icov = np.linalg.inv(gv.evalcov(y['y'].tolist()+prior.tolist()))
        self.assert_arraysclose(fd_icov,icov)
        icov = np.linalg.inv(gv.evalcov(np.concatenate((y.flat,prior.flat))))
        self.assert_arraysclose(fd_icov,icov)

        test_logdet(fdata, prior, y)

        # case 6 - (x,y,ycov)
        yo = dict(y=[gv.gvar(1, 2),gv.gvar(10,4)])
        yo_mean = gv.mean(yo['y'])
        yo_cov = gv.evalcov(yo['y'])
        po = lsqfit._unpack_gvars(dict(p=gv.gvar(2, 4)))
        xo = 20
        svdcut = None
        x, y, prior, fdata = lsqfit._unpack_data(
            data=(xo,yo_mean,yo_cov), prior=po, svdcut=svdcut,
            uncorrelated_data=False,
            )
        self.assertEqual(x,xo)
        self.assert_gvclose(y,yo['y'])
        self.assert_gvclose(prior['p'],po['p'])
        self.assert_arraysequal(fdata.mean,[1,10,2])
        self.assertEqual(set(fdata.inv_wgts[0][0]), set([0, 1, 2]))
        self.assert_arraysequal(
            fdata.inv_wgts[0][1][fdata.inv_wgts[0][0]],
            [0.5, 0.25, 0.25]
            )
        test_logdet(fdata, prior, y)

        # case 8 - svd cuts
        a = gv.gvar(1,1)
        da = gv.gvar(0,0.01)
        yo = gv.gvar([(a+da),(a-da)])
        a = gv.gvar(1,1)
        da = gv.gvar(0,0.01)
        po = gv.gvar([(a+da),(a-da)])
        sc = 0.01
        da_svd = gv.gvar(0,sc**0.5)
        for svdcut in [0.0, sc]:
            x, y, prior, fdata = lsqfit._unpack_data(
                data=yo, prior=po, svdcut=svdcut,
                uncorrelated_data=False,
                )
            self.assertEqual(x,False)
            self.assert_arraysequal(gv.mean(y),gv.mean(yo))
            if svdcut == 0:
                self.assert_gvclose(y,yo)
                self.assert_gvclose((y[1]-y[0])/2,da)
            else:
                self.assert_gvclose((y[1]-y[0])/2,da_svd)
                with self.assertRaises(AssertionError):
                    self.assert_gvclose(y,yo)

            self.assert_arraysequal(gv.mean(prior),gv.mean(po))
            if svdcut == 0:
                self.assert_gvclose(prior,po)
                self.assert_gvclose((prior[1]-prior[0])/2,da)
            else:
                self.assert_gvclose((prior[1]-prior[0])/2,da_svd)
                with self.assertRaises(AssertionError):
                    self.assert_gvclose(prior,po)
            self.assert_arraysclose(
                make_mat(fdata.inv_wgts, 4),
                np.linalg.inv(gv.evalcov(list(y.flat) + list(prior.flat)))
                )
            test_logdet(fdata, prior, y)

        # others: svdcut= #
        # case ?? - wrong length data tuple
        with self.assertRaises(ValueError):
            x, y, prior, fdata = lsqfit._unpack_data(
                data=(xo,yo,yo,yo), prior=po, svdcut=svdcut,
                uncorrelated_data=False,
                )
        with self.assertRaises(ValueError):
            x, y, prior, fdata = lsqfit._unpack_data(
                data=(xo,), prior=po, svdcut=svdcut,
                uncorrelated_data=False,
                )

    def test_uncorrelated_data(self):
        " nonlinear_fit(udata=y ... ) "
        # arrays
        y0 = gv.gvar('1.01(1)')
        y = y0 * gv.gvar(4 * ['1.000(1)'])
        def f(p):
            return 4 * [p]

        # with prior
        fit1 = nonlinear_fit(udata=y, fcn=f, prior='1(1)')
        fit2 = nonlinear_fit( data=y, fcn=f, prior='1(1)')
        self.assertAlmostEqual(fit1.pmean, fit2.pmean, places=3)
        self.assertAlmostEqual(fit1.psdev * 2, fit2.psdev, places=3)

        # without prior
        fit1 = nonlinear_fit(udata=y, fcn=f, p0=1.)
        fit2 = nonlinear_fit( data=y, fcn=f, p0=1.)
        self.assertAlmostEqual(fit1.pmean, fit2.pmean, places=3)
        self.assertAlmostEqual(fit1.psdev * 2, fit2.psdev, places=3)

        # dictionaries
        y0 = gv.gvar('1.01(1)')
        y = dict(y=y0 * gv.gvar(4 * ['1.000(1)']))
        def f(p):
            return dict(y=4 * [p['p']])

        # with prior
        fit1 = nonlinear_fit(udata=y, fcn=f, prior=dict(p='1(1)'))
        fit2 = nonlinear_fit( data=y, fcn=f, prior=dict(p='1(1)'))
        self.assertAlmostEqual(fit1.pmean['p'], fit2.pmean['p'], places=3)
        self.assertAlmostEqual(fit1.psdev['p'] * 2, fit2.psdev['p'], places=3)

        # without prior
        fit1 = nonlinear_fit(udata=y, fcn=f, p0=dict(p='1'))
        fit2 = nonlinear_fit( data=y, fcn=f, p0=dict(p=1))
        self.assertAlmostEqual(fit1.pmean['p'], fit2.pmean['p'], places=3)
        self.assertAlmostEqual(fit1.psdev['p'] * 2, fit2.psdev['p'], places=3)


    def test_unpack_p0(self):
        """ _unpack_p0 """
        prior = gv.BufferDict()
        prior['s'] = gv.gvar(0,2.5)
        prior['v'] = [[gv.gvar(1,2),gv.gvar(0,2)]]
        prior = lsqfit._unpack_gvars(prior)
        # p0 is None or dict or array, with or without prior
        for vin,vout in [
        (None,[[1.,0.2]]),
        ([[]],[[1.,0.2]]),
        ([[20.]],[[20.,0.2]]),
        ([[20.,30.]],[[20.,30.]]),
        ([[20.,30.,40.],[100.,200.,300.]],[[20.,30.]])
        ]:
            p0 = None if vin is None else dict(s=10., v=vin, dummy=30.)
            p = lsqfit._unpack_p0(
                p0=p0, p0file=None, prior=prior,
                )
            self.assertEqual(p['s'], 0.25 if p0 is None else p0['s'])
            self.assert_arraysequal(p['v'], vout)
            p = lsqfit._unpack_p0(
                p0=vin, p0file=None, prior=lsqfit._unpack_gvars(prior['v']),
                )
            self.assert_arraysequal(p,vout)
            if vin is not None and np.size(vin) != 0:
                p = lsqfit._unpack_p0(
                    p0=vin, p0file=None, prior=None,
                    )
                self.assert_arraysequal(p,vin)
                p = lsqfit._unpack_p0(
                    p0=p0, p0file=None, prior=None,
                    )
                self.assertEqual(p['s'], p0['s'])
                self.assert_arraysequal(p['v'], p0['v'])

        # p0 is array, with prior
        p0 = [[20., 30.]]
        prior = lsqfit._unpack_gvars(prior['v'])
        p = lsqfit._unpack_p0(p0=p0, p0file=None, prior=prior)
        p0 = np.array(p0)
        self.assert_arraysequal(p, p0)

        # p0 from file
        fn = "test-lsqfit.p"
        p0 = dict(s=10., v=[[20., 30.]])
        with open(fn,"wb") as pfile:
            pickle.dump(p0, pfile)
        for vin,vout in [
        ([[gv.gvar(1, 2)]], [[20.]]),
        ([[gv.gvar(1, 2), gv.gvar(0, 2)]], [[20., 30.]]),
        ([[gv.gvar(1, 2), gv.gvar(0,2.5), gv.gvar(15, 1)]],[[20., 30., 15.]]),
        ]:
            prior = BufferDict()
            prior['s'] = gv.gvar(0, 2.5)
            prior['v'] = vin
            prior = lsqfit._unpack_gvars(prior)
            p = lsqfit._unpack_p0(
                p0=None, p0file=fn, prior=prior,
                )
            self.assert_arraysequal(p['v'], vout)
        os.unlink(fn)
        p = lsqfit._unpack_p0(p0=None, p0file=fn, prior=prior)
        def nonzero_p0(x):
            if not isinstance(x, np.ndarray):
                return x.mean if x.mean!=0 else x.sdev/10.
            else:
                return np.array(
                    [xi.mean if xi.mean!=0 else xi.sdev/10. for xi in x.flat]
                    ).reshape(x.shape)

        self.assertEqual(p['s'],nonzero_p0(prior['s']))
        self.assert_arraysequal(p['v'],nonzero_p0(prior['v']))

        # p0 is True
        prior = BufferDict()
        prior['s'] = gv.gvar(100.0, 2.5)
        prior['v'] = gv.gvar([1., 2.], [3., 4.])
        # prior = lsqfit._unpack_gvars(prior)
        p = lsqfit._unpack_p0(p0=True, p0file=None, prior=prior)
        self.assertTrue('s' in p)
        self.assertTrue('v' in p)
        self.assertGreater(prior['s'].sdev * 10, abs(prior['s'].mean - p['s']))
        self.assertNotEqual(prior['s'].mean, p['s'])

    def test_unpack_gvars(self):
        """ _unpack_gvars """
        # null prior
        p0 = dict(s=10.,v=[[20.,30.]])
        prior = lsqfit._unpack_gvars(None)
        self.assertEqual(prior,None)

        # real prior
        prior = dict(s=gv.gvar(0.,1.),v=[[gv.gvar(0,2.),gv.gvar(1.,3.)]])
        nprior = lsqfit._unpack_gvars(prior)
        self.assertIsInstance(nprior,BufferDict)
        self.assertEqual(nprior.shape,None)
        self.assertTrue(set(prior.keys())==set(nprior.keys()))
        try:
            self.assertItemsEqual(prior.keys(),nprior.keys())
        except AttributeError:
            self.assertCountEqual(prior.keys(),nprior.keys())
        for k in prior:
            self.assert_gvclose(prior[k],nprior[k])

        # symbolic gvars
        prior = dict(s=gv.gvar(0,1),v=[["0(2)",(1,3)]])
        nprior = lsqfit._unpack_gvars(prior)
        self.assertIsInstance(nprior,BufferDict)
        self.assertEqual(nprior.shape,None)
        self.assertTrue(set(prior.keys())==set(nprior.keys()))
        try:
            self.assertItemsEqual(prior.keys(),nprior.keys())
        except AttributeError:
            self.assertCountEqual(prior.keys(),nprior.keys())
        self.assertEqual(nprior['v'].size,2)
        self.assert_gvclose(nprior['s'],gvar(0,1))
        self.assert_gvclose(nprior['v'],[[gvar(0,2),gvar(1,3)]])


    def test_unpack_fcn(self):
        """ _unpack_fcn """
        ydict = BufferDict()
        ydict['s'] = gv.gvar(10.,1.)
        ydict['v'] = [[gv.gvar(20.,2.),gv.gvar(30.,3)]]
        yarray = np.array([1.,2.,3.])
        prdict = BufferDict(dict(p=10*[gv.gvar("1(1)")]))
        prarray = np.array(10*[gv.gvar("1(1)")])
        self.assertEqual(prdict.size,prarray.size)
        self.assert_gvclose(prdict.flat,prarray.flat)
        p0 = list(gv.mean(prarray.flat))
        def fcn_dd(x,p):
            ans = dict(s=sum(p['p']),v=[p['p'][:2]])
            return ans

        def fcn_da(x,p):
            return p['p'][:3]

        def fcn_ad(x,p):
            ans = dict(s=sum(p),v=[p[:2]])
            return ans

        def fcn_aa(x,p):
            return p[:3]

        def fcn_nox_dd(p):
            ans = dict(s=sum(p['p']),v=[p['p'][:2]])
            return ans

        def fcn_nox_da(p):
            return p['p'][:3]

        def fcn_nox_ad(p):
            ans = dict(s=sum(p),v=[p[:2]])
            return ans

        def fcn_nox_aa(p):
            return p[:3]

        # do all combinations of prior and y
        for x, y, pr, fcn, yout in [
            (None, ydict, prdict, fcn_dd, [sum(p0)] + p0[:2]),
            (None, ydict, prarray, fcn_ad, [sum(p0)] + p0[:2]),
            (None, yarray, prdict, fcn_da, p0[:3]),
            (None, yarray, prarray, fcn_aa, p0[:3]),
            (False, ydict, prdict, fcn_nox_dd, [sum(p0)] + p0[:2]),
            (False, ydict, prarray, fcn_nox_ad, [sum(p0)] + p0[:2]),
            (False, yarray, prdict, fcn_nox_da, p0[:3]),
            (False, yarray, prarray, fcn_nox_aa, p0[:3]),
            ]:
            flatfcn = lsqfit._unpack_fcn(fcn=fcn, p0=pr, y=y, x=x)
            fout = flatfcn(np.array(p0))
            self.assert_arraysequal(np.shape(fout), np.shape(yout))
            self.assert_arraysequal(fout, yout)


    def test_y_fcn_match(self):
        # y = dictionary
        y = gv.BufferDict(a=gv.gvar(1,1),b=[gv.gvar(2,2),gv.gvar(3,3)])
        self.assertTrue(lsqfit._y_fcn_match(y,dict(a=1.,b=[2.,3.])))
        self.assertTrue(lsqfit._y_fcn_match(y,gv.BufferDict(a=1.,b=[2.,3.])))
        self.assertFalse(lsqfit._y_fcn_match(y,dict(a=1.,b=[2.,3.,4.])))
        self.assertFalse(lsqfit._y_fcn_match(y,dict(a=1.,b=2.)))
        self.assertFalse(lsqfit._y_fcn_match(y,3.))
        self.assertFalse(lsqfit._y_fcn_match(y,dict(x=1,y=[2,3.])))
        self.assertFalse(lsqfit._y_fcn_match(y,[2.,3.]))

        # y = array
        y = np.array([gv.gvar(2.,2.),gv.gvar(3.,3.)])
        self.assertTrue(lsqfit._y_fcn_match(y,[4.,5.]))
        self.assertFalse(lsqfit._y_fcn_match(y,[4.,5.,6.]))
        self.assertFalse(lsqfit._y_fcn_match(y,dict(a=[4.,5.])))
        self.assertFalse(lsqfit._y_fcn_match(y,3.))

        # y = number
        y = np.array(gv.gvar(1,1))
        self.assertTrue(lsqfit._y_fcn_match(y,3.))
        self.assertFalse(lsqfit._y_fcn_match(y,[4.,5.]))
        self.assertFalse(lsqfit._y_fcn_match(y,dict(a=[4.,5.])))


    def test_reformat(self):
        # arrays
        p = [[1,2],[3,4]]
        newp = lsqfit._reformat(p,[10,20,30,40])
        self.assert_arraysequal(newp,[[10,20],[30,40]])
        with self.assertRaises(ValueError):
            lsqfit._reformat(p,[10,20])
        with self.assertRaises(ValueError):
            lsqfit._reformat(p,[[10,20,30,40]])

        # dictionaries
        p = BufferDict([(0,0),(1,1),(2,2),(3,3)])
        newp = lsqfit._reformat(p,[10,20,30,40])
        self.assert_arraysequal(newp.flat,[10,20,30,40])
        self.assertEqual(newp[0],10)
        self.assertEqual(newp[1],20)
        self.assertEqual(newp[2],30)
        self.assertEqual(newp[3],40)
        with self.assertRaises(ValueError):
            lsqfit._reformat(p,[10,20])
        with self.assertRaises(ValueError):
            lsqfit._reformat(p,[[10,20,30,40]])


    # def test_dump(self):
    #     y = {0 : gv.gvar(1,2), 1 : gv.gvar(3,4)}
    #     prior = {0 : gv.gvar(1.5, 1)}
    #     def f(p):
    #         return {0:p[0],1:p[0]}

    #     fit = nonlinear_fit(data=y,prior=prior,fcn=f)
    #     fit.dump_p("test-lsqfit.p")
    #     p = nonlinear_fit.load_parameters("test-lsqfit.p")
    #     self.assert_gvclose(p[0],fit.p[0])
    #     self.assert_gvclose(p[0],wavg([y[0],y[1],prior[0]]))
    #     fit.dump_pmean("test-lsqfit.p")
    #     pmean = fit.load_parameters("test-lsqfit.p")
    #     self.assertAlmostEqual(pmean[0],fit.pmean[0])
    #     os.remove("test-lsqfit.p")

    def test_partialerr1(self):
        """ fit.p.der """
        # verifies that derivatives in fit.p relate properly to inputs
        #
        # data
        y = gvar(2.,0.125)
        ny = 3
        y = [gvar(y(),y.sdev) for i in range(ny)]

        # prior
        p = gv.BufferDict()
        p.add("y",gvar(0.1,1e4))
        p.add("not y",gvar(3.0,0.125))

        def fcn(x,p):
            """ p['y'] is the average of the y's """
            return np.array(3*[p['y']])

        fit = nonlinear_fit(data=(None,y),fcn=fcn,prior=p,debug=True)
        if PRINT_FIT:
            print(fit.format(nline=100))

        self.assert_gvclose(fit.p['y'],fit.palt['y'])
        self.assert_arraysclose(gv.evalcov(fit.p['y']),gv.evalcov(fit.palt['y']))
        self.assert_gvclose(wavg(y)/fit.p['y'],gvar(1.0,0.0),rtol=1e-6,atol=1e-6)
        self.assert_arraysclose(fit.p['y'].dotder(y[0].der),1./ny)
        self.assert_gvclose(fit.p['not y']/p['not y'],gvar(1.0,0.0),rtol=1e-6,atol=1e-6)
        self.assert_arraysclose(fit.p['not y'].dotder(p['not y'].der),1.0)

        err = partialerrors({"y":fit.p['y'],"not y":fit.p['not y']},
                            {"y":fit.y, "not y":[p["not y"]],
                                "other prior":[p["y"]]})
        self.assertAlmostEqual(err["y","y"],wavg(y).sdev)
        self.assertAlmostEqual(err["y","not y"],0.0)
        self.assertAlmostEqual(err["y","other prior"],0.0,places=5)
        self.assertAlmostEqual(err["not y","not y"],p["not y"].sdev)
        self.assertAlmostEqual(err["not y","y"],0.0)
        self.assertAlmostEqual(err["not y","other prior"],0.0)

    def test_partialerr2(self):
        """ partialerrors """
        # verifies that derivatives in fit.p relate properly to inputs
        #
        # data
        y = gvar(2.,0.125)
        ny = 3
        y = [gvar(y(),y.sdev) for i in range(ny)]

        # prior
        p = gv.BufferDict()
        p["y"] = gvar(0.1,1e4)
        p["not y"] = gvar(3.0,0.125)

        def fcn(x,p):
            """ p['y'] is the average of the y's """
            return np.array(ny*[p['y']])

        fit = nonlinear_fit(data=(None,mean(y),gv.evalcov(y)),fcn=fcn,prior=p,debug=True)
        if PRINT_FIT:
            print( fit.format(nline=100) )
        self.assert_gvclose(fit.p['y'],fit.palt['y'])
        self.assert_arraysclose(gv.evalcov(fit.p['y']),gv.evalcov(fit.palt['y']))
        self.assert_gvclose(wavg(fit.y)/fit.p['y'],gvar(1.0,0.0),rtol=1e-6,atol=1e-6)
        self.assertAlmostEqual(fit.p['y'].dotder(fit.y[0].der),1./ny)
        self.assert_gvclose(fit.p['not y']/p['not y'],gvar(1.0,0.0),rtol=1e-6,atol=1e-6)
        self.assertAlmostEqual(fit.p['not y'].dotder(p['not y'].der),1.0)

        err = partialerrors({"y":fit.p['y'],"not y":fit.p['not y']},
                            {"y":fit.y, "not y":[p["not y"]],
                                "other prior":[p["y"]]})
        self.assertAlmostEqual(err["y","y"],wavg(y).sdev)
        self.assertAlmostEqual(err["y","not y"],0.0)
        self.assertAlmostEqual(err["y","other prior"],0.0,places=5)
        self.assertAlmostEqual(err["not y","not y"],p["not y"].sdev)
        self.assertAlmostEqual(err["not y","y"],0.0)
        self.assertAlmostEqual(err["not y","other prior"],0.0)

    def test_fit_iter(self):
        " fit.simulated_fit_iter "
        y = gv.gvar(['2.1(1.2)', '1.7(5.2)', '2.2(3)', '3.2(1.5)', '1.9(2)'])
        y[1:] = (y[1:] + y[:1]) / 2.
        def fcn(p):
            return len(y) * [p[0]]
        prior = [gv.gvar(2,1)]
        for svdcut in [None, 5e-1]:
            fit = nonlinear_fit(data=y, fcn=fcn, prior=prior, svdcut=svdcut)
            means = []
            N = 100
            for sfit in fit.simulated_fit_iter(N):
                means.append(sfit.p[0].mean)
            self.assertAlmostEqual(fit.p[0].sdev, sfit.p[0].sdev)
            self.assertEqual(prior[0], sfit.prior[0])
            self.assertLess(
                abs(np.average(means) - fit.p[0].mean),
                5 * fit.p[0].sdev / N ** 0.5
                )
        N = 100
        prior_means = []
        for sfit in fit.simulated_fit_iter(N, bootstrap=True):
            prior_means.append(sfit.prior[0].mean)
        prior_mean = np.average(prior_means)
        prior_sdev = np.std(prior_means)
        self.assertLess(abs(prior_mean-prior[0].mean), 5 * prior[0].sdev)
        self.assertLess(abs(prior_sdev/prior[0].sdev - 1), 5./N ** 0.5)

    def test_normal(self):
        " log-normal priors "
        y = gv.gvar([
            '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
            '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
            '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
            '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
            ])
        prior = gv.BufferDict(a = gv.gvar("0.02(2)"))
        def fcn(p, N=len(y)):
            "fit function"
            return N * [p['a']]
        fit = nonlinear_fit(prior=prior, data=y, fcn=fcn)
        self.assertEqual(fit.p['a'].fmt(), "0.004(18)")

    def test_lognormal(self):
        " normal priors "
        y = gv.gvar([
            '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
            '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
            '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
            '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
            ])
        prior = gv.BufferDict()
        prior['log(a)'] = gv.log(gv.gvar("0.02(2)"))
        def fcn(p, N=len(y)):
            "fit function"
            return N * [p['a']]
        fit = nonlinear_fit(prior=prior, data=y, fcn=fcn)
        # self.assertTrue('a' in fit.p)
        self.assertTrue(gv.equivalent(gv.exp(fit.p['log(a)']), fit.p['a']))
        self.assertEqual(fit.p['a'].fmt(), "0.012(11)")

        # arrays
        y = gv.gvar([['0.1(1)', '1(1)'], ['0.2(1)', '2(1)']])
        prior = gv.BufferDict()
        prior['log(a)'] = gv.log(gv.gvar(['0.1(2)', '10(20)']))
        def fcn(p, N=y.shape[0]):
            return [p['a'], p['a']]
        fit = nonlinear_fit(prior=prior, data=y, fcn=fcn)
        # self.assertTrue('a' in fit.p)
        self.assertTrue(gv.equivalent(gv.exp(fit.p['log(a)']), fit.p['a']))
        self.assertEqual(fit.p['a'][0].fmt(),'0.147(69)')
        self.assertEqual(fit.p['a'][1].fmt(), '1.64(69)')

    def test_sqrtnormal(self):
        " sqrt-normal priors "
        y = gv.gvar([
            '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
            '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
            '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
            '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
            ])
        prior = gv.BufferDict()
        prior['sqrt(a)'] = gv.sqrt(gv.gvar("0.02(2)"))
        def fcn(xdummy, p, N=len(y)):
            "fit function"
            return N * [p['a']]
        fit = nonlinear_fit(prior=prior, data=(None,y), fcn=fcn)
        # self.assertTrue('a' in fit.p)
        self.assertTrue(gv.equivalent(fit.p['sqrt(a)'] ** 2, fit.p['a']))
        self.assertEqual(fit.p['a'].fmt(), "0.010(13)")

    def test_linear_dict(self):
        def fcn(p, t=np.arange(0.,1.,0.2)):
            c = p['c']
            E = np.cumsum(p['dE'])
            i = (c != 0)
            return np.sum(
                c[None, i] * gv.exp(-E[None, i] * t[:, None]), axis=1
                )
        prior = gv.gvar(dict(c=2 * ['1.00(1)'], dE= 2 * ['0.500(1)']))
        data = gv.gvar([
            '2.0008(10)', '1.72452(86)', '1.49030(75)',
            '1.29009(65)', '1.12017(57)'
            ])
        fita = nonlinear_fit(prior=prior, data=data, fcn=fcn)
        fitb = nonlinear_fit(prior=prior, data=data, fcn=fcn, linear=['c'])
        self.assertAlmostEqual(fita.chi2, fitb.chi2)
        self.assertTrue(gv.equivalent(fita.p, fitb.p, rtol=1e-1))
        np.testing.assert_allclose(
            gv.mean(fita.p.flat), gv.mean(fitb.p.flat), rtol=1e-5
            )

    def test_linear_array(self):
        def fcn(p, t=np.arange(0.,1.,0.2)):
            c = p[:2]
            E = np.cumsum(p[2:])
            i = (c != 0)
            return np.sum(
                c[None, i] * gv.exp(-E[None, i] * t[:, None]), axis=1
                )
        prior = gv.gvar(2 * ['1.00(1)'] + 2 * ['0.500(1)'])
        data = gv.gvar([
            '2.0008(10)', '1.72452(86)', '1.49030(75)',
            '1.29009(65)', '1.12017(57)'
            ])
        fita = nonlinear_fit(prior=prior, data=data, fcn=fcn)
        fitb = nonlinear_fit(prior=prior, data=data, fcn=fcn, linear=[0,1])
        self.assertAlmostEqual(fita.chi2, fitb.chi2)
        self.assertTrue(gv.equivalent(fita.p, fitb.p, rtol=1e-1))
        np.testing.assert_allclose(
            gv.mean(fita.p.flat), gv.mean(fitb.p.flat), rtol=1e-5
            )

    def test_multifit_exceptions(self):
        """ multifit exceptions """
        y = gv.gvar(["1(1)", "2(1)"])
        prior = gv.gvar(dict(a="0(2)"))
        with self.assertRaises(ValueError):
            def f(p):
                return [p['a']]*3

            fit = nonlinear_fit(data=y, prior=prior, fcn=f, debug=False)
        with self.assertRaises(ZeroDivisionError):
            def f(p):
                1/0.
                return [p['a']]*2

            fit = nonlinear_fit(data=y, prior=prior, fcn=f, debug=False)

    @unittest.skipIf(not hasattr(lsqfit, 'gsl_multifit'), "because GSL not installed")
    def test_gsl_multifit(self):
        """ gsl_multifit """
        from lsqfit import gsl_multifit as multifit
        nx = 3
        xans = np.arange(nx) + 1.
        def f(x, xans=xans):
            return  (x - xans) ** 2 + (x - xans) ** 4
            # return x ** 3 - 3 * x ** 2 * xans + 3 * x * xans **2 - xans ** 3
            # return (x-xans)**3

        ans = multifit(
            x0=np.ones(nx), n=nx, f=f, alg='lm', tol=(1e-10, 0.0, 0.0),
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 1)
        ans = multifit(
            x0=np.zeros(nx), n=nx, f=f, alg='lmaccel', tol=(0.0, 1e-10, 0.0),
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 2)
        ans = multifit(
            x0=np.zeros(nx),n=nx,f=f, tol=(1e-10, 0, 0), alg='subspace2D'
            )
        self.assertEqual(ans.stopping_criterion, 1)
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)

    @unittest.skipIf(not hasattr(lsqfit, 'gsl_v1_multifit'), "because GSL not installed")
    def test_gsl_v1_multifit(self):
        """ gsl_v1_multifit """
        from lsqfit import gsl_v1_multifit as multifit
        nx = 3
        xans = np.arange(nx) + 1.
        def f(x, xans=xans):
            return  (x - xans) ** 2 + (x - xans) ** 4
            # return x ** 3 - 3 * x ** 2 * xans + 3 * x * xans **2 - xans ** 3
            # return (x-xans)**3

        ans = multifit(
            x0=np.ones(nx), n=nx, f=f, alg='lmsder', tol=(1e-10, 0.0, 0.0),
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 1)
        ans = multifit(
            x0=np.zeros(nx), n=nx, f=f, alg='lmder', tol=(0.0, 1e-10, 0.0),
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 2)
        ans = multifit(
            x0=np.zeros(nx),n=nx,f=f, tol=(1e-10, 0, 0), alg='lmniel'
            )
        self.assertEqual(ans.stopping_criterion, 1)
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)

    @unittest.skipIf(not hasattr(lsqfit, 'scipy_least_squares'), "because scipy not installed")
    def test_scipy_least_squares(self):
        """ scipy_multifit """
        from lsqfit import scipy_least_squares as multifit
        nx = 3
        xans = np.arange(nx) + 1.
        def f(x, xans=xans):
            return  (x - xans) ** 2 + (x - xans) ** 4

        ans = multifit(
            x0=np.ones(nx), n=nx, f=f, tol=(1e-15, 1e-8, 1e-15), method='trf',
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 2)
        ans = multifit(
            x0=np.zeros(nx), n=nx, f=f, tol=(1e-8, 1e-15, 1e-15), method='lm',
            )
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)
        self.assertEqual(ans.stopping_criterion, 1)
        ans = multifit(
            x0=np.zeros(nx),n=nx,f=f, tol=(1e-15, 1e-8, 1e-15), method='dogbox',
            )
        self.assertEqual(ans.stopping_criterion, 2)
        self.assert_arraysclose(ans.x, xans, rtol=1e-3)

    @unittest.skipIf(not hasattr(lsqfit, 'scipy_least_squares'), "because scipy not installed")
    def test_bounds(self):
        " scipy_least_squares with bounds "
        data = gv.gvar(['0.9(1)', '2.2(2)'])

        # dictionary p0
        def fcn(p):
            return p['a']
        p0 = dict(a=[0.25, 0.5])
        lower = dict(a=[0.0, 0.0])
        upper = dict(a=[0.5, 1.0])
        fit = nonlinear_fit(
            p0=p0, fcn=fcn, data=data,
            fitter='scipy_least_squares', bounds=(lower, upper),
            )
        self.assertAlmostEqual(fit.pmean['a'][0], 0.5)
        self.assertAlmostEqual(fit.pmean['a'][1], 1.0)

        # array p0
        def fcn(p):
            return p
        p0 = [0.25, 0.5]
        lower = [0.0, 0.0]
        upper = [0.5, 1.0]
        fit = nonlinear_fit(
            p0=p0, fcn=fcn, data=data,
            fitter='scipy_least_squares', bounds=(lower, upper),
            )
        self.assertAlmostEqual(fit.pmean[0], 0.5)
        self.assertAlmostEqual(fit.pmean[1], 1.0)


    def test_fitters(self):
        """ check that fitters work """
        data = gv.gvar(['0.9(1)', '2.2(2)'])
        prior = gv.gvar(['1.0(5)', '2.0(5)'])
        def fcn(p):
            return p
        options = dict(
            gsl_multifit=[
                dict(alg='lm'), dict(alg='lmaccel'), dict(alg='subspace2D'),
                ],
            gsl_v1_multifit=[
                dict(alg='lmsder'), dict(alg='lmder'), dict(alg='lmniel'),
                ],
            scipy_least_squares=[
                dict(method='trf'), dict(method='dogbox'), dict(method='lm'),
                ]
            )
        for k in lsqfit._FITTERS:
            for opt in options[k]:
                fit = nonlinear_fit(
                    fcn=fcn, data=data, prior=prior, fitter=k, **opt
                    )
                self.assertEqual(str(fit.p), '[0.904(98) 2.17(19)]')

    @unittest.skipIf(not hasattr(lsqfit, 'gsl_multiminex'), "because GSL not installed")
    def test_gsl_multiminex_exceptions(self):
        """ gsl_multiminex exceptions """
        from lsqfit import gsl_multiminex as multiminex
        x0 = np.array([6.0,-4.0])
        with self.assertRaises(ZeroDivisionError):
            def f(x):
                1/0.
                ff = (x[0]-5)**2 + (x[1]+3)**2
                return -np.cos(ff)

            ans = multiminex(x0,f)
        #
        with self.assertRaises(TypeError):
            def f(x):
                ff = (x[0]-5)**2 + (x[1]+3)**2
                return [-np.cos(ff)]

            ans = multiminex(x0,f)

    @unittest.skipIf(not hasattr(lsqfit, 'gsl_multiminex'), "because GSL not installed")
    def test_gsl_multiminex(self):
        """ gsl_multiminex """
        from lsqfit import gsl_multiminex as multiminex

        def f(x):
            ff = (x[0]-5)**2 + (x[1]+3)**2
            return -np.cos(ff)

        x0 = np.array([6.0, -4.0])
        ans = multiminex(x0, f, tol=1e-4, step=1.0, alg="nmsimplex")
        self.assert_arraysclose(ans.x,[5.,-3.], rtol=1e-4)
        self.assert_arraysclose(ans.f,-1., rtol=1e-4)
        x0 = np.array([4.0,-2.0])
        ans = multiminex(x0, f, tol=1e-4, step=1.0, alg="nmsimplex2")
        self.assert_arraysclose(ans.x, [5.,-3.], rtol=1e-4)
        self.assert_arraysclose(ans.f, -1., rtol=1e-4)

    @unittest.skipIf(not hasattr(lsqfit, 'scipy_multiminex'), "because scipy not installed")
    def test_scipy_multiminex(self):
        """ gsl_multiminex """
        from lsqfit import scipy_multiminex as multiminex

        def f(x):
            ff = (x[0]-5)**2 + (x[1]+3)**2
            return -np.cos(ff)

        x0 = np.array([6.0, -4.0])
        ans = multiminex(x0, f, tol=1e-4)
        self.assert_arraysclose(ans.x,[5.,-3.], rtol=1e-4)
        self.assert_arraysclose(ans.f, -1., rtol=1e-4)

    def test_gammaQ(self):
        " gammaQ(a, x) "
        cases = [
            (2.371, 5.243, 0.05371580082389009, 0.9266599665892222),
            (20.12, 20.3, 0.4544782602230986, 0.4864172139106905),
            (100.1, 105.2, 0.29649013488390663, 0.6818457585776236),
            (1004., 1006., 0.4706659307021259, 0.5209695379094582),
            ]
        for a, x, gax, gxa in cases:
            np.testing.assert_allclose(gax, gammaQ(a, x), rtol=0.01)
            np.testing.assert_allclose(gxa, gammaQ(x, a), rtol=0.01)

    @unittest.skipIf(
        not hasattr(lsqfit, 'BayesIntegrator'),
        "because no vegas module"
        )
    # @unittest.skipIf(FAST,"for speed")
    def test_bayesintegrator_dict(self):
        " BayesIntegrator(fit) "
        neval = 250
        # linear fit => BayesIntegrator gives same results for everything, norm=1
        x = np.array([0.2, 0.6, 0.8, 1.2, 1.4])
        y = gv.gvar(['0.38(20)', '0.85(20)', '0.59(20)', '1.44(20)', '0.73(20)'])
        prior = gv.BufferDict(c=['0(5)', '0(5)'])
        def f(x, p):
            c = p['c']
            return c[0] + c[1] * x
        fit = nonlinear_fit(data=(x, y), prior=prior, fcn=f)

        expval = BayesIntegrator(fit, limit=7.)
        norm = expval(neval=neval, nitn=5).norm
        self.assertTrue(abs(norm.mean - 1) <= 5. * norm.sdev)

        def g(p):
            c = p['c']
            return dict(mean=c, outer=np.outer(c, c))
        r = expval(g, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        mean = r['mean']
        cov = r['outer'] - np.outer(mean, mean)
        dmean = mean - fit.pmean['c']
        dcov = cov - gv.evalcov(fit.p['c'])
        dnorm = r.norm - 1.
        for dd in [dmean, dcov]:
            self.assertTrue(
                np.all(gv.mean(dd) ** 2 < 25. * gv.var(dd))
                )

        pdf = BayesPDF(fit)
        r = expval(g, neval=neval, nitn=5, pdf=pdf, adapt=False)
        mean = r['mean']
        cov = r['outer'] - np.outer(mean, mean)
        dmean = mean - fit.pmean['c']
        dcov = cov - gv.evalcov(fit.p['c'])
        dnorm = r.norm - 1.
        for dd in [dmean, dcov]:
            self.assertTrue(
                np.all(gv.mean(dd) ** 2 < 25. * gv.var(dd))
                )

        def ga(p):
            return p['c']
        r = expval(ga, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        dmean = r - fit.pmean['c']
        self.assertTrue(
            np.all(gv.mean(dmean) ** 2 < 25. * gv.var(dmean))
            )

        def gs(p):
            return p['c'][0]
        r = expval(gs, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        dmean = r - fit.pmean['c'][0]
        self.assertTrue(abs(dmean.mean) < 5. * dmean.sdev)

    @unittest.skipIf(
        not hasattr(lsqfit, 'BayesIntegrator'),
        "because no vegas module"
        )
    @unittest.skipIf(FAST,"for speed")
    def test_bayesintegrator_array(self):
        " BayesIntegrator(fit) "
        neval = 250
        # linear fit => BayesIntegrator gives same results for everything, norm=1
        x = np.array([0.2, 0.6, 0.8, 1.2, 1.4])
        y = gv.gvar(['0.38(20)', '0.85(20)', '0.59(20)', '1.44(20)', '0.73(20)'])
        prior = gv.gvar(['0(5)', '0(5)'])
        def f(x, p):
            return p[0] + p[1] * x
        fit = nonlinear_fit(data=(x, y), prior=prior, fcn=f)

        expval = BayesIntegrator(fit, limit=7.)
        norm = expval(neval=neval, nitn=5).norm
        self.assertTrue(abs(norm.mean - 1) <= 5. * norm.sdev)

        def g(p):
            return dict(mean=p, outer=np.outer(p, p))
        r = expval(g, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        mean = r['mean']
        cov = r['outer'] - np.outer(mean, mean)
        dmean = mean - fit.pmean
        dcov = cov - gv.evalcov(fit.p)
        dnorm = r.norm - 1.
        for dd in [dmean, dcov]:
            self.assertTrue(
                np.all(gv.mean(dd) ** 2 < 25. * gv.var(dd))
                )

        pdf = BayesPDF(fit)
        r = expval(g, neval=neval, nitn=5, pdf=pdf, adapt=False)
        mean = r['mean']
        cov = r['outer'] - np.outer(mean, mean)
        dmean = mean - fit.pmean
        dcov = cov - gv.evalcov(fit.p)
        dnorm = r.norm - 1.
        for dd in [dmean, dcov]:
            self.assertTrue(
                np.all(gv.mean(dd) ** 2 < 25. * gv.var(dd))
                )

        def ga(p):
            return p
        r = expval(ga, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        dmean = r - fit.pmean
        self.assertTrue(
            np.all(gv.mean(dmean) ** 2 < 25. * gv.var(dmean))
            )

        def gs(p):
            return p[0]
        r = expval(gs, neval=neval, nitn=5, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        dmean = r - fit.pmean[0]
        self.assertTrue(abs(dmean.mean) < 5. * dmean.sdev)

        r = expval(gs, neval=neval, nitn=5, adapt_to_pdf=False, adapt=False)
        self.assertTrue(abs(r.norm.mean - 1) <= 5. * r.norm.sdev)
        dmean = r - fit.pmean[0]
        self.assertTrue(abs(dmean.mean) < 5. * dmean.sdev)

def partialerrors(outputs,inputs):
    err = {}
    for ko in outputs:
        for ki in inputs:
            err[ko,ki] = outputs[ko].partialsdev(*inputs[ki])
    return err


if __name__ == '__main__':
    unittest.main()

