#!/usr/bin/env python
# encoding: utf-8
"""
test-lsqfit.py

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

from __future__ import print_function

import os
import sys
import pickle
import unittest
import numpy as np
import gvar as gv
from lsqfit import *

FAST = False         # skips embayes and bootstrap tests

nonlinear_fit.fmt_parameter='%12.3f +- %8.3f'
nonlinear_fit.fmt_prior='(%8.2f +- %8.2f)'
nonlinear_fit.fmt_table_header='%12s%12s%12s%12s\n'
nonlinear_fit.fmt_table_line='%12.3f%12.3f%12.3f%12.3f\n'

PRINT_FIT = False

mean = gv.mean
sdev = gv.sdev

bool_iter = ([True,False][i%2] for i in range(1000))
def print_fit(fit,vd):
    """ print out fit """
    output = '\n'
    if fit.prior is None:
        cd = {'data':[fit.y.flat]}
    else:
        cd = {'data':[fit.y.flat],'priors':[fit.prior.flat]}
    output += fit.fmt_values(vd) + '\n' 
    output += fit.fmt_errorbudget(vd,cd,percent=next(bool_iter))
    output += '\n'+fit.format(nline=1000)+'\n'
    if PRINT_FIT:
        print(output)
    return output   
##

class test_lsqfit(unittest.TestCase):
    def setUp(self):
        """ setup """
        global gvar
        gv.gvar = gv.gvar_factory()
        gvar = gv.gvar
        # gv.ranseed((1969,1974))   # don't use; want different rans each time
        self.label = None
        os.system("rm -f test-lsqfit.p")
    ##
    def tearDown(self):
        global gvar
        gvar = None
        os.system("rm -f test-lsqfit.p")
        # if self.label is not None:
        #     print self.label
    ##
    def assert_gvclose(self,x,y,rtol=1e-5,atol=1e-8,prt=False):
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
    ##
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
    ##
    def assert_arraysequal(self,x,y):
        self.assertEqual(np.asarray(x).tolist(),np.asarray(x).tolist())
    ##
    def t_basicfit(self,yfac,pfac,p0file):
        """ checks means, sdevs, fit.cov, etc in extreme cases """
        ycov = np.array([[2.,.25],[.25,4.]])*yfac
        y = gv.gvar([1.,4.],ycov)
        pr = GPrior()
        pcov = np.array([[2.,.5],[.5,1.]])*pfac
        pr['p'] = gv.gvar([4.,16.],pcov)
        def fcn(x,p):
            return dict(y=p['p']**2)
        ##
        y = dict(y=y)
        fit = nonlinear_fit(data=(None,y),prior=pr,fcn=fcn,p0=p0file,debug=True)
        print_fit(fit,dict(y=wavg(fit.p['p']**2)))
        self.assertEqual(fit.dof,2)
        self.assertAlmostEqual(fit.Q,1.0)
        self.assertAlmostEqual(fit.chi2,0.0)
        cd = {'data':[fit.y.flat],'priors':[fit.prior.flat],'p':[fit.prior['p']]}
        err = {}
        err['data'] = fit.p['p'][1].partialsdev(fit.y.flat)
        err['priors'] = fit.p['p'][1].partialsdev(fit.prior.flat)
        err['p'] = fit.p['p'][1].partialsdev(fit.prior['p'])
        if yfac>100*pfac:
            self.assert_gvclose(fit.p,pr)
            self.assert_arraysclose(fit.pmean['p'],gv.mean(pr)['p'])
            self.assert_arraysclose(fit.psdev['p'],gv.sdev(pr)['p'])
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
    ##
    def test_basicfit(self):
        p0file = "test-lsqfit.p"
        for yf,pf in [(1e22,1),(1,1e22)]:
            self.label = ("nonlinear_fit prior-dominated" 
                if yf>1 else "nonlinear_fit data-dominated")
            self.t_basicfit(yf,pf,p0file)
    ##
    def test_wavg1(self):
        """ fit vs wavg uncorrelated """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)
        ##
        c = gvar(4.,0.125)
        for ny,yf,ryf in [(1,1.,1.),(5,1.,1.),(5,2.,1.),(5,1.,2.)]:
            y = np.array([gvar(c(),yf*c.sdev) for i in range(ny)])
            rooty = np.array([gvar(c()**0.5,ryf*(c**0.5).sdev) 
                                for i in range(ny)])
            pr = dict(rooty=rooty)  # prior
            def fcn(x,p):
                return p['rooty']**2
            ##
            fit = nonlinear_fit(data=(None,y),prior=pr,fcn=fcn,
                            reltol=1e-16,debug=True)
            print_fit(fit,dict(y=wavg(fit.p['rooty']**2)))
            output = avg(fit.p['rooty']**2)
            # self.assertEqual(wavg.dof,ny-1)
            input = wavg([avg(y),avg(rooty**2)])
            self.assertEqual(wavg.dof,1)
            # print("*** wavg1",input,output)
            self.assert_gvclose(input,output,rtol=1e-2)
            if ny>1:  # cov diag
                self.assert_arraysequal(fit.cov,np.diag(np.diag(fit.cov)))
    ##
    def test_wavg2(self):
        """ fit vs wavg correlated """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)
        ##
        c = gvar(4.,0.125)
        for ny,yf,ryf in [(1,1.,1.),(5,1.,1.),(5,1.,1.),(5,2.,1.),(5,1.,2.)]:
            yo = np.array([gvar(c(),yf*c.sdev) for i in range(ny+1)])
            rootyo = np.array([gvar(c()**0.5,ryf*(c**0.5).sdev) 
                                for i in range(ny+1)])
            y = (yo[:-1]+yo[1:])/2.             # introduce correlations
            rooty = (rootyo[:-1]+rootyo[1:])/2
            pr = GPrior()
            pr['rooty'] = rooty
            def fcn(x,p):
                return p['rooty']**2
            ##
            fit = nonlinear_fit(data=(None,y),prior=pr,fcn=fcn,
                                reltol=1e-16,debug=True)
            print_fit(fit,dict(y=wavg(fit.p['rooty']**2)))
            ## check mean and std devn ##
            output = avg(fit.p['rooty']**2)
            input = wavg([avg(yo),avg(rootyo**2)])
            # print("*** wavg2",input,output)
            self.assert_gvclose(input,output,rtol=2e-2)
            ##
            ## check cov matrix ##
            invcovy = np.linalg.inv(gv.evalcov(y))
            invcovry2 = np.linalg.inv(gv.evalcov(rooty**2))
            inv_cov_input = invcovy+invcovry2   # add prior and data in quad.
            cov_output = gv.evalcov(fit.p['rooty']**2) # output cov matrix
            io_prod = np.dot(inv_cov_input,cov_output)  # should be unit matrix
            self.assert_arraysclose(io_prod,np.diag(ny*[1.]),rtol=5e-2)
            if ny>1:        # cov not diag
                self.assertTrue(not np.all(fit.cov==np.diag(np.diag(fit.cov))))
            ##
    ##
    def test_wavg_svd(self):
        """ wavg with svd cut """
        a,b,c = gvar(["1(1)","1(1)","1(1)"])
        var = wavg([(a+b)/2,(a+c)/2.,a],svdcut=1-1e-16).var
        self.assertAlmostEqual(var,0.44)
        var = wavg([(a+b)/2.,(a+c)/2.,a],svdcut=1e-16).var
        self.assertAlmostEqual(var,1./3.)
        var = wavg([b,c,a]).var
        self.assertAlmostEqual(var,1./3.)
    ##
    def test_wavg_vec(self):
        """ wavg of arrays """
        ans = wavg([[gvar(2.1,1.),4+gvar(2.1,1.)],
                    [gvar(1.9,10.),4+gvar(1.9,10.)]])
        self.assert_arraysclose(mean(ans),[2.09802,6.09802],rtol=1e-4)
        self.assert_arraysclose(sdev(ans),[0.995037,0.995037],rtol=1e-4)
    ##
    def test_noprior(self):
        """ fit without prior """
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)
        ##
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
        ##
        def dictfcn(x,p):
            return dict(y=p**2) # np.array([p[k]**2 for k in p]))
        ##
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
            fit = nonlinear_fit(data=data,p0=p0,fcn=fcn,debug=True,reltol=1e-16)
            print_fit(fit,dict(y=avg(fit.p**2)))
            self.assertIsNone(fit.logGBF)
            self.assertEqual(fit.dof,0.0)
            self.assertAlmostEqual(fit.chi2,0.0,places=4)
            self.assert_arraysclose(gv.evalcov(fit.p**2),
                                        dataycov,rtol=1e-4)
            self.assert_gvclose(fit.p**2,datay,1e-4)
    ##
    @unittest.skipIf(FAST,"skipping test_bootstrap for speed")
    def test_bootstrap(self):
        """ bootstrap_iter """
        ## data and priors ## 
        def bin(y): # correlates different y's 
            return (y[:-1]+y[1:])/2.
        ##
        def avg(x): # compute of avg of sequence
            return sum(x)/len(x)
        ##
        yc = gvar(4.,0.25)
        p2c = gvar(4.,0.25)
        ny = 3
        y = np.array([gvar(yc(),yc.sdev) for i in range(ny)])
        p = np.array([gvar(p2c(),p2c.sdev)**0.5 for i in range(ny)])
        eps = gvar(1.,1e-4)
        ##
        cases = [(y[:-1],p[:-1],False),
                (bin(y),bin(p),False),
                (bin(y),bin(p),True),
                (bin(y),p[:-1],False),
                (y[:-1],bin(p),False),
                (y[:-1]*eps,p[:-1]*eps**0.5,False),
                (y[:-1],None,False),
                (y[:-1],None,True),
                (bin(y),None,False)]
        for y,p,use_dlist in cases:
            ## fit then bootstrap ##
            prior = None if p is None else np.array(p) 
            p0 = gv.mean(y)**0.5 if p is None else None
            data = None,y
            def fcn(x,p):
                return p**2 # np.array([p[k]**2 for k in p])
            ##
            fit = nonlinear_fit(data=data,fcn=fcn,prior=prior,p0=p0,debug=True)
            # print(fit.format(nline=100))
            def evfcn(p):
                return {0:np.average(p**2)} # [p[k]**2 for k in p])
            ##
            bs_ans = gv.dataset.Dataset()
            nbs = 1000/5
            fit_iter = fit.bootstrap_iter(n=None if use_dlist else nbs,
                        datalist=(((None,yb) for yb in gv.bootstrap_iter(y,nbs))
                                    if use_dlist else None))
            for bfit in fit_iter:
                if bfit.error is None:
                    bs_ans.append(evfcn(bfit.pmean))
            bs_ans = gv.dataset.avg_data(bs_ans,median=True,spread=True)[0]
            target_ans = wavg([avg(y),avg(p**2)]) if p is not None else avg(y)
            fit_ans = avg(fit.p**2)
            rtol = 10.*fit_ans.sdev/nbs**0.5  # 10 sigma
            # print(bs_ans,fit_ans,target_ans,rtol)
            self.assert_gvclose(target_ans,fit_ans,rtol=rtol)
            self.assert_gvclose(bs_ans,fit_ans,rtol=rtol)
            ##
    ##
    def test_svd(self):
        """ svd cuts """
        ## data and priors ## 
        fac = 100.
        rtol = 1/fac
        sig1 = 1./fac
        sig2 = 1e-2/fac
        y0 = gvar(1.,sig1)*np.array([1,1])+gvar(0.1,sig2)*np.array([1,-1])
        y = y0+next(gv.raniter(y0))-gv.mean(y0)
        p02 = gvar(1.,sig1)*np.array([1,1])+gvar(0.1,sig2)*np.array([1,-1])
        p = (p02+next(gv.raniter(p02))-gv.mean(p02))**0.5
        eps = gvar(1.,1.e-8)
        reps = eps**0.5
        ##
        cases = [(y,p,1e-20),(y,p,1e-2),(y*eps,p*reps,1e-20),(y*eps,p*reps,1e-2),
                ((gv.mean(y),gv.evalcov(y)),p,1e-20),
                ((gv.mean(y),gv.evalcov(y)),p,1e-2)]
        for y,p,svdcut in cases:
            ## fit then bootstrap ##
            prior = np.array(p)
            if not isinstance(y,tuple):
                data = None,y
            else:
                data = (None,)+y
            def fcn(x,p):
                return p**2 # np.array([p[k]**2 for k in p])
            ##
            fit = nonlinear_fit(data=data,fcn=fcn,prior=prior,
                                svdcut=(svdcut,svdcut),debug=True)
            # print(fit.format(nline=100))
            y = fit.y.flatten()
            pr = fit.prior.flatten()
            p = fit.p.flatten()
            ans_y = [(y[0]+y[1])/2,(y[0]-y[1])/2]
            ans_pr = [(pr[0]**2+pr[1]**2)/2,(pr[0]**2-pr[1]**2)/2]
            ans_p = [(p[0]**2+p[1]**2)/2,(p[0]**2-p[1]**2)/2]
            target_ans = wavg([ans_y,ans_pr])
            fit_ans = np.array(ans_p)
            self.assert_gvclose(target_ans,fit_ans,rtol=rtol)
            s2 = max(fit_ans[0].sdev*sig2/sig1,svdcut**0.5*fit_ans[0].sdev)
            self.assertAlmostEqual(fit_ans[1].sdev/s2,1.,places=2)
            ##
    ##
    @unittest.skipIf(FAST,"skipping test_empbayes for speed")
    def test_empbayes(self):
        """ empbayes fit """
        ## data ##
        def y0(mean,sdev=0.25,ny=8):
            ans = np.array([(gvar(mean(),sdev) if sdev>0 else mean())
                            for i in range(ny+1)])
            return (ans[:-1]+ans[1:])/2.
        ##
        ysdev = 0.25
        ymean = 4.0
        ny = 30*4         # larger ny means end result closer to fac
        fac = 100.      # pushes starting point away from answer
        y = y0(gvar(ymean,ysdev),sdev=ysdev,ny=ny)
        ##
        p2r = gvar(ymean,ysdev)
        p2mean = np.array([p2r() for i in range(ny)])
        def fitargs(x):
            p2 = p2mean + [gvar(0,np.exp(x[0])*ysdev/fac) for i in range(ny)]
            # p2 = (p2[:-1]+p2[1:])/2
            p = p2**0.5
            return dict(prior=np.array(p),data=(None,y),
                        fcn=(lambda xx,p : np.array(p**2)))
        ##
        x0 = np.array([0.01])
        if PRINT_FIT:
            def analyzer(x,f,it):
                print("%3d  %.3f  ->  %.4f" % (it,np.exp(x)[0],f))
            ##
        else:
            analyzer = None
        fit,x = empbayes_fit(x0,fitargs,analyzer=analyzer,tol=1e-5)
        # print('empbayes -- np.exp(x[0]),fac:',np.exp(x[0]),fac)
        self.assertAlmostEqual(np.exp(x[0]),fac,delta=fac*0.35)
    ## 
    def test_unpack_data(self):
        """ fit._unpack_data """
        yr = gvar(4.,.25)
        ny = 8
        y = np.array([gvar(yr(),0.25) for i in range(ny)]) 
        ymean = np.array([yi.mean for yi in y])
        ysdev = np.array([yi.sdev for yi in y])
        ycov = np.diag(ysdev**2)
        ycorr = (y[:-1]+y[1:])/2
        ycorrcov = gv.evalcov(ycorr)
        ycorrmean = np.array([yi.mean for yi in ycorr])
        x = 1
        ydict = dict(y=y,z=gvar(16.,4.))
        ydicto = BufferDict()
        for k in ydict:
            ydicto[k] = ydict[k]
        ydictmean = np.array([yi.mean for yi in ydicto.flat])
        ydictcov = np.diag(gv.evalcov(ydicto.flat))
        ycorrdict = dict(y=ycorr)
        ycorrdicto = BufferDict()
        prior = GPrior(dict(c=gvar(0,1)))
        for k in ycorrdict:
            ycorrdicto.add(k,ycorrdict[k])
        for input,output in [  #]
            ((x,ymean,ysdev),(ymean,ysdev**2)),
            ((x,ycorr),(ycorrmean,ycorrcov)),
            ((x,ydict),(ydictmean,ydictcov)),
            ((x,ycorrdict),(ycorrmean,ycorrcov,ycorrdict.keys()))
            ]:
            # fit = nonlinear_fit(data=input)
            prior = nonlinear_fit._unpack_gvars(prior)
            xn,ydict = nonlinear_fit._unpack_data(data=input,prior=prior,
                                    svdcut=(None,None),svdnum=(None,None))[:2]
            yn = gv.mean(ydict.flat)
            ycovn = gv.evalcov(ydict.flat)
            if np.all(ycovn==np.diag(np.diag(ycovn))):
                ycovn = np.diag(ycovn)
            self.assertEqual(xn,x)
            self.assert_arraysequal(yn,output[0])
            self.assert_arraysequal(ycovn,output[1])
            self.assert_arraysequal(yn,gv.mean(ydict.flat))
            cov_yd = gv.evalcov(ydict.flat)
            if np.all(np.diag(np.diag(cov_yd))==cov_yd):
                cov_yd = np.diag(cov_yd)
            self.assert_arraysclose(ycovn,cov_yd)
            if hasattr(input[1],'keys'):
                self.assertIsInstance(ydict,BufferDict)
                self.assertEqual(sorted(input[1].keys()),sorted(ydict.keys()))
            else:
                self.assertIsInstance(ydict,np.ndarray)
                self.assertTrue(ydict.shape==input[1].shape)
    ##
    def test_unpack_p0(self):
        """ _unpack_p0 """
        f = nonlinear_fit
        prior = gv.BufferDict()
        prior['s'] = gv.gvar(0,2.5)
        prior['v'] = [[gv.gvar(1,2),gv.gvar(0,2)]]
        prior = nonlinear_fit._unpack_gvars(prior)
        ## p0 is None or dict or array, with or without prior ##
        for vin,vout in [
        (None,[[1.,0.2]]),
        ([[]],[[1.,0.2]]),
        ([[20.]],[[20.,0.2]]),
        ([[20.,30.]],[[20.,30.]]),
        ([[20.,30.,40.],[100.,200.,300.]],[[20.,30.]])
        ]:
            p0 = None if vin is None else dict(s=10.,v=vin)
            p = f._unpack_p0(p0=p0,p0file=None,prior=prior)
            self.assertEqual(p['s'],0.25 if p0 is None else p0['s'])
            self.assert_arraysequal(p['v'],vout)
            p = f._unpack_p0(p0=vin,p0file=None,prior=nonlinear_fit._unpack_gvars(prior['v']))
            self.assert_arraysequal(p,vout)
            if vin is not None and np.size(vin)!=0:
                p = f._unpack_p0(p0=vin,p0file=None,prior=None)
                self.assert_arraysequal(p,vin)
                p = f._unpack_p0(p0=p0,p0file=None,prior=None)
                self.assertEqual(p['s'],p0['s'])
                self.assert_arraysequal(p['v'],p0['v'])
        ##
        ## p0 is array, with prior ##
        p0 = [[20.,30.]]
        prior = nonlinear_fit._unpack_gvars(prior['v'])
        p = f._unpack_p0(p0=p0,p0file=None,prior=prior)
        p0 = np.array(p0)
        self.assert_arraysequal(p,p0)
        ##
        ## p0 from file ##
        fn = "test-lsqfit.p"
        p0 = dict(s=10.,v=[[20.,30.]])
        with open(fn,"wb") as pfile:
            pickle.dump(p0,pfile)
        for vin,vout in [
        ([[gv.gvar(1,2)]],[[20.]]),
        ([[gv.gvar(1,2),gv.gvar(0,2)]],[[20.,30.]]),
        ([[gv.gvar(1,2),gv.gvar(0,2.5),gv.gvar(15,1)]],[[20.,30.,15.]]),
        ]:
            prior = BufferDict()
            prior['s'] = gv.gvar(0,2.5)
            prior['v'] = vin
            prior = nonlinear_fit._unpack_gvars(prior)
            p = f._unpack_p0(p0=None,p0file=fn,prior=prior)
            self.assert_arraysequal(p['v'],vout)
        os.system("rm -f "+fn)
        p = f._unpack_p0(p0=None,p0file=fn,prior=prior)
        def nonzero_p0(x):
            if not isinstance(x,np.ndarray):
                return x.mean if x.mean!=0 else x.sdev/10.
            else:
                return np.array([xi.mean if xi.mean!=0 else xi.sdev/10. 
                                for xi in x.flat]).reshape(x.shape)
        ##
        self.assertEqual(p['s'],nonzero_p0(prior['s']))
        self.assert_arraysequal(p['v'],nonzero_p0(prior['v']))
        ##
    ##
    def test_unpack_gvars(self):
        """ _unpack_gvars """
        ## null prior ##
        p0 = dict(s=10.,v=[[20.,30.]])
        f = nonlinear_fit
        prior = f._unpack_gvars(None)
        self.assertEqual(prior,None)
        ##
        ## real prior ##
        prior = dict(s=gv.gvar(0.,1.),v=[[gv.gvar(0,2.),gv.gvar(1.,3.)]])
        nprior = f._unpack_gvars(prior)
        self.assertIsInstance(nprior,BufferDict)
        self.assertEqual(nprior.shape,None)
        self.assertTrue(set(prior.keys())==set(nprior.keys()))
        try:
            self.assertItemsEqual(prior.keys(),nprior.keys())
        except AttributeError:
            self.assertCountEqual(prior.keys(),nprior.keys())
        for k in prior:
            self.assert_gvclose(prior[k],nprior[k])
        ##
        ## symbolic gvars ##
        prior = dict(s=gv.gvar(0,1),v=[["0(2)",(1,3)]])
        nprior = f._unpack_gvars(prior)
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
        ##
    ##
    def test_unpack_fcn(self):
        """ _unpack_fcn """
        ydict = BufferDict()
        ydict['s'] = gv.gvar(10.,1.)
        ydict['v'] = [[gv.gvar(20.,2.),gv.gvar(30.,3)]]
        yarray = np.array([1.,2.,3.])
        prdict = GPrior(dict(p=10*[gv.gvar("1(1)")]))
        prarray = np.array(10*[gv.gvar("1(1)")])
        self.assertEqual(prdict.size,prarray.size)
        self.assert_gvclose(prdict.flat,prarray.flat)
        p0 = list(gv.mean(prarray.flat))
        def fcn_dd(x,p):
            ans = dict(s=sum(p['p']),v=[p['p'][:2]])
            return ans
        ##
        def fcn_da(x,p):
            return p['p'][:3]
        ##
        def fcn_ad(x,p):
            ans = dict(s=sum(p),v=[p[:2]])
            return ans
        ##
        def fcn_aa(x,p):
            return p[:3]
        ##
        ## do all combinations of prior and y ##
        for y,pr,fcn,yout in [
        (ydict,prdict,fcn_dd,[sum(p0)]+p0[:2]),
        (ydict,prarray,fcn_ad,[sum(p0)]+p0[:2]),
        (yarray,prdict,fcn_da,p0[:3]),
        (yarray,prarray,fcn_aa,p0[:3])
        ]:
            f = nonlinear_fit
            fcn = f._unpack_fcn(fcn=fcn,p0=pr,y=y)
            fout = fcn(None,np.array(p0))
            self.assert_arraysequal(np.shape(fout),np.shape(yout))
            self.assert_arraysequal(fout,yout)
        ##
    ##
    def test_partialerr1(self):
        """ fit.p.der """
        # verifies that derivatives in fit.p relate properly to inputs
        #
        ## data ##
        y = gvar(2.,0.125)
        ny = 3
        y = [gvar(y(),y.sdev) for i in range(ny)]
        ##
        ## prior ##
        p = GPrior()
        p.add("y",gvar(0.1,1e4))
        p.add("not y",gvar(3.0,0.125))
        ##
        def fcn(x,p):
            """ p['y'] is the average of the y's """
            return np.array(3*[p['y']])
        ##
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
                            {"y":[fit.y.flat], "not y":[p["not y"]],
                                "other prior":[p["y"]]})
        self.assertAlmostEqual(err["y","y"],wavg(y).sdev)
        self.assertAlmostEqual(err["y","not y"],0.0)
        self.assertAlmostEqual(err["y","other prior"],0.0,places=5)
        self.assertAlmostEqual(err["not y","not y"],p["not y"].sdev)
        self.assertAlmostEqual(err["not y","y"],0.0)
        self.assertAlmostEqual(err["not y","other prior"],0.0)
    ##
    def test_partialerr2(self):
        """ partialerrors """
        # verifies that derivatives in fit.p relate properly to inputs
        #
        ## data ##
        y = gvar(2.,0.125)
        ny = 3
        y = [gvar(y(),y.sdev) for i in range(ny)]
        ##
        ## prior ##
        p = GPrior()
        p["y"] = gvar(0.1,1e4)
        p["not y"] = gvar(3.0,0.125)
        ##
        def fcn(x,p):
            """ p['y'] is the average of the y's """
            return np.array(ny*[p['y']])
        ##
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
                            {"y":[fit.y.flat], "not y":[p["not y"]],
                                "other prior":[p["y"]]})
        self.assertAlmostEqual(err["y","y"],wavg(y).sdev)
        self.assertAlmostEqual(err["y","not y"],0.0)
        self.assertAlmostEqual(err["y","other prior"],0.0,places=5)
        self.assertAlmostEqual(err["not y","not y"],p["not y"].sdev)
        self.assertAlmostEqual(err["not y","y"],0.0)
        self.assertAlmostEqual(err["not y","other prior"],0.0)
    ##   
    def test_multifit(self):
        nx = 3
        x0 = np.arange(nx)+1.
        def f(x,x0=x0):
            return (x-x0)**3
        ##
        if PRINT_FIT:
            def tabulate(x,f,df):
                print(x[:3],'->',f[:3])
            ##
        else:
            tabulate = None
        ans = multifit(x0=np.ones(nx),n=nx,f=f,analyzer=tabulate,
                        alg='lmsder')
        self.assert_arraysclose(ans.x,x0,rtol=1e-3)
        ans = multifit(x0=np.zeros(nx),n=nx,f=f,analyzer=tabulate,
                        alg='lmder')
        self.assert_arraysclose(ans.x,x0,rtol=1e-3)
    ##
    def test_multiminex(self):
        def f(x):
            ff = (x[0]-5)**2 + (x[1]+3)**2
            return -np.cos(ff)
        ##
        if PRINT_FIT:
            def tabulate(x,f,it):
                print(it,x,'->',f)
            ##
        else:
            tabulate = None
        x0 = np.array([6.0,-4.0])
        ans = multiminex(x0,f,tol=1e-4,step=1.0,
                            analyzer=tabulate,alg="nmsimplex")
        self.assert_arraysclose(ans.x,[5.,-3.],rtol=1e-4)
        self.assert_arraysclose(ans.f,-1.,rtol=1e-4)
        x0 = np.array([4.0,-2.0])
        ans = multiminex(x0,f,tol=1e-4,step=1.0,
                            analyzer=tabulate,alg="nmsimplex2")
        self.assert_arraysclose(ans.x,[5.,-3.],rtol=1e-4)
        self.assert_arraysclose(ans.f,-1.,rtol=1e-4)
    ##
##

def partialerrors(outputs,inputs):
    err = {}
    for ko in outputs:
        for ki in inputs:
            err[ko,ki] = outputs[ko].partialsdev(*inputs[ki])
    return err
##
    
    
    
if __name__ == '__main__':
    unittest.main()

