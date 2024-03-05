from __future__ import print_function   # makes this work for python2 and 3

import unittest
import functools
import numpy as np
import gvar as gv
from lsqfit import nonlinear_fit, vegas_fit
import vegas


class test_vegas_fit(unittest.TestCase):
    def setUp(self):
        # sample fit and vfit
        # problem linear so they should agree exactly
        self.neval = 6000 
        self.data = gv.gvar(3 * ['1(1)']) + gv.gvar('0.0(2)')
        self.prior = gv.gvar(dict(a='0(1)', b='0(1)')) 
        self.prior._buf += gv.gvar('0.0(1)')
        self.x = np.array([0.1, 0.5, 0.9])
        self.fcnp = vegas.rbatchintegrand(functools.partial(test_vegas_fit.fcn, self.x))
        self.fit = nonlinear_fit(data=self.data, prior=self.prior, fcn=self.fcnp)
        gv.ranseed(1)
        self.vfit = vegas_fit(
            data=self.data, prior=self.prior, fcn=self.fcnp, 
            param=self.fit.p, neval=self.neval
            )

    @vegas.rbatchintegrand
    @staticmethod
    def fcn(x, p):
        if np.ndim(p['a']) == 1:
            x = x[:, None] 
        return p['a'] + x * p['b']

    def tearDown(self):
        pass

    def test_linear(self):
        " linear least squares fits "
        # nonlinear_fit and vegas_fit should agree
        def compare(fit, vfit):
            self.assertEqual(str(fit.p), str(vfit.p))
            fmt = '{:.2f}'
            self.assertEqual(fit.dof, vfit.dof)
            self.assertEqual(fmt.format(fit.chi2), fmt.format(vfit.chi2))
            if fit.logGBF is None:
                return
            self.assertEqual(fmt.format(fit.logGBF), fmt.format(vfit.logBF.mean))
        # different initializations
        compare(self.fit, self.vfit)
        gv.ranseed(12)
        vfit = vegas_fit(data=self.data, prior=self.prior, fcn=self.fcnp, neval=self.neval)
        compare(self.fit, vfit)
        gv.ranseed(12)
        vfit = vegas_fit(fit=self.fit, neval=self.neval)
        compare(self.fit, vfit)
        gv.ranseed(12)
        vfit = vegas_fit(fit=self.vfit, neval=self.neval)
        compare(self.fit, vfit)

        # uncorrelated data
        data = gv.gvar(3 * ['1(1)']) 
        prior = gv.gvar(dict(a='0(1)', b='0(1)')) 
        fit = nonlinear_fit(data=data, prior=prior, fcn=self.fcnp)
        gv.ranseed(1)
        vfit = vegas_fit(data=data, prior=prior, fcn=self.fcnp, param=fit.p, neval=self.neval)
        compare(fit, vfit)

        # no prior
        data = gv.gvar(['1.0(1)', '2.0(1)', '3.0(1)']) + gv.gvar('0.0(1)')
        p0 = gv.mean(prior)
        fit = nonlinear_fit(data=data, p0=p0, fcn=self.fcnp)
        gv.ranseed(1)
        vfit = vegas_fit(data=data, fcn=self.fcnp, param=fit.p, neval=self.neval)
        compare(fit, vfit)

        # data=(x,y)
        data = (self.x, self.data)
        fcn = test_vegas_fit.fcn
        vfit = vegas_fit(data=data, prior=self.prior, fcn=fcn, param=self.fit.p, neval=self.neval)
        compare(self.fit, vfit)

        # dictionary data
        data = dict(data=self.data)
        @vegas.rbatchintegrand
        def fcn(p):
            return dict(data=self.fcnp(p))
        # fit = nonlinear_fit(data=data, fcn=fcn, prior=self.prior)
        gv.ranseed(1)
        vfit = vegas_fit(data=data, fcn=fcn, prior=self.prior, param=self.fit.p, neval=self.neval)
        compare(self.fit, vfit)

    def test_dump(self):
        " gvar.dumps(self) "
        fit1 = self.vfit 
        s = gv.dumps(fit1)
        fit2 = gv.loads(s)
        self.assertEqual(fit1.format(True), fit2.format(True))

    def test_chi2(self):
        vfit = self.vfit
        # make lbatch p
        p = gv.BufferDict()
        for k in vfit.p:
            p[k] = [vfit.p[k].mean]
        chi2 = -2 * np.log(vfit.pdf(p))
        np.testing.assert_allclose(chi2, vfit.chi2)

    def test_trivial(self):
        " test on trivial case "
        data = gv.gvar('10(2)')
        prior = gv.gvar('10(2)')
        @vegas.rbatchintegrand
        def fcn(p):
            return p
        fit = vegas_fit(data=data, prior=prior, fcn=fcn) # param=wavg)
        p_var = 1. / (1/data.var + 1/prior.var)
        BF = (
            (2 * np.pi * p_var)**0.5      / 
            (2 * np.pi * data.var) ** 0.5  /
            (2 * np.pi * prior.var) ** 0.5
            )
        np.testing.assert_allclose(data.mean, fit.p.mean, rtol=0.01, atol=0.01)
        np.testing.assert_allclose(p_var, fit.p.var, rtol=0.01, atol=0.01)
        np.testing.assert_allclose(np.log(BF), fit.logBF.mean, rtol=0.01, atol=0.01)

    def test_svdcut(self):
        " fit with svdcut "
        vfit = vegas_fit(prior=self.prior, data=self.data, fcn=self.fcnp, svdcut=0.9)
        self.assertGreater(vfit.svdn, 0)

        # swap dictionary and array
        data = dict(data=self.data)
        prior = [self.prior['a'], self.prior['b']]
        @vegas.rbatchintegrand
        def fcn(p):
            p = dict(a=p[0], b=p[1])
            return dict(data=self.fcnp(p))
        vfit = vegas_fit(prior=prior, data=data, fcn=fcn, svdcut=0.9)
        self.assertGreater(vfit.svdn, 0)

if __name__ == '__main__':
    unittest.main()
