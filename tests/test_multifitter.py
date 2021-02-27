from __future__ import print_function   # makes this work for python2 and 3

import unittest
import inspect
import os 
import numpy as np
import gvar as gv
import lsqfit
from lsqfit import MultiFitter, MultiFitterModel

class test_multifitter(unittest.TestCase):
    def setUp(self):
        gv.ranseed(1)
        # a = gv.gvar('1.000(1)')
        # b = gv.gvar('0.500(1)')
        a = gv.gvar('1.0(1)')
        b = gv.gvar('0.5(1)')
        self.x = np.array([0.1, 0.2, 0.3, 0.4])
        def fcn(p, x=self.x):
            ans = gv.BufferDict()
            ans['l'] = p['a'] + p['b'] * x
            ans['c1'] = 4 * [p['a']]
            ans['c2'] = 4 * [p['a']]
            return ans
        self.prior = gv.BufferDict([('a', a), ('b', b)])
        self.data = gv.make_fake_data(fcn(self.prior))
        self.fcn = fcn
        # reference fit without using MultiFitter
        self.ref_fit = lsqfit.nonlinear_fit(
            prior=self.prior, fcn=self.fcn, data=self.data
            )
        # these data should be ignored
        self.data['dummy'] = gv.gvar(['1(1)', '2(2)'])

    def agree_ref(self, p):
        r = self.ref_fit.p 
        for k in ['a', 'b']:
            if abs(r[k].mean - p[k].mean) > r[k].sdev:
                return False 
            if abs(r[k].sdev - p[k].sdev) > r[k].sdev:
                return False 
        return True

    def make_models(self, ncg):
        return [
            Linear(datatag='l', a='a', b='b', x = self.x, ncg=ncg),
            Constant(datatag='c1', a='a', ndata=4, ncg=ncg),
            Constant(datatag='c2', a='a', ndata=4, ncg=ncg),
            ]

    def tearDown(self):
        pass

    def test_flatten_models(self):
        " MultiFitter.flatten_models "
        models = self.make_models(ncg=1)
        alltags = [m.datatag for m in models]

        models = self.make_models(ncg=1)
        models = [models[0], tuple(models[1:])]
        tags = [m.datatag for m in MultiFitter.flatten_models(models)]
        self.assertEqual(alltags, tags)

        models = self.make_models(ncg=1)
        models = [models[0], models[1:]]
        tags = [m.datatag for m in MultiFitter.flatten_models(models)]
        self.assertEqual(alltags, tags)

        models = self.make_models(ncg=1)
        models = [models[0], [(models[0], models[1]), (models[0], models[2])]]
        tags = [m.datatag for m in MultiFitter.flatten_models(models)]
        self.assertEqual(alltags, tags)

        models = self.make_models(ncg=1)
        models = [models[0], [tuple(models[1:]),()]]
        tags = [m.datatag for m in MultiFitter.flatten_models(models)]
        self.assertEqual(alltags, tags)

        models = self.make_models(ncg=1)
        models = [
            (), models[0], [], (models[1], (), models[2], () ),
            [(), tuple(models[1:]),()], [], [()]
            ]
        tags = [m.datatag for m in MultiFitter.flatten_models(models)]
        self.assertEqual(alltags, tags)

        models = self.make_models(ncg=1)
        tags = [m.datatag for m in MultiFitter.flatten_models(models[0])]
        self.assertEqual(alltags[:1], tags)

    def test_builddata(self):
        " MultiFitter.builddata "
        # ncg=1
        fitter = MultiFitter(models=self.make_models(ncg=1))
        data = fitter.builddata(data=self.data, mopt=fitter.mopt)
        del self.data['dummy']
        self.assertEqual(str(data), str(self.data))

        # ncg=2
        simpledata = gv.BufferDict()
        simpledata['l'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata['c1'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata['c2'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata = gv.gvar(simpledata)
        fitter = MultiFitter(models=self.make_models(ncg=2))
        data = fitter.builddata(data=simpledata, mopt=fitter.mopt)
        for k in simpledata:
            self.assertEqual(str(data[k]), '[1.50(25) 3.50(25)]')

    def test_builddata_pdata(self):
        " MultiFitter.builddata(pdata=..., ...) "
        # ncg=1
        fitter = MultiFitter(models=self.make_models(ncg=1))
        data = fitter.builddata(data=self.data, mopt=fitter.mopt)
        data['extra'] = np.array([1.])
        ndata = fitter.builddata(pdata=data, mopt=fitter.mopt)
        del data['extra']
        self.assertEqual(str(data), str(ndata))

        # ncg=2
        fitter = MultiFitter(models=self.make_models(ncg=2))
        data = fitter.builddata(data=self.data, mopt=fitter.mopt)
        data['extra'] = np.array([1.])
        ndata = fitter.builddata(pdata=data, mopt=fitter.mopt)
        del data['extra']
        self.assertEqual(str(data), str(ndata))

    def test_builddata_marginalized(self):
        " MultiFitter.builddata with marginalization "
        fitter = MultiFitter(models=self.make_models(ncg=1), mopt=True)
        data = fitter.builddata(data=self.data, prior=self.prior, mopt=fitter.mopt)
        self.assertEqual(
            str(self.data['l'] - self.prior['b'] * self.x),
            str(data['l']),
            )
        self.assertEqual(str(self.data['c1']), str(data['c1']))
        self.assertEqual(str(self.data['c2']), str(data['c2']))

    def test_buildprior(self):
        " MultiFitter.buildprior "
        prior = gv.BufferDict(self.prior)
        prior['dummy'] = gv.gvar('12(12)')
        fitter = MultiFitter(models=self.make_models(ncg=1))
        prior = fitter.buildprior(prior=prior)
        self.assertEqual(str(prior), str(self.prior))

    def test_buildprior_fast(self):
        " MultiFitter.buildprior with fast=False "
        prior = gv.BufferDict(self.prior)
        prior['dummy'] = gv.gvar('12(12)')
        fitter = MultiFitter(models=self.make_models(ncg=1), fast=False)
        newprior = fitter.buildprior(prior=prior)
        self.assertEqual(str(prior), str(newprior))

    def test_set(self):
        " fitter.set(...) "
        keys = ['fast', 'mopt', 'ratio', 'wavg_kargs', 'wavg_all', 'fitterargs', 'fitname']
        def collect_args(f):
            return {k : getattr(f, k) for k in keys}
        # 0
        fitter = MultiFitter(
            models=self.make_models(ncg=1),
            fast=True, mopt=True, alg='dogleg', ratio=True
            )
        args0 = collect_args(fitter)
        # 1
        kargs, oldkargs = fitter.set(
            fast=False, mopt=False, alg='lm', maxit=10
            )
        args1 = collect_args(fitter)
        self.assertEqual(args1, kargs)
        self.assertEqual(set(kargs.keys()), set(keys))
        self.assertEqual(
            oldkargs,
            dict(fast=True, mopt=True, fitterargs=dict(alg='dogleg'))
            )
        self.assertEqual(args1['fitterargs'], dict(alg='lm', maxit=10))
        # 2
        fitter.set(**oldkargs)
        args2 = collect_args(fitter)
        self.assertEqual(args0, args2)
        # 3
        kargs, oldkargs = fitter.set(fitterargs=dict(alg='subspace2D'), maxit=100)
        args3 = collect_args(fitter)
        self.assertEqual(kargs, args3)
        self.assertEqual(set(kargs.keys()), set(keys))
        self.assertEqual(oldkargs, dict(fitterargs=dict(alg='dogleg')))
        self.assertEqual(args3['fitterargs'], dict(alg='subspace2D', maxit=100))
        # 4
        fitter.set(**oldkargs)
        args4 = collect_args(fitter)
        self.assertEqual(args0, args4)

    def test_buildprior_marginalized(self):
        " MultiFitter.buildprior with marginalization"
        prior = gv.BufferDict(self.prior)
        prior['dummy'] = gv.gvar('12(12)')
        fitter = MultiFitter(models=self.make_models(ncg=1), mopt=True)
        prior = fitter.buildprior(prior=prior, mopt=fitter.mopt)
        del self.prior['b']
        self.assertEqual(str(prior), str(self.prior))

    def test_buildfitfcn(self):
        " MultiFitter.buildfitfcn "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fcn = fitter.buildfitfcn()
        self.assertEqual(
            str(fcn(self.prior)), str(self.fcn(self.prior))
            )

    def test_lsqfit(self):
        " MultiFitter.lsqfit "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit2 = fitter.lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(self.ref_fit.format(), fit2.format())

    def test_lsqfit_p0(self):
        " MultiFitter.lsqfit with p0 "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        p0 = {'a': 0.9992476083689589,'b': 0.4996757090188109}
        fit2 = fitter.lsqfit(data=self.data, prior=self.prior, p0=p0)
        self.assertEqual(self.ref_fit.format()[:-22], fit2.format()[:-22])
        self.assertEqual(p0, fit2.p0)
        fit2 = fitter.lsqfit(data=self.data, prior=self.prior, p0=3 * [p0])
        self.assertEqual(self.ref_fit.format()[:-20], fit2.format()[:-20])
        self.assertEqual(p0, fit2.p0)
        fn = 'test_multifitter.p'
        fit1 = fitter.lsqfit(data=self.data, prior=self.prior, p0=fn)
        fit1 = fitter.lsqfit(data=self.data, prior=self.prior, p0=fn)
        # should be converged
        fit2 = fitter.lsqfit(data=self.data, prior=self.prior, p0=fn)
        self.assertTrue(self.agree_ref(fit2.p))
        self.assertEqual(fit1.format()[:-70], fit2.format()[:len(fit1.format())-70])
        self.assertEqual(fit1.pmean, fit2.p0)
        os.unlink(fn)


    def test_dump_lsqfit(self):
        " MultiFitter.lsqfit "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        dfit = gv.loads(gv.dumps(
            fitter.lsqfit(data=self.data, prior=self.prior)
            ))
        self.assertEqual(self.ref_fit.format(True)[:-2], dfit.format(True)[:-2])

    def test_lsqfit_coarse_grain(self):
        " MultiFitter.lsqfit(..., ncg=2) "
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit3 = fitter.lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit3.p))

    def test_lsqfit_pdata_coarse_grain(self):
        " MultiFitter.lsqfit(pdata=..., ..., ncg=2) "
        fitter = MultiFitter(models=self.make_models(ncg=2))
        pdata = MultiFitter.process_data(data=self.data, models=fitter.models)
        fit3 = fitter.lsqfit(pdata=pdata, prior=self.prior)
        self.assertTrue(self.agree_ref(fit3.p))

    def test_marginalization(self):
        " MultiFitter.lsqfit(..., mopt=...) "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit4 = fitter.lsqfit(data=self.data, prior=self.prior, mopt=True)
        self.assertEqual(str(fit4.p['a']), str(self.ref_fit.p['a']))
        self.assertEqual(gv.fmt_chi2(fit4), gv.fmt_chi2(self.ref_fit))
        self.assertTrue('b' not in fit4.p)

    def test_extend(self):
        " MultiFitter.lsqfit(...) "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fit5 = fitter.lsqfit(data=self.data, prior=prior)
        self.assertTrue(self.agree_ref(fit5.p))
        self.assertTrue(abs(fit5.chi2 - self.ref_fit.chi2) / 0.1 /self.ref_fit.chi2)
        self.assertTrue('log(a)' in fit5.p)

    def test_fast(self):
        " MultiFitter.lsqfit(..., fast=False) "
        # with fast=False
        self.prior['aa'] = self.prior['a'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit6 = fitter.lsqfit(data=self.data, prior=self.prior, fast=False)
        self.assertTrue(self.agree_ref(fit6.p))
        self.assertEqual((fit6.p['a']/fit6.p['aa']).fmt(ndecimal=5), '1.00000(0)')

        # with fast=True (default)
        self.prior['aa'] = self.prior['a'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit7 = fitter.lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit7.p))
        self.assertTrue('aa' not in fit7.p)

    def test_chained_lsqfit(self):
        " MultiFitter.chained_lsqfit(...) "
        # sequential fit
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit1 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])

        # with coarse grain, marginalization and extend, and with fast=False
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b']),
            ])
        prior['log(aa)'] = prior['log(a)'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=2), fast=False)
        fit2 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True
            )
        self.assertTrue(self.agree_ref(fit2.p))

    def test_chained_lsqfit_keyword(self):
        " MultiFitter.chained_lsqfit(...) "
        # sequential fit
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit1 = fitter.lsqfit(data=self.data, prior=self.prior, chained=True)
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])

    def test_chained_lsqfit_p0(self):
        " MultiFitter.chained_lsqfit(...) "
        # sequential fit
        fitter = MultiFitter(models=self.make_models(ncg=1))
        p0 = gv.BufferDict({'a': 0.9991638707908023,'b': 0.4995927960301173})
        p0list = [p0, gv.BufferDict(a=0.9991638707908023), gv.BufferDict(a=0.9991638707908023)]
        fit1 = fitter.chained_lsqfit(data=self.data, prior=self.prior, p0=p0)
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])
        self.assertEqual(fit1.p0, p0list)
        fit1 = fitter.chained_lsqfit(data=self.data, prior=self.prior, p0=3 * [p0])
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])
        self.assertEqual(fit1.p0, p0list)
        fn = 'test_multifitter.p'
        fit1 = fitter.chained_lsqfit(data=self.data, prior=self.prior, p0=fn)
        fit2 = fitter.chained_lsqfit(data=self.data, prior=self.prior, p0=fn)
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])
        self.assertEqual([f.pmean for f in fit1.chained_fits.values()], fit2.p0)
        os.unlink(fn)
         
    def test_dump_chained_lsqfit(self):
        " MultiFitter.chained_lsqfit(...) "
        # sequential fit
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit1 = gv.loads(gv.dumps(fitter.chained_lsqfit(data=self.data, prior=self.prior)))
        self.assertTrue(self.agree_ref(fit1.p))
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])

    def test_chained_fit_simul(self):
        " MultiFitter(models=[m1, (m2,m3)], ...) "
        models = self.make_models(ncg=1)
        models = [models[0], tuple(models[1:])]
        fitter = MultiFitter(models=models)
        fit3 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit3.p))
        self.assertEqual(list(fit3.chained_fits.keys()), ['l', '(c1,c2)'])

        # with coarse grain, marginalization and extend
        models = self.make_models(ncg=2)
        models = [models[0], tuple(models[1:])]
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit4 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True
            )
        self.assertTrue(self.agree_ref(fit4.p))

    def test_chained_fit_kargs(self):
        " MultiFitter(models=[m1, dict(...), m2, ...]) "
        models = self.make_models(ncg=1)
        models = [models[2], dict(mopt=True), models[1], models[0]]
        fitter = MultiFitter(models=models, mopt=None)
        fit = fitter.chained_lsqfit(data=self.data, prior=self.prior, fast=True, wavg_all=False)
        self.assertTrue(fit.p['b'] is self.prior['b'])
        self.assertEqual(fitter.mopt, None)
        fitter = MultiFitter(models=fitter.flatten_models(models), mopt=None)
        fit = fitter.chained_lsqfit(data=self.data, prior=self.prior, fast=True)
        self.assertTrue(fit.p['b'] is not self.prior['b'])

    def test_chained_fit_parallel(self):
        " MultiFitter(models=[m1, [m2,m3]], ...) "
        models = self.make_models(ncg=1)
        models = [models[0], models[1:]]
        fitter = MultiFitter(models=models)
        fit5 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit5.p))
        self.assertEqual(list(fit5.chained_fits.keys()), ['l', 'c1', 'c2', 'wavg(c1,c2)'])

        # degenerate parallel fit (nfit=1)
        models = self.make_models(ncg=1)
        models = [[models[0]], models[1:]]
        fitter = MultiFitter(models=models)
        fit5 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit5.p))
        self.assertEqual(list(fit5.chained_fits.keys()), ['l', 'c1', 'c2', 'wavg(c1,c2)'])

        # dictionaries in parallel fits
        models = self.make_models(ncg=1)
        models = [[dict(svdcut=1e-12), models[0]], [dict(svdcut=1e-12)] + models[1:]]
        fitter = MultiFitter(models=models)
        fit5 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertTrue(self.agree_ref(fit5.p))
        self.assertEqual(list(fit5.chained_fits.keys()), ['l', 'c1', 'c2', 'wavg(c1,c2)'])

        # with coarse grain, marginalization
        models = self.make_models(ncg=2)
        models = [models[0], models[1:]]
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit6 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True
            )
        self.assertTrue(self.agree_ref(fit6.p))
        # self.assertEqual(str(fit6.p), "{'log(a)': -0.081(64),'b': 0.50(10)}") # "{'log(a)': -0.00083(62),'b': 0.5000(10)}") #"{'log(a)': -0.00073(48),'b': 0.5000(10)}")

    def test_bootstrap_lsqfit(self):
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit = fitter.lsqfit(data=self.data, prior=self.prior)
        datalist = gv.bootstrap_iter(self.data, n=10)
        ds = gv.dataset.Dataset()
        for bf in fit.bootstrapped_fit_iter(datalist=datalist):
            ds.append(bf.pmean)
        p = gv.dataset.avg_data(ds, bstrap=True)
        self.assertTrue(abs(p['a'].mean - 1.) < 5 * p['a'].sdev)
        self.assertTrue(abs(p['b'].mean - 0.5) < 5 * p['b'].sdev)
        self.assertEqual(ds.samplesize, 10)

        pdatalist = (
            fitter.process_data(d, fitter.models)
            for d in gv.bootstrap_iter(self.data, n=10)
            )
        ds = gv.dataset.Dataset()
        for bf in fit.bootstrapped_fit_iter(pdatalist=pdatalist):
            ds.append(bf.pmean)
        p = gv.dataset.avg_data(ds, bstrap=True)
        self.assertTrue(abs(p['a'].mean - 1.) < 5 * p['a'].sdev)
        self.assertTrue(abs(p['b'].mean - 0.5) < 5 * p['b'].sdev)
        self.assertEqual(ds.samplesize, 10)

        ds = gv.dataset.Dataset()
        for bf in fit.bootstrapped_fit_iter(n=10):
            ds.append(bf.pmean)
        p = gv.dataset.avg_data(ds, bstrap=True)
        self.assertTrue(abs(p['a'].mean - 1.) < 5 * p['a'].sdev)
        self.assertTrue(abs(p['b'].mean - 0.5) < 5 * p['b'].sdev)
        self.assertEqual(ds.samplesize, 10)

    def test_bootstrap_chained_lsqfit(self):
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        datalist = gv.bootstrap_iter(self.data, n=10)
        ds = gv.dataset.Dataset()
        for bf in fit.bootstrapped_fit_iter(datalist=datalist):
            ds.append(bf.pmean)
        p = gv.dataset.avg_data(ds, bstrap=True)
        self.assertTrue(abs(p['a'].mean - 1.) < 5 * p['a'].sdev)
        self.assertTrue(abs(p['b'].mean - 0.5) < 5 * p['b'].sdev)
        # self.assertEqual(str(p), "{'a': 0.99905(40),'b': 0.49995(85)}")

    def test_process_data(self):
        data = gv.BufferDict()
        data['l'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        data['c1'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        data['c2'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        data['dummy'] = ['10(10)']
        data = gv.gvar(data)

        models = self.make_models(ncg=1)
        pdata = MultiFitter.process_data(data, models)
        self.assertTrue('dummy' not in pdata)
        for tag in pdata:
            self.assertEqual(str(pdata[tag]), str(data[tag]))

        models = self.make_models(ncg=2)
        pdata = MultiFitter.process_data(data, models)
        self.assertTrue('dummy' not in pdata)
        for tag in pdata:
            cgdata = np.sum([data[tag][:2], data[tag][2:]], axis=1) / 2.
            self.assertEqual(str(pdata[tag]), str(cgdata))

    def test_process_dataset(self):
        dataset = gv.BufferDict()
        dataset['l'] = [[1., 2., 3., 4.], [2., 3., 4. , 5.]]
        dataset['c1'] = [[1., 2., 3., 4.], [2., 3., 4. , 5.]]
        dataset['c2'] = [[1., 2., 3., 4.], [2., 3., 4. , 5.]]
        dataset['dummy'] = [[1.], [2.]]
        data = gv.dataset.avg_data(dataset)

        models = self.make_models(ncg=1)
        pdata = MultiFitter.process_dataset(dataset, models)
        self.assertTrue('dummy' not in pdata)
        for tag in pdata:
            self.assertEqual(str(pdata[tag]), str(data[tag]))

        models = self.make_models(ncg=2)
        pdata = MultiFitter.process_dataset(dataset, models)
        self.assertTrue('dummy' not in pdata)
        for tag in pdata:
            cgdata = np.sum([data[tag][:2], data[tag][2:]], axis=1) / 2.
            self.assertEqual(str(pdata[tag]), str(cgdata))

    def test_get_prior_keys(self):
        prior = gv.BufferDict({'log(a)':1., 'b':2.})
        self.assertEqual(
            gv.get_dictkeys(prior, ['a', 'b', 'log(a)']),
            ['log(a)', 'b', 'log(a)']
            )
        self.assertEqual(
            [ gv.dictkey(prior, k) for k in [
                'a', 'b', 'log(a)'
                ]],
            ['log(a)', 'b', 'log(a)']
            )

    def test_empbayes_fit(self):
        fitter = MultiFitter(models=self.make_models(ncg=1))
        def fitargs(z):
            prior = dict(a=gv.gvar(1.0, z), b=gv.gvar(0.5, z))
            return dict(prior=prior, data=self.data, chained=False)
        fit,z = fitter.empbayes_fit(0.1, fitargs, step=0.01)
        self.assertNotAlmostEqual(z, 0.1)
        self.assertGreater(fit.logGBF, self.ref_fit.logGBF)

class Linear(MultiFitterModel):
    def __init__(self, datatag, a, b, x, ncg):
        super(Linear, self).__init__(datatag=datatag, ncg=ncg)
        self.a = a
        self.b = b
        self.x = x

    def fitfcn(self, p):
        try:
            return p[self.a] + p[self.b] * self.x
        except KeyError:
            # slope marginalized
            return len(self.x) * [p[self.a]]

    def buildprior(self, prior, mopt=None, extend=False):
        nprior = gv.BufferDict()
        if mopt is None:
            for k in [self.a, self.b]:
                k = gv.dictkey(prior, k)
                nprior[k] = prior[k]
        else:
            k = gv.dictkey(prior, self.a)
            nprior[k] = prior[k]
        self.mopt = mopt
        # use self.mopt to marginalize fitfcn
        return nprior

    def builddata(self, data):
        return data[self.datatag]

    def builddataset(self, dataset):
        return dataset[self.datatag]

class Constant(MultiFitterModel):
    def __init__(self, datatag, a, ndata, ncg):
        super(Constant, self).__init__(datatag=datatag, ncg=ncg)
        self.a = a
        self.ndata = ndata
        self.ncg = ncg

    def fitfcn(self, p):
        return self.ndata * [p[self.a]]

    def buildprior(self, prior, mopt=None, extend=False):
        nprior = gv.BufferDict()
        k = gv.dictkey(prior, self.a)
        nprior[k] = prior[k]
        return nprior

    def builddata(self, data):
        return data[self.datatag]

    def builddataset(self, dataset):
        return dataset[self.datatag]

if __name__ == '__main__':
    unittest.main()

