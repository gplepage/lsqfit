from __future__ import print_function   # makes this work for python2 and 3

import unittest
import inspect
import numpy as np
import gvar as gv
import lsqfit
from lsqfit import MultiFitter, MultiFitterModel

class test_multifitter(unittest.TestCase):
    def setUp(self):
        gv.ranseed(1)
        a = gv.gvar('1.000(1)')
        b = gv.gvar('0.500(1)')
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
        data = fitter.builddata(self.data)
        del self.data['dummy']
        self.assertEqual(str(data), str(self.data))

        # ncg=2
        simpledata = gv.BufferDict()
        simpledata['l'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata['c1'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata['c2'] = ['1.0(3)', '2.0(4)', '3.0(3)', '4.0(4)']
        simpledata = gv.gvar(simpledata)
        fitter = MultiFitter(models=self.make_models(ncg=2))
        data = fitter.builddata(simpledata)
        for k in simpledata:
            self.assertEqual(str(data[k]), '[1.50(25) 3.50(25)]')

    def test_builddata_pdata(self):
        " MultiFitter.builddata(pdata=..., ...) "
        # ncg=1
        fitter = MultiFitter(models=self.make_models(ncg=1))
        data = fitter.builddata(self.data)
        data['extra'] = np.array([1.])
        ndata = fitter.builddata(pdata=data)
        del data['extra']
        self.assertEqual(str(data), str(ndata))

        # ncg=2
        fitter = MultiFitter(models=self.make_models(ncg=2))
        data = fitter.builddata(self.data)
        data['extra'] = np.array([1.])
        ndata = fitter.builddata(pdata=data)
        del data['extra']
        self.assertEqual(str(data), str(ndata))

    def test_builddata_marginalized(self):
        " MultiFitter.builddata with marginalization "
        fitter = MultiFitter(models=self.make_models(ncg=1), mopt=True)
        data = fitter.builddata(self.data, prior=self.prior)
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

    def test_buildprior_marginalized(self):
        " MultiFitter.buildprior with marginalization"
        prior = gv.BufferDict(self.prior)
        prior['dummy'] = gv.gvar('12(12)')
        fitter = MultiFitter(models=self.make_models(ncg=1), mopt=True)
        prior = fitter.buildprior(prior=prior)
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

    def test_lsqfit_coarse_grain(self):
        " MultiFitter.lsqfit(..., ncg=2) "
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit3 = fitter.lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(str(fit3.p), "{'a': 0.99937(46),'b': 0.49994(78)}")

    def test_lsqfit_pdata_coarse_grain(self):
        " MultiFitter.lsqfit(pdata=..., ..., ncg=2) "
        fitter = MultiFitter(models=self.make_models(ncg=2))
        pdata = MultiFitter.process_data(data=self.data, models=fitter.models)
        fit3 = fitter.lsqfit(pdata=pdata, prior=self.prior)
        self.assertEqual(str(fit3.p), "{'a': 0.99937(46),'b': 0.49994(78)}")

    def test_marginalization(self):
        " MultiFitter.lsqfit(..., mopt=...) "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit4 = fitter.lsqfit(data=self.data, prior=self.prior, mopt=True)
        self.assertEqual(str(fit4.p['a']), str(self.ref_fit.p['a']))
        self.assertEqual(gv.fmt_chi2(fit4), gv.fmt_chi2(self.ref_fit))
        self.assertTrue('b' not in fit4.p)

    def test_extend(self):
        " MultiFitter.lsqfit(..., extend=True) "
        fitter = MultiFitter(models=self.make_models(ncg=1))
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fit5 = fitter.lsqfit(data=self.data, prior=prior, extend=True)
        self.assertEqual(str(fit5.p['a']), str(self.ref_fit.p['a']))
        self.assertEqual(gv.fmt_chi2(fit5), gv.fmt_chi2(self.ref_fit))
        self.assertTrue('log(a)' in fit5.p)

    def test_fast(self):
        " MultiFitter.lsqfit(..., fast=False) "
        # with fast=False
        self.prior['aa'] = self.prior['a'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit6 = fitter.lsqfit(data=self.data, prior=self.prior, fast=False)
        self.assertEqual(
            str(fit6.p),
            "{'a': 0.99938(46),'b': 0.49985(78),'aa': 0.99938(46)}",
            )

        # with fast=True (default)
        self.prior['aa'] = self.prior['a'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit7 = fitter.lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(
            str(fit7.p), "{'a': 0.99938(46),'b': 0.49985(78)}",
            )

    def test_chained_lsqfit(self):
        " MultiFitter.chained_lsqfit(models=[m1, m2, m3], ...) "
        # sequential fit
        fitter = MultiFitter(models=self.make_models(ncg=1))
        fit1 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(str(fit1.p), "{'a': 0.99929(48),'b': 0.50004(81)}")
        self.assertEqual(list(fit1.chained_fits.keys()), ['l', 'c1', 'c2'])

        # with coarse grain, marginalization and extend, and with fast=False
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b']),
            ])
        prior['log(aa)'] = prior['log(a)'] + gv.gvar('0(1)') * 1e-6
        fitter = MultiFitter(models=self.make_models(ncg=2), fast=False)
        fit2 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True, extend=True
            )
        self.assertEqual(
            str(fit2.p),
            "{'log(a)': -0.00073(48),'b': 0.50015(82),"
            "'log(aa)': -0.00073(48),'a': 0.99927(48),"
            "'aa': 0.99927(48)}",
            )

    def test_chained_fit_seq_simul(self):
        " MultiFitter.chained_lsqfit(models=[m1, (m2,m3)], ...) "
        models = self.make_models(ncg=1)
        models = [models[0], tuple(models[1:])]
        fitter = MultiFitter(models=models)
        fit3 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(str(fit3.p), "{'a': 0.99931(48),'b': 0.50000(81)}")
        self.assertEqual(list(fit3.chained_fits.keys()), ['l', '(c1,c2)'])

        # with coarse grain, marginalization and extend
        models = self.make_models(ncg=2)
        models = [models[0], tuple(models[1:])]
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit4 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True, extend=True
            )
        self.assertEqual(str(fit4.p), "{'log(a)': -0.00073(48),'a': 0.99927(48)}")

    def test_chained_fit_seq_parallel(self):
        " MultiFitter.chained_lsqfit(models=[m1, [m2,m3]], ...) "
        models = self.make_models(ncg=1)
        models = [models[0], models[1:]]
        fitter = MultiFitter(models=models)
        fit5 = fitter.chained_lsqfit(data=self.data, prior=self.prior)
        self.assertEqual(str(fit5.p), "{'a': 0.99932(48),'b': 0.49998(81)}")
        self.assertEqual(list(fit5.chained_fits.keys()), ['l', '[c1,c2]'])
        self.assertEqual(
            list(fit5.chained_fits['[c1,c2]'].sub_fits.keys()), ['c1', 'c2']
            )

        # with coarse grain, marginalization and extend
        models = self.make_models(ncg=2)
        models = [models[0], models[1:]]
        prior = gv.BufferDict([
            ('log(a)', gv.log(self.prior['a'])), ('b', self.prior['b'])
            ])
        fitter = MultiFitter(models=self.make_models(ncg=2))
        fit6 = fitter.chained_lsqfit(
            data=self.data, prior=prior, mopt=True, extend=True
            )
        self.assertEqual(str(fit6.p), "{'log(a)': -0.00073(48),'a': 0.99927(48)}")

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

class Linear(MultiFitterModel):
    def __init__(self, datatag, a, b, x, ncg):
        super(Linear, self).__init__(datatag=datatag, ncg=ncg)
        self.a = a
        self.b = b
        self.x = x

    def fitfcn(self, p):
        if self.b in p:
            return p[self.a] + p[self.b] * self.x
        else:
            return len(self.x) * [p[self.a]]

    def buildprior(self, prior, mopt=None, extend=False):
        nprior = gv.BufferDict()
        if mopt is None:
            for k in self.get_prior_keys(prior, [self.a, self.b], extend=extend):
                nprior[k] = prior[k]
        else:
            for k in self.get_prior_keys(prior, [self.a], extend=extend):
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
        for k in self.get_prior_keys(prior, [self.a], extend=extend):
            nprior[k] = prior[k]
        return nprior

    def builddata(self, data):
        return data[self.datatag]

    def builddataset(self, dataset):
        return dataset[self.datatag]

if __name__ == '__main__':
    unittest.main()

