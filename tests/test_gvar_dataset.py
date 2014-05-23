#!/usr/bin/env python
# encoding: utf-8
"""
test-dataset.py

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

import pickle
import os
import unittest
import warnings
import numpy as np
import random
import gvar as gv
from gvar import *
from gvar.dataset import *

FAST = False

class ArrayTests(object):
    def __init__(self):
        pass

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



class test_dataset(unittest.TestCase,ArrayTests):
    def setUp(self): pass

    def tearDown(self): pass

    def test_bin_data(self):
        """ bin_data """
        self.assertEqual(bin_data([1,2,3,4]),[1.5,3.5])
        self.assertEqual(bin_data(np.array([1,2,3,4])),[1.5,3.5])
        self.assert_arraysequal(bin_data([[1,2],[3,4]]),[[2.,3.]])
        self.assert_arraysequal(bin_data([[[1,2]],[[3,4]]]),[[[2.,3.]]])
        self.assertEqual(bin_data([1]),[])
        self.assertEqual(bin_data([1,2,3,4,5,6,7],binsize=3),[2.,5.])
        data = dict(s=[1,2,3,4],
                    v=[[1,2],[3,4],[5,6,],[7,8],[9,10]])
        bd = bin_data(data)
        self.assertEqual(bd['s'],[1.5,3.5])
        self.assert_arraysequal(bd['v'],[[2,3],[6,7]])
        data = dict(s=[1,2,3,4],
                    v=[[1,2],[3,4],[5,6,],[7,8],[9,10]])
        bd = bin_data(data,binsize=3)
        self.assertEqual(bd['s'],[2])
        self.assert_arraysequal(bd['v'],[[3,4]])
        with self.assertRaises(ValueError):
            bd = bin_data([[1,2],[[3,4]]])
        self.assertEqual(bin_data([]),[])
        self.assertEqual(bin_data(dict()),Dataset())

    def test_avg_data(self):
        """ avg_data """
        self.assertTrue(avg_data([]) is None)
        self.assertEqual(avg_data(dict()),BufferDict())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            avg_data(dict(s=[1.],v=[1.,2.]))
            self.assertEqual(len(w), 1)
        with self.assertRaises(ValueError):
            avg_data(dict(s=[],v=[1.,2.]), warn=False)
        with self.assertRaises(ValueError):
            avg_data(dict(s=[], v=[]))
        with self.assertRaises(ValueError):
            avg_data([1,2,"s"])
        mean = avg_data([1])
        self.assertEqual(mean.mean,1.)
        self.assertEqual(mean.sdev,0.)
        #
        mean = avg_data([1,2])
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,sum((vi-1.5)**2 
                               for vi in [1,2])/4.)
        mean2 = avg_data(np.array([1.,2.]))
        self.assertEqual(mean.mean,mean2.mean)
        self.assertEqual(mean.sdev,mean2.sdev)
        #
        mean = avg_data([1,2],spread=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,sum((vi-1.5)**2 
                               for vi in [1,2])/2.)
        #
        mean = avg_data([1,2],median=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,0.5**2/2.)
        #
        mean = avg_data([1,2],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,1.5)
        self.assertAlmostEqual(mean.var,0.5**2)
        #
        mean = avg_data([1,2,3])
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,sum((vi-2.)**2 
                               for vi in [1,2,3])/9.)
        #
        mean = avg_data([1,2,3], noerror=True)
        self.assertAlmostEqual(mean, 2.0)
        #
        mean = avg_data([[1],[2],[3]])
        self.assertAlmostEqual(mean[0].mean,2.0)
        self.assertAlmostEqual(mean[0].var,sum((vi-2.)**2 
                               for vi in [1,2,3])/9.)
        
        mean = avg_data([[1],[2],[3]], noerror=True)
        self.assertAlmostEqual(mean[0], 2.0)

        mean = avg_data([1,2,3],spread=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,sum((vi-2.)**2 
                               for vi in [1,2,3])/3.)
        #
        mean = avg_data([1,2,3],median=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,1./3.)
        #
        mean = avg_data([[1],[2],[3]],median=True)
        self.assertAlmostEqual(mean[0].mean,2.0)
        self.assertAlmostEqual(mean[0].var,1./3.)
        #
        mean = avg_data([1,2,3],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,2.0)
        self.assertAlmostEqual(mean.var,1.)
        #            
        mean = avg_data([1,2,3,4,5,6,7,8,9],median=True)
        self.assertAlmostEqual(mean.mean,5)
        self.assertAlmostEqual(mean.var,3.**2/9.)
        #            
        mean = avg_data([1,2,3,4,5,6,7,8,9],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,5.)
        self.assertAlmostEqual(mean.var,3.**2)
        #            
        mean = avg_data([1,2,3,4,5,6,7,8,9,10],median=True)
        self.assertAlmostEqual(mean.mean,5.5)
        self.assertAlmostEqual(mean.var,3.5**2/10.)
        #            
        mean = avg_data([1,2,3,4,5,6,7,8,9,10],median=True,spread=True)
        self.assertAlmostEqual(mean.mean,5.5)
        self.assertAlmostEqual(mean.var,3.5**2)
        # 
        data = dict(s=[1,2,3],v=[[1,1],[2,2],[3,3]])
        mean = avg_data(data,median=True,spread=True)
        self.assertAlmostEqual(mean['s'].mean,2.0)
        self.assertAlmostEqual(mean['s'].var,1.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_gvclose(mean['v'],[gvar(2,1),gvar(2,1)])

        mean = avg_data(data, median=True, noerror=True)
        self.assertAlmostEqual(mean['s'],2.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_arraysclose(mean['v'], [2,2])

        mean = avg_data(data, noerror=True)
        self.assertAlmostEqual(mean['s'],2.0)
        self.assertEqual(mean['v'].shape,(2,))
        self.assert_arraysclose(mean['v'], [2,2])

    def test_autocorr(self):
        """ dataset.autocorr """
        N = 10000
        eps = 10./float(N)**0.5
        x = gvar(2,0.1)
        a = np.array([x() for i in range(N)])
        a = (a[:-2]+a[1:-1]+a[2:])/3.
        ac_ex = np.zeros(a.shape,float)
        ac_ex[:3] = np.array([1.,0.66667,0.33333])
        ac_a = autocorr(a)
        self.assertLess(numpy.std(ac_a-ac_ex)*2,eps)
        b = np.array([[x(),x()] for i in range(N)])
        b = (b[:-2]+b[1:-1]+b[2:])/3.
        ac_ex = np.array(list(zip(ac_ex,ac_ex)))
        ac_b = autocorr(b)
        self.assertLess(numpy.std(ac_b-ac_ex),eps)
        c = dict(a=a,b=b)
        ac_c = autocorr(c)
        self.assert_arraysequal(ac_c['a'],ac_a)
        self.assert_arraysequal(ac_c['b'],ac_b)

    def test_dataset_append(self):
        """ Dataset.append() """
        data = Dataset()
        data.append(s=1,v=[10,100])
        self.assert_arraysequal(data['s'],[1.])
        self.assert_arraysequal(data['v'],[[10.,100.]])
        data.append(s=2,v=[20,200])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.append(dict(s=3,v=[30,300]))
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.append('s',4.)
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.append('v',[40.,400.])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal( #
            data['v'],[[10.,100.],[20.,200.],[30.,300.],[40.,400.]])
        with self.assertRaises(ValueError):
            data.append('v',5.)
        with self.assertRaises(ValueError):
            data.append('s',[5.])
        with self.assertRaises(ValueError):
            data.append('s',"s")
        with self.assertRaises(ValueError):
            data.append('v',[[5.,6.]])
        with self.assertRaises(ValueError):
            data.append('v',[.1],'s')
        with self.assertRaises(ValueError):
            data.append([1.])
        #
        data = Dataset()
        data.append('s',1)
        self.assertEqual(data['s'],[1.])
        data = Dataset()
        data.append(dict(s=1,v=[10,100]))
        self.assertEqual(data['s'],[1.])
        self.assert_arraysequal(data['v'],[[10.,100.]])

    def test_dataset_extend(self):
        """ Dataset.extend """
        data = Dataset()
        data.extend(s=[1,2],v=[[10.,100.],[20.,200.]])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(s=[])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(s=[3],v=[[30,300]])
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.extend('s',[4.])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data.extend('v',[[40,400.]])
        self.assert_arraysequal(data['s'],[1.,2.,3.,4.])
        self.assert_arraysequal( #
            data['v'],[[10.,100.],[20.,200.],[30.,300.],[40.,400.]])
        with self.assertRaises(TypeError):
            data.extend('s',5.)
        with self.assertRaises(ValueError):
            data.extend('v',[5.,6.])
        with self.assertRaises(ValueError):
            data.extend('s',"s")
        with self.assertRaises(ValueError):
            data.extend('v',[[[5.,6.]]])
        #
        with self.assertRaises(ValueError):
            data.extend('v',[[5,6],[[5.,6.]]])
        with self.assertRaises(ValueError):
            data.extend('v',[.1],'s')
        with self.assertRaises(ValueError):
            data.extend([1.])
        #
        data = Dataset()
        data.extend(dict(s=[1,2],v=[[10.,100.],[20.,200.]]))
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])
        data.extend(dict(s=[3],v=[[30,300]]))
        self.assert_arraysequal(data['s'],[1.,2.,3.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.],[30.,300.]])
        data = Dataset()
        data.extend('s',[1,2])
        data.extend('v',[[10.,100.],[20.,200.]])
        self.assert_arraysequal(data['s'],[1.,2.])
        self.assert_arraysequal(data['v'],[[10.,100.],[20.,200.]])

    def test_dataset_init(self):
        """ Dataset() """
        fin = ['test-gvar.input1','test-gvar.input2']
        with open(fin[0],"w") as f:
            f.write("""
                # first
                s 1
                v 10 100
                #second
                s 2
                v 20 200
                s 3
                v 30 300
                """)  # """
        with open(fin[1],"w") as f:
            f.write("""
                a [[1,10]]
                a [[2,20]]
                a [[3,30]]
                """)  # """
        data = Dataset(fin[0])
        self.assertEqual(data['s'],[1,2,3])
        self.assert_arraysequal(data['v'],[[10,100],[20,200],[30,300]])
        data = Dataset(fin)
        self.assertEqual(data['s'],[1,2,3])
        self.assert_arraysequal(data['v'],[[10,100],[20,200],[30,300]])
        self.assert_arraysequal(data['a'],[[[1,10]],[[2,20]],[[3,30]]])
        data = Dataset(fin[0],binsize=2)
        self.assertEqual(data['s'],[1.5])
        self.assert_arraysequal(data['v'],[[15,150]])
        data = Dataset(fin,keys=['s'])
        self.assertTrue('v' not in data)
        self.assertTrue('a' not in data)
        self.assertTrue('s' in data)
        self.assertEqual(data['s'],[1,2,3])
        with self.assertRaises(TypeError): 
            data = Dataset("xxx.input1","xxx.input2")
        os.remove(fin[0])
        os.remove(fin[1])
    
    def test_dataset_init2(self):
        """ init from dictionaries or datasets """
        def assert_dset_equal(d1, d2):
            for k in d1:
                assert k in d2, 'key mismatch'
            for k in d2:
                assert k in d1, 'key mismatch'
                self.assertTrue(np.all(np.array(d1[k]) == np.array(d2[k])))
        data = Dataset(dict(a=[[1.,3.], [3.,4.]], b=[1., 2.])) 
        data_reduced = Dataset(dict(a=[[1.,3.], [3.,4.]]))
        data_binned = Dataset(dict(a=[[2.,3.5]], b=[1.5])) 
        data_empty = Dataset()
        self.assertEqual(data['a'], [[1.,3.], [3.,4.]])
        self.assertEqual(data['b'], [1., 2.])
        assert_dset_equal(data, Dataset(data))
        assert_dset_equal(data_reduced, Dataset(data,keys=['a']))
        assert_dset_equal(data, 
            Dataset([('a', [[1.,3.], [3.,4.]]), ('b', [1., 2.])])
            )
        assert_dset_equal(data, 
            Dataset([['a', [[1.,3.], [3.,4.]]], ['b', [1., 2.]]])
            )
        assert_dset_equal(data_reduced, Dataset(data, keys=['a']))
        assert_dset_equal(data_reduced, Dataset(data, grep='[^b]'))
        assert_dset_equal(data_empty, Dataset(data, grep='[^b]', keys=['b']))
        assert_dset_equal(data_binned, Dataset(data, binsize=2))
        assert_dset_equal(
            Dataset(data_binned, keys=['a']), 
            Dataset(data, binsize=2, keys=['a'])
            )
        assert_dset_equal(
            Dataset(data_binned, keys=['a']), 
            Dataset(data, binsize=2, grep='[^b]')
            )
        assert_dset_equal(
            Dataset(data_binned, keys=['a']), 
            Dataset(data, binsize=2, grep='[^b]', keys=['a'])
            )
        s = pickle.dumps(data)
        assert_dset_equal(data, pickle.loads(s))

    def test_dataset_toarray(self):
        """ Dataset.toarray """
        data = Dataset()
        data.extend(s=[1,2],v=[[1,2],[2,3]])
        data = data.toarray()
        self.assert_arraysequal(data['s'],[1,2])
        self.assert_arraysequal(data['v'],[[1,2],[2,3]])
        self.assertEqual(data['s'].shape,(2,))
        self.assertEqual(data['v'].shape,(2,2))

    def test_dataset_slice(self):
        """ Dataset.slice """
        data = Dataset()
        data.extend(a=[1,2,3,4],b=[[1],[2],[3],[4]])
        ndata = data.slice(slice(0,None,2))
        self.assert_arraysequal(ndata['a'],[1,3])
        self.assert_arraysequal(ndata['b'],[[1],[3]])

    def test_dataset_grep(self):
        """ Dataset.grep """
        data = Dataset()
        data.extend(aa=[1,2,3,4],ab=[[1],[2],[3],[4]])
        ndata = data.grep("a")
        self.assertTrue('aa' in ndata and 'ab' in ndata)
        self.assert_arraysequal(ndata['ab'],data['ab'])
        self.assert_arraysequal(ndata['aa'],data['aa'])
        ndata = data.grep("b")
        self.assertTrue('aa' not in ndata and 'ab' in ndata)
        self.assert_arraysequal(ndata['ab'],data['ab'])

    def test_dataset_samplesize(self):
        """ Dataset.samplesize """
        data = Dataset()
        data.extend(aa=[1,2,3,4],ab=[[1],[2],[3]])
        self.assertEqual(data.samplesize,3)

    def test_dataset_trim(self):
        """ Dataset.trim """
        data = Dataset()
        data.append(a=1,b=10)
        data.append(a=2,b=20)
        data.append(a=3)
        ndata = data.trim()
        self.assertEqual(ndata.samplesize,2)
        self.assert_arraysequal(ndata['a'],[1,2])
        self.assert_arraysequal(ndata['b'],[10,20])

    def test_dataset_arrayzip(self):
        """ Dataset.arrayzip """
        data = Dataset()
        data.extend(a=[1,2,3], b=[10,20,30])
        a = data.arrayzip([['a'], ['b']])
        self.assert_arraysequal(a, [[[1],[10]],[[2],[20]],[[3],[30]]])
        with self.assertRaises(ValueError):
            data.append(a=4)
            a = data.arrayzip(['a','b'])

    def test_dataset_bootstrap_iter(self):
        """ bootstrap_iter(data_dict) """
        # make data
        N = 100
        a0 = dict(n=gvar(1,1),a=[gvar(2,2),gvar(100,100)])
        dset = Dataset()
        for ai in raniter(a0,30):
            dset.append(ai)
        a = avg_data(dset)

        # do bootstrap -- calculate means
        bs_mean = Dataset()
        for ai in bootstrap_iter(dset,N):
            for k in ai:
                bs_mean.append(k,np.average(ai[k],axis=0))
                for x in ai[k]:
                    self.assertTrue(   #
                        x in numpy.asarray(dset[k]), 
                        "Bootstrap element not in original dataset.")
        a_bs = avg_data(bs_mean,bstrap=True)

        # 6 sigma tests
        an_mean = a['n'].mean
        an_sdev = a['n'].sdev
        self.assertGreater(6*an_sdev/N**0.5,abs(an_mean-a_bs['n'].mean))
        self.assertGreater(6*an_sdev/N**0.5,abs(an_sdev-a_bs['n'].sdev))


    def test_array_bootstrap_iter(self):
        """ bootstrap_iter(data_array) """
        N = 100
        a0 = [[1,2],[3,4],[5,6]]
        for ai in bootstrap_iter(a0,N):
            self.assertTrue(len(ai)==len(a0),"Bootstrap copy wrong length.")
            for x in ai:
                self.assertTrue(    #
                    x in numpy.asarray(a0), 
                    "Bootstrap element not in original dataset.")        



if __name__ == '__main__':
	unittest.main()

