#!/usr/bin/env python
# encoding: utf-8
"""
test-bufferdict.py

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

import unittest
import pickle as pckl
import numpy as np
import gvar as gv
from gvar import BufferDict

class ArrayTests(object):
    def __init__(self):
        pass
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
        self.assertSequenceEqual(np.shape(x),np.shape(y))
        x = [float(xi) for xi in np.array(x).flatten()]
        y = [float(yi) for yi in np.array(y).flatten()]
        self.assertSequenceEqual(x,y)
    ##
##

class test_bufferdict(unittest.TestCase,ArrayTests):
    def setUp(self):
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        b = BufferDict()
        bkeys = ['scalar','vector','tensor']
        bvalues = [0.,np.array([1.,2.]),np.array([[3.,4.],[5.,6.]])]
        bslices = [0,slice(1, 3, None),slice(3, 7, None)]
        bbuf = np.arange(7.)
        bkeybuf = ['scalar','vector','','tensor','','','']
        b.add('scalar',0.)
        b['scalar']  # test flipping bt list and array
        b.add('vector',np.array([1.,2.]))
        b.add('tensor',[[3.,4.],[5.,6.]])
    ##
    def tearDown(self):
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        b = None
    ##
    def test_b_flat(self):
        """ b.flat """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        self.assert_arraysequal(b.flat,bbuf)
        self.assertEqual(b.size,len(bbuf))
        b.flat = 10.+bbuf
        self.assert_arraysequal(b.flat,10.+bbuf)
        for k,v in zip(bkeys,bvalues):
            self.assert_arraysequal(b[k],10.+v)
        b.flat = bbuf
        for k in b:
            b[k] = 10.
        self.assert_arraysequal(bbuf,np.zeros(np.shape(bbuf))+10.)
    ##
    def test_b_keys(self):
        """ b.keys """
        global b,bkeys
        self.assertSequenceEqual(list(b.keys()),bkeys)
    ##
    def test_b_slice(self):
        """ b.slice(k) """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        for k,sl in zip(bkeys,bslices):
            self.assertEqual(sl,b.slice(k))
    ##
    def test_b_getitem(self):
        """ v = b[k] """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        for k,v in zip(bkeys,bvalues):
            self.assert_arraysequal(b[k],v)
    ##
    def test_b_setitem(self):
        """ b[k] = v """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        for k,v in zip(bkeys,bvalues):
            b[k] = v + 10.
            self.assert_arraysequal(b[k],v+10.)
            self.assert_arraysequal(b.flat[b.slice(k)],((v+10.).flatten() 
                                                if k=='tensor' else v+10.))
        b['pseudoscalar'] = 11.
        self.assertEqual(b['pseudoscalar'],11)
        self.assertEqual(b.flat[-1],11)
        self.assertSequenceEqual(list(b.keys()),bkeys+['pseudoscalar'])
    ##
    def test_str(self):
        """ str(b) repr(b) """
        outstr = "{'scalar':0.0,'vector':array([1.,2.]),"
        outstr += "'tensor':array([[3.,4.],[5.,6.]])}"
        self.assertEqual(''.join(str(b).split()),outstr)
        outstr = "BufferDict([('scalar',0.0),('vector',array([1.,2.])),"
        outstr += "('tensor',array([[3.,4.],[5.,6.]]))])"
        self.assertEqual(''.join(repr(b).split()),outstr)
    ##
    def test_bufferdict_b(self):
        """ BufferDict(b) """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        nb = BufferDict(b)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
            nb[k] += 10.
            self.assert_arraysequal(nb[k] , b[k]+10.)
        nb = BufferDict(nb,buf=b.flat)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
        self.assertEqual(b.size,nb.size)
        b.flat[-1] = 130.
        self.assertEqual(nb.flat[-1],130.)
        with self.assertRaises(ValueError):
            nb = BufferDict(b,buf=nb.flat[:-1])
    ##
    def test_b_adddict(self):
        """ b.add(dict(..)) """
        global b,bkeys,bvalues,bslices,bbuf,bkeybuf
        nb = BufferDict()
        nb.add(b)
        for k in bkeys:
            self.assert_arraysequal(nb[k] , b[k])
            nb[k] += 10.
            self.assert_arraysequal(nb[k] , b[k]+10.)
    ##
    def test_b_add_err(self):
        """ b.add err """
        global b
        with self.assertRaises(ValueError):
            b.add(bkeys[1],10.)
    ##
    def test_b_getitem_err(self):
        """ b[k] err """
        global b
        with self.assertRaises(KeyError):
            x = b['pseudoscalar']
    ##
    def test_b_buf_err(self):
        """ b.flat assignment err """
        global b,bbuf
        with self.assertRaises(ValueError):
            b.flat = bbuf[:-1]
    ##
    def test_b_del_err(self):
        """ del b[k] """
        global b
        with self.assertRaises(NotImplementedError):
            del b['scalar']
    ##
    def test_pickle(self):
        global b
        sb = pckl.dumps(b)
        c = pckl.loads(sb)
        for k in b:
            self.assert_arraysequal(b[k],c[k])
    ##
    def test_pickle_gvar(self):
        b = BufferDict(dict(a=gv.gvar(1,2),b=[gv.gvar(3,4),gv.gvar(5,6)]))
        sb = pckl.dumps(b)
        c = pckl.loads(sb)
        for k in b:
            self.assert_gvclose(b[k],c[k],rtol=1e-6)
    ##
##    
        
if __name__ == '__main__':
    unittest.main()

