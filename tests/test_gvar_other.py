import os
import unittest
import collections
import numpy as np
import random
import gvar as gv
from gvar import *

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

class test_ode(unittest.TestCase,ArrayTests):
    def setUp(self): pass
        
    def tearDown(self): pass

    def test_scalar(self):
        # exponential (scalar)
        def f(x, y):
            return y

        odeint = ode.Integrator(deriv=f, h=1, tol=1e-13)
        y0 = 1
        y1 = odeint(y0, (0, 1))
        exact = numpy.exp(1)
        self.assertAlmostEqual(y1, exact)
    
    def test_gvar_scalar(self):
        # exponential with errors
        gam = gv.gvar('1.0(1)')
        def f(x, y):
            return gam * y
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-10)
        y0 = gv.gvar('1.0(1)')
        y1 = odeint(y0, (0, 2))
        exact = y0 * np.exp(gam * 2)
        self.assertAlmostEqual((y1 / exact).mean, 1.)
        self.assertGreater(1e-8, (y1 / exact).sdev)

    def test_vector(self):
        # harmonic oscillator with vectors
        def f(x, y):
            return numpy.array([y[1], -y[0]])
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-10)
        y0 = [0., 1.]
        y1 = odeint(y0, (0, 1))
        exact = [numpy.sin(1), numpy.cos(1)]
        self.assert_arraysclose(y1, exact)

    def test_gvar_dict(self):
        # harmonic oscillator with dictionaries and errors
        w2 = gv.gvar('1.00(2)')
        w = w2 ** 0.5
        def f(x, y):
            deriv = {}
            deriv['y'] = y['dydx']
            deriv['dydx'] =  -w2 * y['y']
            return deriv
        odeint = ode.DictIntegrator(deriv=f, h=1, tol=1e-10)
        x0 = 0
        y0 = dict(y=numpy.sin(w*x0), dydx=w * numpy.cos(w*x0))
        x1 = 10
        y1 = odeint(y0, (x0,x1))
        exact = dict(y=numpy.sin(w * x1), dydx=w * numpy.cos(w * x1))
        self.assert_gvclose(y1, exact)

class test_cspline(unittest.TestCase,ArrayTests):
    def setUp(self): pass
        
    def tearDown(self): pass
    
    def f(self, x):
        return 1 + 2. * x + 3 * x ** 2 + 4 * x ** 3

    def Df(self, x):
        return 2. + 6. * x + 12. * x ** 2

    def D2f(self, x):
        return 6. + 24. * x

    def integf(self, x, x0=0):
        return x + x**2 + x**3 + x**4 - (x0 + x0**2 + x0**3 + x0**4)

    def test_normal(self):
        x = np.array([0, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], warn=False)
        x = np.arange(0.4, 3., 0.4)
        for xi in x:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))

    def test_out_of_range(self):
        x = np.array([0, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], warn=False)
        for xi in [-1.5, -1., 0., 3., 4.]:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))


    def test_left_natural_bc(self):
        x = np.array([-0.25, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[None, yp[-1]], warn=False)
        x = [-1., 0., 0.5, 2., 3., 4.]
        for xi in x:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))

    def test_right_natural_bc(self):
        x = np.array([-3., -1. , -0.25])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], None], warn=False)
        x = [-5., -2., -1., 0., 0.5, 2.]
        for xi in x:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))

    def test_gvar(self):
        x = gvar(['0(1)', '1(1)', '3(1)'])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], warn=False)
        for xi in x:
            self.assert_gvclose(self.f(xi), s(xi))
            self.assert_gvclose(self.Df(xi), s.D(xi))
            self.assert_gvclose(self.integf(xi, x0), s.integ(xi))
        x = np.arange(0.4, 3., 0.4)
        for xi in x:
            self.assertAlmostEqual(self.f(xi), mean(s(xi)))
            self.assertGreater(1e-9, sdev(s(xi)))
            self.assertAlmostEqual(self.Df(xi), mean(s.D(xi)))
            self.assertGreater(1e-9, sdev(s.D(xi)))
            self.assertAlmostEqual(self.D2f(xi), mean(s.D2(xi)))
            self.assertGreater(1e-9, sdev(s.D2(xi)))
            self.assertAlmostEqual(mean(self.integf(xi, x0.mean)), mean(s.integ(xi)))
            self.assert_gvclose(s.integ(xi), self.integf(xi, x0))

if __name__ == '__main__':
    unittest.main()
