import os
import unittest
import collections
import numpy as np
import random
import gvar as gv
from gvar import *
from gvar.powerseries import PowerSeries

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
        yN = odeint(y0, [0.0, 0.5, 1.0])
        self.assert_arraysclose(yN, numpy.exp([0.5, 1.0]))
    
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
        self.assertTrue(
            gv.equivalent(odeint(y1, (2, 0)), y0, rtol=1e-6, atol=1e-6)
            )

    def test_vector(self):
        # harmonic oscillator with vectors
        def f(x, y):
            return numpy.array([y[1], -y[0]])
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-10)
        y0 = [0., 1.]
        y1 = odeint(y0, (0, 1))
        exact = [numpy.sin(1), numpy.cos(1)]
        self.assert_arraysclose(y1, exact)
        yN = odeint(y0, [0., 0.5, 1.0])
        self.assert_arraysclose(
            yN, 
            [[numpy.sin(0.5), numpy.cos(0.5)], [numpy.sin(1.0), numpy.cos(1.0)]],
            )

    def test_vector_dict(self):
        # harmonic oscillator with vectors
        def f(x, y):
            deriv = {}
            deriv['y'] = y['dydx']
            deriv['dydx'] = - y['y']
            return deriv
        odeint = ode.DictIntegrator(deriv=f, h=1, tol=1e-10)
        y0 = dict(y=0., dydx=1.)
        y1 = odeint(y0, (0, 1))
        exact = [numpy.sin(1), numpy.cos(1)]
        self.assert_arraysclose([y1['y'], y1['dydx']], exact)
        yN = odeint(y0, [0., 0.5, 1.0])
        self.assert_arraysclose(
            [[yN[0]['y'], yN[0]['dydx']], [yN[1]['y'], yN[1]['dydx']]], 
            [[numpy.sin(0.5), numpy.cos(0.5)], [numpy.sin(1.0), numpy.cos(1.0)]],
            )

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
        self.assertTrue(
            gv.equivalent(odeint(y1, (x1, x0)), y0, rtol=1e-6, atol=1e-6)
            )

    def test_delta(self):
        def delta(yerr, y, delta_y):
            return np.max(
                np.abs(yerr) / (np.abs(y) + np.abs(delta_y))
                )
        def f(x, y):
            return y * (1 + 0.1j)
        odeint = ode.Integrator(deriv=f, h=1, tol=1e-13, delta=delta)
        y0 = 1
        y1 = odeint(y0, (0, 1))
        exact = numpy.exp(1 + 0.1j)
        self.assertAlmostEqual(y1, exact)

    def test_integral(self):
        def f(x):
            return 3 * x**2
        ans = ode.integral(f, (1,2), tol=1e-10)
        self.assertAlmostEqual(ans, 7)
        def f(x):
            return numpy.array([[1.], [2 * x], [3 * x**2]])
        ans = ode.integral(f, (1, 2))
        self.assert_arraysclose(ans, [[1], [3], [7]])
        def f(x):
            return dict(a=1., b=[2 * x, 3 * x**2])
        ans = ode.integral(f, (1, 2))
        self.assertAlmostEqual(ans['a'], 1.)
        self.assert_arraysclose(ans['b'], [3, 7])

    def test_integral_gvar(self):
        def f(x, a=gvar(1,1)):
            return a * 3 * x**2
        ans = ode.integral(f, (1,2), tol=1e-10)
        self.assertAlmostEqual(ans.mean, 7)
        self.assertAlmostEqual(ans.sdev, 7)
        a = gvar(1, 1)
        def f(x, a=a):
            return a * numpy.array([[1.], [2 * x], [3 * x**2]])
        ans = ode.integral(f, (1, 2))
        self.assert_arraysclose(mean(ans), [[1], [3], [7]])
        self.assert_arraysclose(sdev(ans), [[1], [3], [7]])
        def f(x, a=a):
            return dict(a=a, b=[2 * a * x, 3 * a * x**2])
        ans = ode.integral(f, (1, 2))
        self.assertAlmostEqual(ans['a'].mean, 1.)
        self.assert_arraysclose(mean(ans['b']), [3, 7])
        self.assertAlmostEqual(ans['a'].sdev, 1.)
        self.assert_arraysclose(sdev(ans['b']), [3, 7])
        self.assertTrue(gv.equivalent(
            ans,
            dict(a=a, b=[3 * a, 7 * a]), 
            rtol=1e-6, atol=1e-6)
            )

    def test_solution(self):
        def f(x, y):
            return y
        y = ode.Integrator(deriv=f, h=1, tol=1e-13).solution(0., 1.)
        self.assertAlmostEqual(y(1.), np.exp(1.))
        self.assertEqual(y.x, 1)
        self.assertAlmostEqual(y.y, np.exp(1))
        self.assertAlmostEqual(y(1.5), np.exp(1.5))
        self.assertEqual(y.x, 1.5)
        self.assertAlmostEqual(y.y, np.exp(1.5))
        self.assert_arraysclose(y([2.0, 2.5]), np.exp([2.0, 2.5]))
        self.assertAlmostEqual(y.x, 2.5)
        self.assertAlmostEqual(y.y, np.exp(2.5))

    def test_solution_dict(self):
        def f(x, y):
            return dict(y=y['y'])
        y = ode.DictIntegrator(deriv=f, h=1, tol=1e-13).solution(0., dict(y=1.))
        y1 = y(1.)
        self.assertAlmostEqual(y1['y'], np.exp(1.))
        yN = y([0.5, 1.])
        self.assertAlmostEqual(yN[0]['y'], np.exp(0.5))
        self.assertAlmostEqual(yN[1]['y'], np.exp(1.0))

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
        " out of range, extrap_order=3 "
        x = np.array([0, 1., 3.])
        x0 = x[0]
        y = self.f(x)
        yp= self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], warn=False)
        for xi in [ -1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(self.f(xi), s(xi))
            self.assertAlmostEqual(self.Df(xi), s.D(xi))
            self.assertAlmostEqual(self.D2f(xi), s.D2(xi))
            self.assertAlmostEqual(self.integf(xi, x0), s.integ(xi))
        for xx in [s([]), s.D([]), s.D2([]), s.integ([])]:
            self.assertEqual(list(xx), [])

    def test_out_of_range0(self):
        " out of range, extrap_order=0 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f0(x):
            if np.shape(x) == ():
                return f0(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = self.f(xl)
            ans[x>xr] = self.f(xr)
            return ans
        def Df0(x):
            if np.shape(x) == ():
                return Df0(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans        
        def D2f0(x):
            if np.shape(x) == ():
                return D2f0(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans 
        def integf0(x):
            if np.shape(x) == ():
                return integf0(np.array([x]))[0]
            ans = self.integf(x)
            ans[x<xl] = (x[x<xl] - xl) * self.f(xl)
            ans[x>xr] = self.integf(xr, xl) + (x[x>xr] - xr) * self.f(xr)
            return ans       
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=0, warn=False)
        for xi in [ -1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f0(xi), s(xi))
            self.assertAlmostEqual(Df0(xi), s.D(xi))
            self.assertAlmostEqual(D2f0(xi), s.D2(xi))
            self.assertAlmostEqual(integf0(xi), s.integ(xi))

    def test_out_of_range1(self):
        " out of range, extrap_order=1 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f1(x):
            if np.shape(x) == ():
                return f1(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = self.f(xl) + (x[x<xl] - xl) * self.Df(xl)
            ans[x>xr] = self.f(xr) + (x[x>xr] - xr) * self.Df(xr)
            return ans
        def Df1(x):
            if np.shape(x) == ():
                return Df1(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = self.Df(xl)
            ans[x>xr] = self.Df(xr)
            return ans        
        def D2f1(x):
            if np.shape(x) == ():
                return D2f1(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = 0
            ans[x>xr] = 0
            return ans        
        def integf1(x):
            if np.shape(x) == ():
                return integf1(np.array([x]))[0]
            ans = self.integf(x)
            dx = x[x<xl] - xl
            ans[x<xl] = dx * self.f(xl) + dx**2 * self.Df(xl) / 2.
            dx = x[x>xr] - xr
            ans[x>xr] = self.integf(xr, xl) + dx * self.f(xr) + dx**2 * self.Df(xr) / 2.
            return ans       
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=1, warn=False)
        for xi in [-1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f1(xi), s(xi))
            self.assertAlmostEqual(Df1(xi), s.D(xi))
            self.assertAlmostEqual(D2f1(xi), s.D2(xi))
            self.assertAlmostEqual(integf1(xi), s.integ(xi))

    def test_out_of_range2(self):
        " out of range, extrap_order=2 "
        x = np.array([0, 1., 3.])
        xl = x[0]
        xr = x[-1]
        def f2(x):
            if np.shape(x) == ():
                return f2(np.array([x]))[0]
            ans = self.f(x)
            ans[x<xl] = (
                self.f(xl) + (x[x<xl] - xl) * self.Df(xl) 
                + 0.5 * (x[x<xl] - xl) ** 2 * self.D2f(xl)
                )
            ans[x>xr] = (
                self.f(xr) + (x[x>xr] - xr) * self.Df(xr) 
                + 0.5 * (x[x>xr] - xr) ** 2 * self.D2f(xr)
                )
            return ans
        def Df2(x):
            if np.shape(x) == ():
                return Df2(np.array([x]))[0]
            ans = self.Df(x)
            ans[x<xl] = self.Df(xl) + (x[x<xl] - xl) * self.D2f(xl)
            ans[x>xr] = self.Df(xr) + (x[x>xr] - xr) * self.D2f(xr)
            return ans        
        def D2f2(x):
            if np.shape(x) == ():
                return D2f2(np.array([x]))[0]
            ans = self.D2f(x)
            ans[x<xl] = self.D2f(xl)
            ans[x>xr] = self.D2f(xr)
            return ans        
        def integf2(x):
            if np.shape(x) == ():
                return integf2(np.array([x]))[0]
            ans = self.integf(x)
            dx = x[x<xl] - xl
            ans[x<xl] = (
                dx * self.f(xl) + dx**2 * self.Df(xl) / 2. 
                + dx**3 * self.D2f(xl) / 6.
                )
            dx = x[x>xr] - xr
            ans[x>xr] = (
                self.integf(xr, xl) + dx * self.f(xr) 
                + dx**2 * self.Df(xr) / 2. + dx**3 * self.D2f(xr) / 6.
                )
            return ans       
        y = self.f(x)
        yp = self.Df(x)
        s = cspline.CSpline(x, y, deriv=[yp[0], yp[-1]], extrap_order=2, warn=False)
        for xi in [-1.55, -1., 0., 2., 3., 4., 6.2]:
            self.assertAlmostEqual(f2(xi), s(xi))
            self.assertAlmostEqual(Df2(xi), s.D(xi))
            self.assertAlmostEqual(D2f2(xi), s.D2(xi))
            self.assertAlmostEqual(integf2(xi), s.integ(xi))

    def test_left_natural_bc(self):
        # choose left bdy so that self.D2f(xl) = 0 ==> get exact fcn
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
        # choose right bdy so that self.D2f(xr) = 0 ==> get exact fcn
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

    def test_sin(self):
        " test CSpline with real function "
        x = np.arange(-1.,1.,0.0005)
        fs = cspline.CSpline(x, np.sin(x), [np.cos(x[0]), np.cos(x[-1])])
        for x in [-0.666, -0.333, -0.123, 0.123, 0.333, 0.666]:
            self.assertAlmostEqual(fs(x), np.sin(x))
            self.assertAlmostEqual(fs.D(x), np.cos(x))
            self.assertAlmostEqual(fs.D2(x), -np.sin(x))
            self.assertAlmostEqual(fs.integ(x), -np.cos(x) + np.cos(-1.))

class PowerSeriesTests(object):
    def __init__(self):
        pass

    def assert_close(self, a, b, rtol=1e-8, atol=1e-8):
        np.testing.assert_allclose(a.c, b.c, rtol=rtol, atol=atol)

class test_powerseries(unittest.TestCase, PowerSeriesTests):
    """docstring for test_powerseries"""
    def setUp(self):
        self.order = 10
        self.x = PowerSeries([0.,1.], order=self.order)
        self.x2 = PowerSeries([0., 0., 1.], order=self.order)
        self.z = PowerSeries([0+0j, 1.+0j], order=self.order)
        self.one = PowerSeries([1.], order=self.order)
        self.zero = PowerSeries([0.], order=self.order)
        coef = 1. / np.cumprod([1.] + (1. + np.arange(0., self.order+1.)).tolist())
        self.exp_x = PowerSeries(coef, order=self.order)
        osc_coef = coef * (1j)**np.arange(len(coef))
        self.cos_x = PowerSeries([xi.real for xi in osc_coef], order=self.order)
        self.sin_x = PowerSeries([xi.imag for xi in osc_coef], order=self.order)

    def test_constructor(self):
        " PowerSeries(c, order) "
        x = PowerSeries(1. + np.arange(2*self.order), order=self.order)
        self.assertEqual(len(x.c), self.order + 1)
        np.testing.assert_allclose(x.c, 1. + np.arange(self.order + 1))
        x = PowerSeries(order=self.order)
        y = PowerSeries(numpy.zeros(self.order + 1, object), order=self.order)
        self.assertEqual(len(x.c), len(y.c))
        for xi, yi in zip(x.c, y.c):
            self.assertEqual(xi, yi)
        for i in range(self.order + 1):
            self.assertEqual(x.c[i], y.c[i])
        y = PowerSeries(self.exp_x)      
        for i in range(self.order + 1):
            y.c[i] *= 2
        self.assert_close(y, 2 * self.exp_x)
        self.assert_close(PowerSeries(), PowerSeries([0.]))
        with self.assertRaises(ValueError):
            PowerSeries([])
        with self.assertRaises(ValueError):
            PowerSeries(order=-1)
        with self.assertRaises(ValueError):
            PowerSeries([1,2], order=-1)

    def test_arith(self):
        " x+y x-y x*y x/y x**2 "
        x = self.x
        y = self.exp_x
        self.assert_close(x * x, self.x2)
        self.assert_close(y / y, self.one)
        self.assert_close((y * y) / y, y)
        self.assert_close((x * y) / y, x)
        self.assert_close(y - y, self.zero)
        self.assert_close((x + y) - x, y)
        self.assert_close(x + y - x - y, self.zero)
        self.assert_close(x ** 2, self.x2)
        self.assert_close(y * y * y, y ** 3)
        self.assert_close(2 ** x, self.exp_x ** log(2.))
        self.assert_close(y + y, 2 * y)
        self.assert_close(y + y, y * 2)
        self.assert_close(x + 2, PowerSeries([2, 1], order=self.order))
        self.assert_close(2 + x, PowerSeries([2, 1], order=self.order))
        self.assert_close(x - 2, PowerSeries([-2, 1], order=self.order))
        self.assert_close(2 - x, PowerSeries([2, -1], order=self.order))
        self.assert_close(2 * (y / 2), y)
        self.assert_close(y * (2 / y), 2 * self.one)
        self.assertEqual(y ** 0, 1.)
        self.assert_close(y ** (-2), 1 / y / y)

        # check division where c[0] = 0
        self.assert_close(x / x, PowerSeries([1], order=self.order - 1))
        self.assert_close((x + x ** 2) / x, PowerSeries([1, 1], order=self.order-1))
        self.assert_close((x * x) / self.x2, PowerSeries([1], order=self.order - 2))

        # check error checks
        with self.assertRaises(ZeroDivisionError):
            self.x / self.zero

    def test_sqrt(self):
        " sqrt "
        y = self.exp_x
        self.assert_close(sqrt(y) ** 2, y)
        self.assert_close(sqrt(y ** 2), y)
        self.assert_close(sqrt(y), y ** 0.5)

    def test_exp(self):
        x = self.x
        y = self.exp_x
        self.assert_close(exp(x), self.exp_x)
        self.assert_close(log(exp(y)), y)
        self.assert_close(2 ** y, exp(log(2) * y))
        self.assert_close(y ** x, exp(log(y) * x))
        self.assert_close(y ** 2.5, exp(log(y) * 2.5))

    def test_trig(self):
        jx = self.x * 1j
        x = self.x * (1+0j)
        self.assert_close(sin(x), (exp(jx) - exp(-jx))/2j)
        self.assert_close(cos(x), (exp(jx) + exp(-jx))/2)
        x = self.x
        self.assert_close(self.sin_x, sin(self.x))
        self.assert_close(self.cos_x, cos(self.x))
        self.assert_close(tan(x), sin(x) / cos(x))
        self.assert_close(cos(arccos(x)), x)
        self.assert_close(arccos(cos(1 + x)), 1 + x)
        self.assert_close(sin(arcsin(x)), x)
        self.assert_close(arcsin(sin(1 + x)), 1 + x)
        self.assert_close(tan(arctan(x)), x)
        self.assert_close(arctan(tan(1 + x)), 1 + x)

    def test_hyp(self):
        x = self.x
        self.assert_close(sinh(x), (self.exp_x - 1 / self.exp_x)/2)
        self.assert_close(cosh(x), (self.exp_x + 1 / self.exp_x)/2)
        self.assert_close(tanh(x), sinh(x) / cosh(x))
        self.assert_close(arcsinh(sinh(x)), x)
        self.assert_close(sinh(arcsinh(x)), x)
        self.assert_close(arccosh(cosh(2 + x)), 2 + x)
        self.assert_close(cosh(arccosh(1.25 + x)), 1.25 + x)
        self.assert_close(arctanh(tanh(0.25 + x)), 0.25 + x)
        self.assert_close(tanh(arctanh(0.25 + x)), 0.25 + x)

    def test_call(self):
        self.assert_close(self.sin_x(self.x), sin(self.x))
        f = log(1 + self.x)
        self.assert_close(self.exp_x(log(1 + self.x)), 1 + self.x)

    def test_str(self):
        " str(p) repr(p) "
        self.assertEqual(str(self.x), "[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]")
        y = eval(repr(self.exp_x))
        self.assert_close(y, self.exp_x)

    def test_deviv_integ(self):
        " p.deriv() p.integ() "
        self.assert_close(self.exp_x.integ().deriv(), self.exp_x)
        self.assert_close(self.exp_x.integ(n=2).deriv(n=2), self.exp_x)
        self.assert_close(self.exp_x.integ(n=0), self.exp_x)
        self.assert_close(self.exp_x.deriv(n=0), self.exp_x)
        self.assert_close(self.x.deriv(self.order + 2), PowerSeries([0]))
        self.assert_close(
            self.exp_x.deriv(), 
            PowerSeries(self.exp_x, order=self.order - 1)
            )
        self.assert_close(
            self.exp_x.integ(x0=0), 
            exp(PowerSeries(self.x, order=self.order+1)) - 1
            )

class test_root(unittest.TestCase,ArrayTests):
    def setUp(self):
        pass

    def test_search(self):
        " root.search(fcn, x0) "
        interval = root.search(np.sin, 0.5, incr=0.5, fac=1.)
        np.testing.assert_allclose(interval, (3.0, 3.5))
        self.assertEqual(root.search.nit, 6)
        interval = root.search(np.sin, 1.0, incr=0.0, fac=1.5)
        np.testing.assert_allclose(interval, (27./8., 9./4.))
        with self.assertRaises(RuntimeError):
            root.search(np.sin, 0.5, incr=0.5, fac=1., maxit=5)

    def test_refine(self):
        " root.refine(fcn, interval) "
        def fcn(x):
            return (x + 1) ** 3 * (x - 0.5) ** 11
        r = root.refine(fcn, (0.1, 2.1))
        self.assertAlmostEqual(r, 0.5)
        r = root.refine(fcn, (2.1, 0.1))
        self.assertAlmostEqual(r, 0.5)
        rtol = 0.1/100000
        nit = 0
        for rtol in [0.1, 0.01, 0.001, 0.0001]:
            r = root.refine(fcn, (0.1, 2.1), rtol=rtol)
            self.assertGreater(rtol * 0.5, abs(r - 0.5))
            self.assertGreater(root.refine.nit, nit)
            nit = root.refine.nit
        def f(x, w=gv.gvar(1,0.1)):
            return np.sin(w * x)
        r = root.refine(f, (1, 4))
        self.assertAlmostEqual(r.mean, np.pi)
        self.assertAlmostEqual(r.sdev, 0.1 * np.pi)

class test_linalg(unittest.TestCase, ArrayTests):
    def setUp(self):
        pass

    def make_random(self, a, g='1.0(1)'):
        a = numpy.asarray(a)
        ans = numpy.empty(a.shape, object)
        for i in numpy.ndindex(a.shape):
            ans[i] = a[i] * gvar(g)
        return ans

    def test_det(self):
        m = self.make_random([[1., 0.1], [0.1, 2.]])
        detm = linalg.det(m)
        self.assertTrue(gv.equivalent(detm, m[0,0] * m[1,1] - m[0,1] * m[1,0]))
        m = self.make_random([[1.0,2.,3.],[0,4,5], [0,0,6]])
        self.assertTrue(gv.equivalent(linalg.det(m), m[0, 0] * m[1, 1] * m[2, 2]))
        s, logdet = linalg.slogdet(m)
        self.assertTrue(gv.equivalent(s * gv.exp(logdet), linalg.det(m)))

    def test_inv(self):
        m = self.make_random([[1., 0.1], [0.1, 2.]])
        one = gv.gvar([['1(0)', '0(0)'], ['0(0)', '1(0)']])
        invm = linalg.inv(m)
        self.assertTrue(gv.equivalent(linalg.inv(invm), m))
        for mm in [invm.dot(m), m.dot(invm)]:
            np.testing.assert_allclose(
                gv.mean(mm), [[1, 0], [0, 1]], rtol=1e-10, atol=1e-10
                )
            np.testing.assert_allclose(
                gv.sdev(mm), [[0, 0], [0, 0]], rtol=1e-10, atol=1e-10
                )
        p = linalg.det(m) * linalg.det(invm)
        self.assertAlmostEqual(p.mean, 1.)
        self.assertGreater(1e-10, p.sdev)

    def test_solve(self):
        m = self.make_random([[1., 0.1], [0.2, 2.]])
        b = self.make_random([3., 4.])
        x = linalg.solve(m, b)
        self.assertTrue(gv.equivalent(m.dot(x), b))
        m = self.make_random([[1., 0.1], [0.2, 2.]])
        b = self.make_random([[3., 1., 0.], [4., 2., 1.]])
        x = linalg.solve(m, b)
        self.assertTrue(gv.equivalent(m.dot(x), b))

    def test_eigvalsh(self):
        m = gv.gvar([['2.1(1)', '0(0)'], ['0(0)', '0.5(3)']])
        th = 0.92
        cth = numpy.cos(th)
        sth = numpy.sin(th)
        u = numpy.array([[cth, sth], [-sth, cth]])
        mrot = u.T.dot(m.dot(u))
        vals =  linalg.eigvalsh(mrot)
        self.assertTrue(gv.equivalent(vals[0], m[1, 1]))
        self.assertTrue(gv.equivalent(vals[1], m[0, 0]))

if __name__ == '__main__':
    unittest.main()
