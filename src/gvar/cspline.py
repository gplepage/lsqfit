# Created by G. Peter Lepage (Cornell University) on 2014-04-27.
# Copyright (c) 2014 G. Peter Lepage. 
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

import warnings
import numpy

class CSpline:
    """ Cubic spline resentation of a function.

    Given ``N`` values of a function ``y[i]`` at ``N`` points
    ``x[i]`` for ``i=0..N-1`` (the 'knots' of the spline), 
    the code ::

        from gvar.cspline import CSpline

        f = CSpline(x, y)

    defines a function ``f`` such that: a) ``f(x[i]) = y[i]`` for 
    all ``i``; and b) ``f(x)`` is 
    continuous, as are its first and second derivatives.
    Function ``f(x)`` is a cubic polynomial between the knots ``x[i]``.

    ``CSpline(x,y)`` creates a *natural spline*, which has zero second
    derivative at the end points, ``x[0]`` and ``x[-1]``. More generally
    one can specify the derivatives of ``f(x)`` at one or both of the
    endpoints::

        f = CSpline(x, y, deriv=[dydx_i, dydx_f])

    where ``dydx_i`` is the derivative at ``x[0]`` and ``dydx_f`` is the 
    derivative at ``x[-1]``. Replacing either (or both) of these with 
    ``None`` results in a derivative corresponding to zero second 
    derivative at that boundary (i.e., a *natural* boundary).

    Derivatives and integrals of the spline function can also be evaluated: 

        ``f.D(x)`` --- first derivative at ``x``;

        ``f.D2(x)`` --- second derivative at ``x``;

        ``f.integ(x)`` --- integral from ``x[0]`` to ``x``.

    :param x: The knots of the spline, where the function values are
        specified.
    :type x: 1-d sequence of numbers
    :param y: Function values at the locations specified by ``x[i]``.
    :type y: 1-d sequence
    :param deriv: Derivatives at initial and final boundaries of the 
        region specified by ``x[i]``. Default value is ``None`` for 
        each boundary.
    :type deriv: 2-component sequence
    :param warn: If ``True``, warnings are generated 
        when the spline function is called for ``x`` values that 
        fall outside of the original range of ``x``\s used to 
        define the spline. Default value is ``True``;
        out-of-range warnings are suppressed if set to ``False``.
    """
    def __init__(self, x, y, deriv=(None, None), warn=True):
        x, y = zip(*sorted(zip(x, y)))
        x = numpy.array(x)
        y = numpy.array(y)
        self.warn = warn

        # solve for dydx
        if x.dtype == object or y.dtype == object:
            a = numpy.zeros(y.shape, object)
            b = numpy.zeros(y.shape, object)
            c = numpy.zeros(y.shape, object)
            d = numpy.zeros(y.shape, object)
            self.intydx = numpy.zeros(y.shape, object)
        else:
            a = numpy.zeros(y.shape, float)
            b = numpy.zeros(y.shape, float)
            c = numpy.zeros(y.shape, float)
            d = numpy.zeros(y.shape, float)
            self.intydx = numpy.zeros(y.shape, float)
        for i in range(1, len(y)-1):
            # m[i, i - 1] 
            a[i] = 1. / (x[i] - x[i - 1])
            # m[i, i + 1] 
            c[i] = 1./ (x[i + 1] - x[i])
            # m[i, i] 
            b[i] = 2. * (a[i] + c[i])
            d[i] = 3. * (
                (y[i] - y[i - 1]) * a[i] ** 2 +
                (y[i + 1] - y[i]) * c[i] ** 2
                )
        if deriv[0] is None:
            # m[0, 0] 
            b[0] = 2. / (x[1] - x[0])
            # m[0, 1] 
            c[0] = 1. / (x[1] - x[0])
            d[0] = 3. * (y[1] - y[0]) / (x[1] - x[0]) ** 2
        else:
            b[0] = 1.
            d[0] = deriv[0]
        if deriv[1] is None:
            # m[-1, -2] 
            a[-1] = 1. / (x[-1] - x[-2])
            # m[-1, -1] 
            b[-1] = 2. / (x[-1] - x[-2])
            d[-1] = 3. * (y[-1] - y[-2]) / (x[-1] - x[-2]) ** 2
        else:
            b[-1] = 1.
            d[-1] = deriv[1]
        self.x = x
        self.y = y
        self.dydx = numpy.array(tri_diag_solve(a, b, c, d))
        self.n = len(self.x)
        ydx = self.integ(self.x)
        self.intydx[0] = ydx[0]
        for i in range(1, self.n):
            self.intydx[i] = self.intydx[i - 1] + ydx[i]

    def __call__(self, x):
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        if self.warn and (numpy.any(x < self.x[0]) or numpy.any(x > self.x[-1])):
            warnings.warn('x outside of spline range: ' + str(x))
        j = numpy.searchsorted(self.x, x)
        j[j <= 0] = 1   
        j[j >= self.n] = self.n - 1
        i = j - 1
        x1 = self.x[i]
        x2 = self.x[j]
        y1 = self.y[i]
        y2 = self.y[j]
        k1 = self.dydx[i]
        k2 = self.dydx[j]
        t = (x - x1) / (x2 - x1)
        a = k1 * (x2 - x1) - (y2 - y1)
        b = - k2 * (x2 - x1) + (y2 - y1)
        ans = (1 - t) * y1 + t * y2 + t * (1-t) * (a * (1 - t) + b * t)
        return ans.reshape(xshape) if xshape != () else ans[0]

    def integ(self, x):
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        if self.warn and (numpy.any(x < self.x[0]) or numpy.any(x > self.x[-1])):
            warnings.warn('x outside of spline range: ' + str(x))
        j = numpy.searchsorted(self.x, x)
        j[j <= 0] = 1   
        j[j >= self.n] = self.n - 1
        i = j - 1
        x1 = self.x[i]
        x2 = self.x[j]
        y1 = self.y[i]
        y2 = self.y[j]
        k1 = self.dydx[i]
        k2 = self.dydx[j]
        t = (x - x1) / (x2 - x1)
        a = k1 * (x2 - x1) - (y2 - y1)
        b = - k2 * (x2 - x1) + (y2 - y1)
        ans = (x2 - x1) * (
            t * (1. - t / 2.) * y1
            + t ** 2 / 2. * y2
            + t ** 2 * (0.5 - 2. * t /3. + t ** 2 / 4.) * a
            + t ** 3 * (1 / 3. - t / 4.) * b
            )
        ans += self.intydx[i]
        return ans.reshape(xshape) if xshape != () else ans[0]

    def D(self, x):
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        if self.warn and (numpy.any(x < self.x[0]) or numpy.any(x > self.x[-1])):
            warnings.warn('x outside of spline range: ' + str(x))
        j = numpy.searchsorted(self.x, x)
        j[j <= 0] = 1   
        j[j >= self.n] = self.n - 1
        i = j - 1
        x1 = self.x[i]
        x2 = self.x[j]
        y1 = self.y[i]
        y2 = self.y[j]
        k1 = self.dydx[i]
        k2 = self.dydx[j]
        t = (x - x1) / (x2 - x1)
        a = k1 * (x2 - x1) - (y2 - y1)
        b = - k2 * (x2 - x1) + (y2 - y1)
        ans = (
            (y2 - y1) / (x2 - x1) 
            + (1 - 2 * t) * (a * (1 - t) + b * t) / (x2 - x1)
            + t * (1 - t) * (b - a) / (x2 - x1)
            )
        return ans.reshape(xshape) if xshape != () else ans[0]

    def D2(self, x):
        x = numpy.asarray(x)
        xshape = x.shape
        x = x.flatten()
        if self.warn and (numpy.any(x < self.x[0]) or numpy.any(x > self.x[-1])):
            warnings.warn('x outside of spline range: ' + str(x))
        j = numpy.searchsorted(self.x, x)
        j[j <= 0] = 1   
        j[j >= self.n] = self.n - 1
        i = j - 1
        x1 = self.x[i]
        x2 = self.x[j]
        y1 = self.y[i]
        y2 = self.y[j]
        k1 = self.dydx[i]
        k2 = self.dydx[j]
        t = (x - x1) / (x2 - x1)
        a = k1 * (x2 - x1) - (y2 - y1)
        b = - k2 * (x2 - x1) + (y2 - y1)
        ans = 2 * (b - 2 * a + (a - b) * 3 * t) / (x2 - x1) ** 2
        return ans.reshape(xshape) if xshape != () else ans[0]

def tri_diag_solve(a, b, c, d):
    """ Solve a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i] for x[i]

    Adapted from: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm. 
    """
    n = len(a)
    a, b, c, d = map(numpy.array, (a, b, c, d))
    c[0] /= b[0]
    d[0] /= b[0]
    for i in range(1, n):
        m = b[i] - a[i] * c[i - 1]
        c[i] /= m
        d[i] = (d[i] - a[i] * d[i - 1]) / m
    x = d
    for i in range(n - 2, -1, -1):
        x[i] -= c[i] * x[i + 1]
    return x
