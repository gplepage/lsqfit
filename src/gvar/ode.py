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

import sys

import numpy
import gvar

TINY = sys.float_info.min

class Integrator(object):
    """ Integrate ``dy/dx = deriv(x,y)``.

    An :class:`Integrator` object ``odeint`` integrates ``dy/dx = f(x,y)``
    to obtain ``y(x1)`` from ``y0 = y(x0)``. ``y`` and ``f(x,y)`` can
    be scalars or :mod:`numpy` arrays. Typical usage is illustrated 
    by the following code for integrating ``dy/dx = y``::

        from gvar.ode import Integrator

        def f(x, y):
            return y

        odeint = Integrator(deriv=f,  tol=1e-8)
        y0 = 1.
        y1 = odeint(y0, interval=(0, 1.))
        y2 = odeint(y1, interval=(1., 2.))
        ...

    Here the first call to ``odeint`` integrates the differential equation
    from ``x=0`` to ``x=1`` starting with ``y=y0`` at ``x=0``; the result
    is ``y1=exp(1)``, of course. Similarly the second call to ``odeint``
    continues the integration from ``x=1`` to ``x=2``, giving ``y2=exp(2)``.

    If the ``interval`` is a list with more than two entries,
    then ``odeint(y0, interval=[x0, x1, x2 ...])`` in the example above
    returns an array of solutions for points ``x1, x2 ...``. So the example
    above could have been written equivalently as ::

        ...

        odeint = Integrator(deriv=f,  tol=1e-8)
        y0 = 1.
        y1, y2 ... = odeint(y0, interval=[0, 1., 2. ...])


    An alternative interface creates a new function which is the 
    solution of the differential equation for specific initial conditions.
    The code above could be rewritten::

        x0 = 0.         # initial conditions
        y0 = 1.
        y = Integrator(deriv=f, tol=1e-8).solution(x0, y0)
        y1 = y(1)
        y2 = y(2)
        ...

    Here method :meth:`Integrator.solution` returns a function ``y(x)`` 
    where: a) ``y(x0) = y0``; and b) ``y(x)`` uses the integator to 
    integrate the differential equation to point ``x`` starting 
    from  the last point at which ``y`` was evaluated 
    (or from ``x0`` for the first call to ``y(x)``). The function can
    also be called with an array of ``x`` values, in which case an 
    array containing the corresponding ``y`` values is returned.

    The integrator uses an adaptive Runge-Kutta algorithm that adjusts
    the integrator's step size to obtain relative accuracy ``tol`` in the solution.
    An initial step size can be set in the :class:`Integrator` by specifying
    parameter ``h``. A minimum step size ``hmin`` can also be specified;
    the :class:`Integrator` raises an exception if the step size becomes
    smaller than ``hmin``. The :class:`Integrator` keeps track of the
    number of good steps, where ``h`` is increased, and the number of
    bad steps, where ``h`` is decreased and the step is repeated:
    ``odeint.ngood`` and ``odeint.nbad``, respectively.

    A custom criterion for step-size changes can be implemented by 
    specifying a function for parameter delta. This is a function
    ``delta(yerr, y, delta_y)`` --- of the estimated error ``yerr``
    after a given step, the proposed value for ``y``, and the 
    proposed change ``delta_y`` in ``y`` --- that returns a number 
    to compare with tolerance ``tol``. The step size is 
    decreased and the step repeated if ``delta(yerr, y, delta_y) > tol``;
    otherwise the step is accepted and the step size increased. 
    The default definition of ``delta`` is roughly equivalent to::

        import numpy as np
        import gvar as gv
        
        def delta(yerr, y, delta_y):
            return np.max(
                np.fabs(yerr) / (np.fabs(y) + np.fabs(delta_y) + gv.ode.TINY)
                )
    
    A custom definition can be used to allow an ``Integrator`` to 
    work with data types other than floats or :mod:`numpy` arrays of floats. 
    All that is required of the data type is that it support 
    ordinary arithmetic. Therefore, for example, defining 
    ``delta(yerr, y, delta_y)`` with ``np.abs()`` instead of ``np.fabs()``
    allows ``y`` to be complex valued. (Actually the default ``delta`` 
    allows this as well.)

    An analyzer ``analyzer(x,y)`` can be specified using parameter
    ``analyzer``. This function is called after every full step of
    the integration, with the current values of ``x`` and ``y``. 
    Objects of type :class:`gvar.ode.Solution` are examples of 
    (simple) analyzers.

    :param deriv: Function of ``x`` and ``y`` that returns ``dy/dx``. 
        The return value should have the same shape as ``y`` if arrays
        are used.
    :param tol: Relative accuracy in ``y`` relative to ``|y| + h|dy/dx|``
        for each step in the integration. Any integration step that achieves
        less precision is repeated with a smaller step size. The step size
        is increased if precision is higher than needed. Default is 1e-5.
    :type tol: float
    :param h: Absolute value of initial step size. The default value equals the
        entire width of the integration interval.
    :type h: float or None
    :param hmin: Smallest step size allowed. An exception is raised
        if a smaller step size is needed. This is mostly useful for 
        preventing infinite loops caused by programming errors. The 
        default value is zero (which does *not* prevent infinite loops).
    :type hmin: float or None
    :param delta: Function ``delta(yerr, y, delta_y)`` that returns 
        a number to be compared  with ``tol`` at each integration step: 
        if it is larger than ``tol``, the step is repeated with a smaller 
        step size; if it is smaller the step is accepted and a larger 
        step size used for the subsequent step. Here ``yerr`` is an 
        estimate of the error in ``y`` on the last step; ``y`` is the 
        proposed value; and ``delta_y`` is the change in ``y`` over 
        the last step.
    :param analyzer: Function of ``x`` and ``y`` that is called after each
        step of the integration. This can be used to analyze intermediate
        results. 
    """
    def __init__(self, deriv=None, tol=1e-5, h=None, hmin=None, delta=None, analyzer=None):
        self.deriv = deriv
        self.tol = tol
        self.h = h
        self.hmin = hmin
        self.delta = delta
        self.analyzer = analyzer
        self.ngood = 0
        self.nbad = 0

    def __call__(self, y0, interval):
        """ Integrate from ``x0`` to ``x1`` where ``interval=(x0,x1)`` and ``y0=y(x0)``. """
        if len(interval) > 2:
            ans = []
            xlast = interval[0]
            ylast = y0
            for x in interval[1:]:
                y = self(ylast, interval=(xlast,x))
                ans.append(y)
            return numpy.array(ans)
        if self.deriv is None or interval[1] == interval[0]:
            return y0
        tol = abs(self.tol)
        self.nbad = 0
        self.ngood = 0
        x0, x1 = interval

        h = self.h if (self.h is not None and self.h != 0) else (x1 - x0)
        h = abs(h)
        xdir = 1 if x1 > x0 else -1
        hmin = 0.0 if self.hmin is None else abs(self.hmin)
        x = x0
        y = numpy.asarray(y0)
        y_shape = y.shape
        while (xdir>0 and x<x1) or (xdir<0 and x>x1):
            hold = h
            xold = x
            yold = y
            if h > abs(x - x1):
                h = abs(x - x1)
            x, y, yerr = rk5_stepper(xold, h*xdir, yold, self.deriv, errors=True)
            if self.delta is not None:
                delta = self.delta(yerr, y, y-yold)
            else:
                try:
                    delta = numpy.fabs(yerr) / (numpy.fabs(y) + numpy.fabs(y-yold) + TINY)
                except (AttributeError, TypeError):
                    # kludge having to do with idiocyncracy in numpy:
                    #     fabs doesn't work on, eg, array([1.,-2.], object);
                    #     the 'object' messes it up -- get AttributeError.
                    #     Unfortunately abs can't be made to work for GVars.
                    delta = numpy.abs(yerr) / (numpy.abs(y) + numpy.abs(y-yold) + TINY)
                delta = numpy.max(delta)
                if isinstance(delta, gvar.GVar):
                    delta = delta.mean
            if delta >= tol:
                # smaller step size -- adjust and redo step
                h *= 0.97 * (tol / delta) ** 0.25
                if h < hmin:
                    raise RuntimeError(
                        'step size smaller than hmin: %g < %g' % (h, hmin)
                        )
                x = xold
                y = yold
                self.nbad += 1
            else:
                # larger step size -- adjust and continue
                if delta > 0:
                    h *= 0.97 * (tol / delta) ** 0.20
                else:
                    h *= 2.
                self.ngood += 1
            if self.analyzer is not None:
                self.analyzer(x, y)
        return y

    def solution(self, x0, y0):
        """ Create a solution function ``y(x)`` such that ``y(x0) = y0``. 

        A list of solution values ``[y(x0), y(x1) ...]`` is returned if the
        function is called with a list ``[x0, x1 ...]`` of ``x`` values.
        """
        def soln(x):
            if numpy.size(x) > 1:
                x = [soln.x] + list(x)
                ans = self(soln.y, interval=x)
                soln.x = x[-1]
                soln.y = ans[-1]
                return ans
            else:
                soln.y = self(soln.y, interval=(soln.x, x))
                soln.x = x
                return soln.y
        soln.x = x0
        soln.y = y0
        return soln

def rk5_stepper(x, h, y , deriv, errors=False):
    """ Compute y(x+h) from y and dy/dx=deriv(x,y).

    Uses a one-step 5th-order Runge-Kutta algorithm. 

    Returns x+h, y(x+h) if errors is False; otherwise
    returns x+h, y(x+h), yerr where yerr is an error 
    estimate.
        
    Adapted from Numerical Recipes.
    """
    k1 = h * deriv(x,y)
    k2 = h * deriv(x+0.2*h, y+0.2*k1)
    k3 = h * deriv(x+0.3*h, y+0.075*k1+0.225*k2)
    k4 = h * deriv(x+0.6*h, y+0.3*k1-0.9*k2+1.2*k3)
    k5 = h * deriv(x+h, y-.2037037037037037037037037*k1
             +2.5*k2-2.592592592592592592592593*k3
             +1.296296296296296296296296*k4)
    k6 = h * deriv(x+0.875*h, y+.2949580439814814814814815e-1*k1
             +.341796875*k2+.4159432870370370370370370e-1*k3
             +.4003454137731481481481481*k4+.61767578125e-1*k5)
    yn = y + (.9788359788359788359788361e-1*k1
             +.4025764895330112721417070*k3
             +.2104377104377104377104378*k4
             +.2891022021456804065499718*k6)
    xn = x+h
    if errors:
        yerr = (-.429377480158730158730159e-2*k1
                +.186685860938578329882678e-1*k3
                -.341550268308080808080807e-1*k4
                -.1932198660714285714285714e-1*k5
                +.391022021456804065499718e-1*k6)
        return xn,yn,yerr
    else:
        return xn,yn

class DictIntegrator(Integrator):
    """ Integrate ``dy/dx = deriv(x,y)`` where ``y`` is a dictionary.

    An :class:`DictIntegrator` object ``odeint`` integrates ``dy/dx = f(x,y)``
    to obtain ``y(x1)`` from ``y0 = y(x0)``. ``y`` and ``f(x,y)`` are 
    dictionary types having the same keys, and containing scalars 
    and/or :mod:`numpy` arrays as values. Typical usage is::

        from gvar.ode import DictIntegrator

        def f(x, y):
            ...

        odeint = DictIntegrator(deriv=f,  tol=1e-8)
        y1 = odeint(y0, interval=(x0, x1))
        y2 = odeint(y1, interval=(x1, x2))
        ...

    The first call to ``odeint`` integrates from ``x=x0`` to ``x=x1``,
    returning ``y1=y(x1)``. The second call continues the integration
    to ``x=x2``, returning ``y2=y(x2)``. Multiple integration points
    can be specified in ``interval``, in which case a list of the 
    corresponding ``y`` values is returned: for example, ::

        odeint = DictIntegrator(deriv=f,  tol=1e-8)
        y1, y2 ... = odeint(y0, interval=[x0, x1, x2 ...])
    
    The integrator uses an adaptive Runge-Kutta algorithm that adjusts
    the integrator's step size to obtain relative accuracy ``tol`` in the solution.
    An initial step size can be set in the :class:`DictIntegrator` by specifying
    parameter ``h``. A minimum ste psize ``hmin`` can also be specified;
    the :class:`Integrator` raises an exception if the step size becomes
    smaller than ``hmin``. The :class:`DictIntegrator` keeps track of the
    number of good steps, where ``h`` is increased, and the number of
    bad steps, where ``h`` is decreases and the step is repeated:
    ``odeint.ngood`` and ``odeint.nbad``, respectively.

    An analyzer ``analyzer(x,y)`` can be specified using parameter
    ``analyzer``. This function is called after every full step of
    the integration with the current values of ``x`` and ``y``.     
    Objects of type :class:`gvar.ode.Solution` are examples of 
    (simple) analyzers.


    :param deriv: Function of ``x`` and ``y`` that returns ``dy/dx``. 
        The return value should be a dictionary with the same 
        keys as ``y``, and values that have the same
        shape as the corresponding values in ``y``.
    :param tol: Relative accuracy in ``y`` relative to ``|y| + h|dy/dx|``
        for each step in the integration. Any integration step that achieves
        less precision is repeated with a smaller step size. The step size
        is increased if precision is higher than needed. 
    :type tol: float
    :param h: Absolute value of initial step size. The default value equals the
        entire width of the integration interval. 
    :type h: float
    :param hmin: Smallest step size allowed. An exception is raised
        if a smaller step size is needed. This is mostly useful for 
        preventing infinite loops caused by programming errors. The 
        default value is zero (which does *not* prevent infinite loops).
    :type hmin: float
    :param analyzer: Function of ``x`` and ``y`` that is called after each
        step of the integration. This can be used to analyze intermediate
        results. 
    """
    def __init__(self, **args):
        super(DictIntegrator, self).__init__(**args)
    def __call__(self, y0, interval):
        if len(interval) > 2:
            ans = []
            xlast = interval[0]
            ylast = y0
            for x in interval[1:]:
                y = self(ylast, interval=(xlast,x))
                ans.append(y)
            return ans
        if not isinstance(y0, gvar.BufferDict):
            y0 = gvar.BufferDict(y0)
        deriv_orig = self.deriv
        def deriv(x, y):
            y = gvar.BufferDict(y0, buf=y)
            dydx = gvar.BufferDict(deriv_orig(x, y), keys=y0.keys())
            return dydx.buf
        self.deriv = deriv
        ans = super(DictIntegrator, self).__call__(y0.buf, interval)
        self.deriv = deriv_orig
        return gvar.BufferDict(y0, buf=ans)
        
class Solution:
    """ ODE analyzer for storing intermediate values.

    Usage: eg, given ::
    
        odeint = Integrator(...)
        soln = Solution()
        y0 = ...
        y = odeint(y0, interval=(x0, x), analyzer=soln)

    then the ``soln.x[i]`` are the points at which the integrator 
    evaluated the solution, and ``soln.y[i]`` is the solution
    of the differential equation at that point.
    """
    def __init__(self):
        self.x = []
        self.y = []

    def __call__(self,x,y):
        self.x.append(x)
        self.y.append(y)
