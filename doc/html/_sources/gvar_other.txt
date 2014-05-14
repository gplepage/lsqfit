.. _numerical-analysis-modules-in-gvar:

Numerical Analysis Modules in :mod:`gvar`
==========================================

.. |GVar| replace:: :class:`gvar.GVar`

|GVar|\s can be used in many numerical algorithms, to propagates errors 
through the algorithm. A code that is written in pure Python is likely to 
work well with |GVar|\s, perhaps with minor modifications. 
Here we describe some sample numerical codes, included in
:mod:`gvar`, that have been adapted
to work with |GVar|\s, as well as with ``float``\s. 
More examples will follow with time.


Cubic Splines
-----------------

The module :mod:`gvar.cspline` implements a class for smoothing and/or 
interpolating one-dimensional data using cubic splines:

.. autoclass:: gvar.cspline.CSpline


Ordinary Differential Equations
-------------------------------

The module :mod:`gvar.ode` implements two classes for integrating systems
of first-order differential equations using an adaptive Runge-Kutta 
algorithm. One integrates scalar- or array-valued equations, while the 
other integrates dictionary-valued equations:

.. autoclass:: gvar.ode.Integrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

.. autoclass:: gvar.ode.DictIntegrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

A simple analyzer class is:

.. autoclass:: gvar.ode.Solution()


Power Series
--------------
.. automodule:: gvar.powerseries
    :synopsis: Power series arithmetic and evaluation.

.. autoclass:: gvar.powerseries.PowerSeries
    :members:


