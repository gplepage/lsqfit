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

Linear Algebra
---------------------

The module :mod:`gvar.linalg` implements several methods for doing basic 
linear algebra with matrices whose elements can be either numbers or 
:class:`gvar.GVar`\s:

.. automethod:: gvar.linalg.det

.. automethod:: gvar.linalg.slogdet

.. automethod:: gvar.linalg.inv

.. automethod:: gvar.linalg.solve

.. automethod:: gvar.linalg.eigvalsh


Ordinary Differential Equations 
------------------------------------------------

The module :mod:`gvar.ode` implements two classes for integrating systems
of first-order differential equations using an adaptive Runge-Kutta 
algorithm. One integrates scalar- or array-valued equations, while the 
other integrates dictionary-valued equations: 

.. autoclass:: gvar.ode.Integrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

.. autoclass:: gvar.ode.DictIntegrator(deriv, tol=1e-05, h=None, hmin=None, analyzer=None)

A simple analyzer class is:

.. autoclass:: gvar.ode.Solution()


One-Dimensional Integration
----------------------------

The module :mod:`gvar.ode` also provides a method for evaluating 
one-dimensional integrals (using its adaptive Runge-Kutta algorithm):

.. automethod:: gvar.ode.integral

Power Series
--------------
.. automodule:: gvar.powerseries
    :synopsis: Power series arithmetic and evaluation.

.. autoclass:: gvar.powerseries.PowerSeries
    :members:


Root Finding
--------------

The module :mod:`gvar.root` contains methods for finding the roots of 
of one-dimensional functions: that is, finding ``x`` such that 
``fcn(x)=0`` for a given function ``fcn``. It has two routines. The 
first does a coarse search for an interval containing a root:

.. automethod:: gvar.root.search

The second method refines estimates for a root given an interval 
containing one:

.. automethod:: gvar.root.refine

