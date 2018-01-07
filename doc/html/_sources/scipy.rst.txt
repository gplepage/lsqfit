:mod:`scipy` Routines
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

Fitter
--------
:mod:`lsqfit` uses routines from the open-source :mod:`scipy` Python module
provided it is installed. These routines are used in place of GSL routines
if the latter are not installed. There is one fitter available for use by
|nonlinear_fit|.

.. autoclass:: lsqfit.scipy_least_squares


Minimizer
----------
The :func:`lsqfit.empbayes_fit` uses a minimizer from the :mod:`scipy`
module to minimize ``logGBF``.

.. autoclass:: lsqfit.scipy_multiminex(x0, f, tol=1e-4, maxit=1000, step=1, alg='nmsimplex2', analyzer=None)
