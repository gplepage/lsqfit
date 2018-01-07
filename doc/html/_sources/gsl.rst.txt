GSL Routines
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

Fitters
--------
:mod:`lsqfit` uses routines from the GSL C-library provided it is
installed; GSL is the open-source Gnu Scientific Library. There are two
fitters that are available for use by |nonlinear_fit|.

.. autoclass:: lsqfit.gsl_multifit

.. autoclass:: lsqfit.gsl_v1_multifit


Minimizer
----------
The :func:`lsqfit.empbayes_fit` uses a minimizer from the GSL library
to minimize ``logGBF``.

.. autoclass:: lsqfit.gsl_multiminex(x0, f, tol=1e-4, maxit=1000, step=1, alg='nmsimplex2', analyzer=None)
