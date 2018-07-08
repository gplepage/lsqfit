lsqfit
------

This package facilitates least-squares fitting of noisy data by
multi-dimensional, nonlinear functions of arbitrarily many parameters.
``lsqfit`` supports Bayesian priors for the fit parameters, with arbitrarily
complicated multidimensional Gaussian distributions. A tutorial on fitting is
included in the documentation; documentation is in the ``doc/``
subdirectory — see ``doc/html/index.html`` or <https://lsqfit.readthedocs.io>.

The fitter uses automatic differentiation to compute gradients of the fit
function. This greatly simplifies coding of the fit function since only the
function itself need be coded. Coding is also simplified by using dictionaries
(instead of arrays) for representing fit data and fit priors.

``lsqfit`` makes heavy use of Python package ``gvar``, which
simplifies the analysis of error propagation and the creation of
multi-dimensional Gaussian distributions (for fit priors).

This code has been used on a laptop to fit functions of tens-to-thousands of
parameters to tens-to-thousands of pieces of data.  ``lsqfit`` uses the GNU
Scientific Library (GSL) and/or ``scipy`` to do the fitting, ``numpy`` for
efficient array arithmetic, and ``cython`` to compile efficient code that
interfaces between Python and the C-based GSL.

Information on how to install the components is in the ``INSTALLATION`` file.

To test the libraries try ``make tests``. Some examples are give in the
``examples/`` subdirectory.

Version numbers: Incompatible changes are signaled by incrementing
the ``major`` version number, where version numbers have the form
``major.minor.patch``. The ``minor`` number signals new features, and the
``patch`` number bug fixes.


| Created by G. Peter Lepage (Cornell University) 2008
| Copyright (c) 2008-2018 G. Peter Lepage
