lsqfit
------

This package facilitates least-squares fitting of noisy data by
multi-dimensional, nonlinear functions of arbitrarily many parameters. The
central package is ``lsqfit`` which provides the fitting capability. ``lsqfit``
makes heavy use of package ``gvar``, which provides tools for the analysis of
error propagation, and also for the creation of complicated multi-dimensional
Gaussian distributions. (``gvar`` is  distributed separately.) ``lsqfit``
supports Bayesian priors for the fit parameters, with arbitrarily complicated
multidimensional gaussian distributions. A tutorial on fitting is included in
the documentation; documentation is in the ``doc/`` subdirectory — see
``doc/html/index.html`` (or <https://lsqfit.readthedocs.io>).

This code has been used on a laptop to fit functions of tens-to-thousands
of parameters to tens-to-thousands of pieces of data. The use of
dictionaries (rather than arrays) to represent fit data facilitates
simultaneous fits to multiple types of data. Fit-function parameters can
also be represented as dictionaries, usually leading to much more
intelligible code.

These packages use the Gnu Scientific Library (GSL) and/or scipy to do the
fitting, numpy for efficient array arithmetic, and cython to compile efficient
code that interfaces between Python and the C-based GSL. The fitter uses
automatic differentiation to compute gradients, which greatly simplifies the
design of fitting functions.

Information on how to install the components is in the ``INSTALLATION`` file.

To test the libraries try ``make tests``. (Some tests involve random
numbers and so may occasionally — less than 1 in 100 runs — fail due to
rare multi-sigma fluctuations; rerun the tests if they do fail.) Some
examples are give in the ``examples/`` subdirectory.

Versioning: Version numbers for lsqfit are now (5.0 and later) based upon
*semantic  versioning* (http://semver.org). Incompatible changes will be
signaled by incrementing the major version number, where version numbers have
the form major.minor.patch. The minor number signals new features, and  the
patch number bug fixes.


| Created by G. Peter Lepage (Cornell University) 2008
| Copyright (c) 2008-2018 G. Peter Lepage
