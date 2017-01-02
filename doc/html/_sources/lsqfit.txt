:mod:`lsqfit` - Nonlinear Least Squares Fitting
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. automodule:: lsqfit
   :synopsis: Nonlinear least squares fitting.


.. Formal Background
.. ----------------------
.. The formal structure structure of a least-squares problem involves
.. fitting input data :math:`y_i` with functions :math:`f_i(p)` by adjusting
.. fit parameters :math:`p_a` to minimize

.. .. math::

..    \chi^2 &\equiv \sum_{ij} \Delta y(p)_i\,(\mathrm{cov}_y^{-1})_{ij}\,
..    \Delta y(p)_j  \\
..    &\equiv (\Delta y(p))^\mathrm{T}\cdot \mathrm{cov}_y^{-1}\cdot
..    \Delta y(p)

.. where :math:`\mathrm{cov}_y` is the covariance matrix for the input data
.. and

.. .. math::

..     \Delta y(p)_i \equiv f_i(p) - y_i.

.. There are generally two types of input data --- actual data and
.. prior information for each fit parameter --- but we lump these together
.. here since they enter in the same way (that is, the sums over :math:`i`
.. and :math:`j` are over all data and priors).

.. The best-fit values :math:`\overline{p}_a` for the fit parameters are those
.. that minimize :math:`\chi^2`:

.. .. math::

..    (\partial_a \Delta y(\overline{p}))^\mathrm{T}
..    \cdot\mathrm{cov}_y^{-1}\cdot
..    \Delta y(\overline{p}) = 0

.. where the derivatives are :math:`\partial_a = \partial/\partial
.. \overline{p}_a`. The covariance matrix :math:`\mathrm{cov}_p` for these is
.. obtained (approximately) from

.. .. math::

..         (\mathrm{cov^{-1}_p})_{ab} \equiv
..     (\partial_a \Delta y(\overline p))^\mathrm{T}
..     \cdot \mathrm{cov}^{-1}_y \cdot
..     (\partial_b\Delta y(\overline p)).

.. Consequently the variance for any function :math:`g(\overline p)` of the
.. best-fit parameters is given by (approximately)

.. .. math::

..    \sigma^2_{g} = (\partial g(\overline p))^\mathrm{T} \cdot
..    \mathrm{cov}_p \cdot \partial g(\overline p)

.. The definition of the covariance matrix implies that it and any variance
.. :math:`\sigma^2_g` derived from it depend linearly (approximately) on the
.. elements of the input data covariance matrix :math:`\mathrm{cov}_y`, at
.. least when errors are small:

.. .. math::

..    \sigma^2_g \approx \sum_{ij} c(\overline p)_{ij} \,
..     (\mathrm{cov}_y)_{ij}

.. This allows us to associate different portions of the output error
.. :math:`\sigma^2_g` with different parts of the input error
.. :math:`\mathrm{cov}_y`, creating an "error budget" for
.. :math:`g(\overline p)`.
.. Such information helps pinpoint the input errors that most affect the
.. output errors for any particular quantity  :math:`g(\overline p)`,
.. and also indicates how those output errors might change for a given change
.. in input error.

.. The relationship between the input and output errors is only
.. approximately linear because the coefficients in the expansion depend upon
.. the best-fit values for the parameters, and these depend upon the input
.. errors --- but only weakly when errors are small. Neglecting such variation
.. in the parameters, the error budget for any quantity is easily computed
.. using

.. .. math::

..    \frac{\partial (\mathrm{cov}_p)_{ab}}{\partial (\mathrm{cov}_y)_{ij}}
..     = D_{ai}\,D_{bj}

.. where

.. .. math::

..    D_{ai} \equiv (\mathrm{cov}_p \cdot \partial \Delta y \cdot
..       \mathrm{cov}_y^{-1})_{ai}

.. and, trivially,
.. :math:`\mathrm{cov}_p = D\cdot\mathrm{cov}_y\cdot D^\mathrm{T}`.

.. This last formula suggests that

.. .. math::

..    \frac{\partial \overline{p}_a}{\partial y_i} = D_{ai}.

.. This relationship is true in the limit of small errors, as is easily derived
.. from the minimum condition for the fit, which defines (implicitly)
.. :math:`\overline{p}_a(y)`: Differentiating with respect to
.. :math:`y_i` we obtain

.. .. math::

..    (\partial_a \Delta y(\overline{p}))^\mathrm{T}\cdot\mathrm{cov}_y^{-1}\cdot
..    \frac{\partial\Delta y(\overline{p})}{\partial y_i} = 0

.. where we have ignored terms suppressed by a factor of :math:`\Delta y(p)`.
.. This leads immediately to the relationship above.

.. The data's covariance matrix :math:`\mathrm{cov}_y` is sometimes rather
.. singular, making it difficult to invert. This problem is dealt with using
.. an SVD cut: the covariance matrix is diagonalized, some number of the
.. smallest (and therefore least-well determined) eigenvalues and their
.. eigenvectors are discarded, and the inverse matrix is reconstituted from
.. the eigenmodes that remain. (Instead of discarding modes one can replace
.. their eigenvalues by the smallest eigenvalue that is retained; this is less
.. conservative and usually leads to more accurate results.)

nonlinear_fit Objects
---------------------

.. autoclass:: lsqfit.nonlinear_fit(data, fcn, prior=None, p0=None, extend=False, svdcut=1e-12, debug=False, tol=1e-8, maxit=1000, fitter='gsl_multifit', **fitterargs)

   Additional methods are provided for printing out detailed information
   about the fit, testing fits with simulated data,
   doing bootstrap analyses of the fit errors,
   dumping (for later use) and loading parameter values, and checking for roundoff
   errors in the final error estimates:

   .. automethod:: format(maxline=0, pstyle='v')

   .. automethod:: fmt_errorbudget(outputs, inputs, ndecimal=2, percent=True)

   .. automethod:: fmt_values(outputs, ndecimal=None)

   .. automethod:: simulated_fit_iter(n=None, pexact=None, **kargs)

   .. automethod:: bootstrap_iter(n=None, datalist=None)

   .. automethod:: dump_p(filename)

   .. automethod:: dump_pmean(filename)

   .. automethod:: nonlinear_fit.load_parameters(filename)

   .. automethod:: check_roundoff(rtol=0.25,atol=1e-6)

   .. automethod:: set_defaults(clear=False, **defaults)

Functions
---------
.. autofunction:: lsqfit.empbayes_fit

.. autofunction:: lsqfit.wavg

.. autofunction:: lsqfit.gammaQ

.. autofunction:: gvar.add_parameter_distribution

.. autofunction:: gvar.del_parameter_distribution

.. autofunction:: gvar.add_parameter_parentheses

Classes for Bayesian Integrals
-------------------------------

.. autoclass:: lsqfit.BayesPDF(fit, svdcut=1e-15)

   .. automethod:: __call__(p)

   .. automethod:: logpdf(p)

.. autoclass:: lsqfit.BayesIntegrator(fit, limit=1e15, scale=1, pdf=None, svdcut=1e-15)

   .. automethod:: __call__(f=None, mpi=False, pdf=None, **kargs)

Requirements
------------
:mod:`lsqfit` relies heavily on the :mod:`gvar`, and :mod:`numpy` modules.
Also the fitting and minimization routines are from
the Gnu Scientific Library (GSL) and/or the Python :mod:`scipy` module.


