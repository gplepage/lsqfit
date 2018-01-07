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

   .. automethod:: simulated_fit_iter(n=None, pexact=None, **kargs)

   .. automethod:: bootstrap_iter(n=None, datalist=None)

   .. automethod:: dump_p(filename)

   .. automethod:: dump_pmean(filename)

   .. automethod:: nonlinear_fit.load_parameters(filename)

   .. automethod:: check_roundoff(rtol=0.25,atol=1e-6)

   .. automethod:: set(clear=False, **defaults)

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
:mod:`lsqfit` provides support for doing Bayesian integrals, using results
from a least-squares fit to optimize the multi-dimensional integral. This
is useful for severely non-Gaussian situations. Module :mod:`vegas` is
used to do the integrals, using an adaptive Monte Carlo algorithm.

The integrator class is:

.. autoclass:: lsqfit.BayesIntegrator(fit, limit=1e15, scale=1, pdf=None, adapt_to_pdf=True, svdcut=1e-15)

   .. automethod:: __call__(f=None, pdf=None, adapt_to_pdf=None, **kargs)

A class that describes the Bayesian probability distribution associated
with a fit is:

.. autoclass:: lsqfit.BayesPDF(fit, svdcut=1e-15)

   .. automethod:: __call__(p)

   .. automethod:: logpdf(p)


:class:`lsqfit.MultiFitter` Classes
-------------------------------------
:class:`lsqfit.MultiFitter` provides a framework for fitting multiple pieces
of data using a set of custom-designed models, derived  from
:class:`lsqfit.MultiFitterModel`, each of which  encapsulates a particular fit
function. This framework was developed to support the :mod:`corrfitter`
module, but is more general. Instances of model classes  associate specific
subsets of the fit data with  specific subsets of the fit parameters. This
allows fit problems to be broken down down into more manageable pieces, which
are then aggregated by :class:`lsqfit.MultiFitter` into a single fit.

A trivial example of a model would be one that encapsulates
a linear fit function::

   import numpy as np
   import lsqfit

   class Linear(lsqfit.MultiFitterModel):
       def __init__(self, datatag, x, intercept, slope):
           super(Linear, self).__init__(datatag)
           # the independent variable
           self.x = np.array(x)
           # keys used to find the intercept and slope in a parameter dictionary
           self.intercept = intercept
           self.slope = slope

       def fitfcn(self, p):
           if self.slope in p:
               return p[self.intercept] + p[self.slope] * self.x
           else:
               # slope parameter marginalized
               return len(self.x) * [p[self.intercept]]

       def buildprior(self, prior, mopt=None, extend=False):
           " Extract the model's parameters from prior. "
           newprior = {}
           newprior[self.intercept] = prior[self.intercept]
           if mopt is None:
               # slope parameter marginalized if mopt is not None
               newprior[self.slope] = prior[self.slope]
           return newprior

       def builddata(self, data):
           " Extract the model's fit data from data. "
           return data[self.datatag]

Imagine four sets of data, each corresponding to ``x=1,2,3,4``, all of which
have the same intercept but different slopes::

    data = gv.gvar(dict(
        d1=['1.154(10)', '2.107(16)', '3.042(22)', '3.978(29)'],
        d2=['0.692(10)', '1.196(16)', '1.657(22)', '2.189(29)'],
        d3=['0.107(10)', '0.030(16)', '-0.027(22)', '-0.149(29)'],
        d4=['0.002(10)', '-0.197(16)', '-0.382(22)', '-0.627(29)'],
        ))

To find the common intercept, we define a model for each set of
data::

   models = [
      Linear('d1', x=[1,2,3,4], intercept='a', slope='s1'),
      Linear('d2', x=[1,2,3,4], intercept='a', slope='s2'),
      Linear('d3', x=[1,2,3,4], intercept='a', slope='s3'),
      Linear('d4', x=[1,2,3,4], intercept='a', slope='s4'),
      ]

This says that ``data['d3']``, for example, should be fit with  function
``p['a'] + p['s3'] * np.array([1,2,3,4])`` where ``p`` is  a dictionary of fit
parameters.  The models here all share the same intercept, but have different
slopes. Assume that we know *a priori* that the intercept and slopes are all
order one::

   prior = gv.gvar(dict(a='0(1)', s1='0(1)', s2='0(1)', s3='0(1)', s4='0(1)'))

Then we can fit all the data to determine the intercept::

   fitter = lsqfit.MultiFitter(models=models)
   fit = fitter.lsqfit(data=data, prior=prior)
   print(fit)
   print('intercept =', fit.p['a'])

The output from this code is::

   Least Square Fit:
     chi2/dof [dof] = 0.49 [16]    Q = 0.95    logGBF = 18.793

   Parameters:
                 a    0.2012 (78)      [  0.0 (1.0) ]
                s1    0.9485 (53)      [  0.0 (1.0) ]
                s2    0.4927 (53)      [  0.0 (1.0) ]
                s3   -0.0847 (53)      [  0.0 (1.0) ]
                s4   -0.2001 (53)      [  0.0 (1.0) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 5/0.0)

   intercept = 0.2012(78)

Model class ``Linear`` is configured to allow
marginalization of the slope parameter, if desired. Calling
``fitter.lsqfit(data=data, prior=prior, mopt=True)`` moves the slope
parameters into the data (by subtracting ``m.x * prior[m.slope]``
from the data for each model ``m``), and does a single-parameter fit for the
intercept::

   Least Square Fit:
     chi2/dof [dof] = 0.49 [16]    Q = 0.95    logGBF = 18.793

   Parameters:
                 a   0.2012 (78)     [  0.0 (1.0) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

   intercept = 0.2012(78)

Marginalization can be useful when fitting large data sets since it
reduces the number of fit parameters and simplifies the fit.

Another variation is to replace the simultaneous fit of the four models
by a chained fit, where one model is fit at a time and its
results are fed into the next fit through that fit's prior. Replacing the
fit code by ::

   fitter = lsqfit.MultiFitter(models=models)
   fit = fitter.chained_lsqfit(data=data, prior=prior)
   print(fit.formatall())
   print('slope =', fit.p['a'])

gives the following output::

   ========== d1
   Least Square Fit:
     chi2/dof [dof] = 0.32 [4]    Q = 0.86    logGBF = 2.0969

   Parameters:
                 a    0.213 (16)     [  0.0 (1.0) ]
                s1   0.9432 (82)     [  0.0 (1.0) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 5/0.0)

   ========== d2
   Least Square Fit:
     chi2/dof [dof] = 0.58 [4]    Q = 0.67    logGBF = 5.3792

   Parameters:
                 a    0.206 (11)     [  0.213 (16) ]
                s2   0.4904 (64)     [   0.0 (1.0) ]
                s1   0.9462 (64)     [ 0.9432 (82) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 5/0.0)

   ========== d3
   Least Square Fit:
     chi2/dof [dof] = 0.66 [4]    Q = 0.62    logGBF = 5.3767

   Parameters:
                 a    0.1995 (90)      [  0.206 (11) ]
                s3   -0.0840 (57)      [   0.0 (1.0) ]
                s1    0.9493 (57)      [ 0.9462 (64) ]
                s2    0.4934 (57)      [ 0.4904 (64) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

   ========== d4
   Least Square Fit:
     chi2/dof [dof] = 0.41 [4]    Q = 0.81    logGBF = 5.9402

   Parameters:
                 a    0.2012 (78)      [  0.1995 (90) ]
                s4   -0.2001 (53)      [    0.0 (1.0) ]
                s1    0.9485 (53)      [  0.9493 (57) ]
                s2    0.4927 (53)      [  0.4934 (57) ]
                s3   -0.0847 (53)      [ -0.0840 (57) ]

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

   intercept = 0.2012(78)

Note how the value for ``s1`` improves with each fit despite the fact that
it appears only in the first fit function. This happens because its value
is correlated with that of the intercept ``a``, which appears in every fit
function.

Chained fits are most useful
with very large data sets when it is possible to break the data into
smaller, more manageable chunks. There are a variety of options for
organizing the chain of fits; these are discussed in the
:meth:`MultiFitter.chained_lsqfit` documentation.


.. autoclass:: lsqfit.MultiFitter(models, mopt=None, ratio=False, fast=True, extend=False, **fitterargs)

   .. automethod:: MultiFitter.lsqfit

   .. automethod:: MultiFitter.chained_lsqfit

   .. automethod:: process_data

   .. automethod:: process_dataset

   .. automethod:: show_plots

   .. automethod:: flatten_models

:class:`lsqfit.MultiFitter` models are derived from the following
class. Methods ``buildprior``, ``builddata``, ``fitfcn``, and
``builddataset`` are not implemented in this base
class. They need to be overwritten by the derived class (except
for ``builddataset`` which is optional).

.. autoclass:: lsqfit.MultiFitterModel(datatag, ncg=1)

   .. automethod:: MultiFitterModel.buildprior

   .. automethod:: MultiFitterModel.builddata

   .. automethod:: MultiFitterModel.fitfcn

   .. automethod:: MultiFitterModel.builddataset

   .. automethod:: MultiFitterModel.get_prior_keys

   .. automethod:: MultiFitterModel.prior_key

.. :class:`lsqfit.MultiFitter` was inspired by an unconventional

Requirements
------------
:mod:`lsqfit` relies heavily on the :mod:`gvar`, and :mod:`numpy` modules.
Also the fitting and minimization routines are from
the Gnu Scientific Library (GSL) and/or the Python :mod:`scipy` module.


