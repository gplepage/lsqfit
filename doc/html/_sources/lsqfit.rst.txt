:mod:`lsqfit` - Nonlinear Least Squares Fitting
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`
.. |MultiFitter| replace:: :class:`lsqfit.MultiFitter`
.. |~| unicode:: U+00A0
.. |,| unicode:: U+2009 

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

.. autoclass:: lsqfit.nonlinear_fit(data, fcn, prior=None, p0=None, svdcut=1e-12, eps=None, noise=False, debug=False, tol=1e-8, maxit=1000, fitter='gsl_multifit', **fitterargs)

   Additional methods are provided for printing out detailed information
   about the fit, evaluating ``chi**2``, testing fits with simulated data,
   doing bootstrap analyses of the fit errors,
   dumping (for later use) and loading parameter values, and checking for roundoff
   errors in the final error  estimates:

   .. automethod:: format(maxline=0, pstyle='v')

   ..  method:: dchi2(p)
      
      ``chi**2(p) - fit.chi2`` for fit parameters ``p``.

      **Paramters:**
          **p:** Array or dictionary containing values for fit parameters, using
              the same layout as in the fit function.

      **Returns:**
          ``chi**2(p) - fit.chi2`` where ``chi**2(p)`` is the fit's
          ``chi**2`` for fit parameters ``p`` and ``fit.chi2`` is the ``chi**2``
          value for the best fit.

   .. method:: pdf(p)
  
      ``exp(-(chi**2(p) - fit.chi2)/2)`` for fit parameters ``p``.

      ``fit.pdf(p)`` is proportional to the probability density
      function (PDF) used in the fit: ``fit.pdf(p)/exp(fit.pdf.lognorm)``
      is the product of the Gaussian PDF for the data ``P(data|p,M)`` 
      times the Gaussian PDF for the prior ``P(p|M)`` where ``M`` is the model 
      used in the fit (i.e., the fit function and prior). The product of PDFs
      is ``P(data,p|M)`` by Bayes' Theorem; integrating over fit parameters
      p gives the Bayes Factor or Evidence ``P(data|M)``, which is proportional
      to the probability that the fit data come from fit model ``M``. The logarithm 
      of the Bayes Factor should agree with ``fit.logGBF`` when the Gaussian 
      approximation assumed in the fit is accurate.

      ``fit.pdf(p)`` is useful for checking a least-squares fit 
      against the corresponding Bayesian integrals. In the following 
      example, :class:`vegas.PDFIntegrator` from the :mod:`vegas` module
      is used to evaluate Bayesian expectation values of ``s*g`` 
      and its standard deviation where ``s`` and ``g`` are fit 
      parameters::

          import gvar as gv
          import lsqfit
          import numpy as np
          import vegas

          def main():
              # least-squares fit
              x = np.array([0.1, 1.2, 1.9, 3.5])
              y = gv.gvar(['1.2(1.0)', '2.4(1)', '2.0(1.2)', '5.2(3.2)'])
              prior = gv.gvar(dict(a='0(5)', s='0(2)', g='2(2)'))
              fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=fitfcn, debug=True)
              print(fit)

              # create integrator and adapt it to PDF (warmup)
              neval = 10_000 
              nitn = 10     
              expval = vegas.PDFIntegrator(fit.p, pdf=fit.pdf, nproc=4)
              warmup = expval(neval=neval, nitn=nitn)

              # calculate expectation value of g(p)
              results = expval(g, neval=neval, nitn=nitn, adapt=False)
              print(results.summary(True))
              print('results =', results, '\n')

              sg = results['sg']
              sg2 = results['sg2']
              sg_sdev = (sg2 - sg**2) ** 0.5
              print('s*g from Bayes integral:  mean =', sg, '  sdev =', sg_sdev)
              print('s*g from fit:', fit.p['s'] * fit.p['g'])
              print()
              print('logBF =', np.log(results.pdfnorm) - fit.pdf.lognorm)

          def fitfcn(x, p):
              return p['a'] + p['s'] * x ** p['g']

          def g(p):
              sg = p['s'] * p['g']
              return dict(sg=sg, sg2=sg**2)

          if __name__ == '__main__':
              main()

      Here the probability density function used for the expectation values 
      is ``fit.pdf(p)``, and the expectation values are returned 
      in dictionary ``results``. :mod:`vegas` uses adaptive Monte 
      Carlo integration. The  ``warmup`` calls to the integrator are 
      used to adapt it to the probability density function, and 
      then the adapted integrator is  called again to evaluate the 
      expectation value. Parameter ``neval`` is the (approximate)
      number of function calls per iteration of the :mod:`vegas` algorithm
      and ``nitn`` is the number of iterations. We use the integrator to
      calculated the expectation value of ``s*g`` and ``(s*g)**2`` so we can
      compute a mean and standard deviation.

      The output from this code shows that the Gaussian approximation
      for ``s*g`` (0.78(66)) is somewhat different from the result
      obtained from a Bayesian integral (0.49(53))::

          Least Square Fit:
          chi2/dof [dof] = 0.32 [4]    Q = 0.87    logGBF = -9.2027

          Parameters:
                      a    1.61 (90)     [  0.0 (5.0) ]  
                      s    0.62 (81)     [  0.0 (2.0) ]  
                      g    1.2 (1.1)     [  2.0 (2.0) ]  

          Settings:
          svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 18/0.0)

          itn   integral        average         chi2/dof        Q
          -------------------------------------------------------
           1   0.954(11)       0.954(11)           0.00     1.00
           2   0.9708(99)      0.9622(74)          0.74     0.53
           3   0.964(12)       0.9627(63)          0.93     0.47
           4   0.9620(93)      0.9626(52)          0.86     0.56
           5   0.964(14)       0.9629(50)          0.71     0.74
           6   0.957(17)       0.9619(50)          0.65     0.84
           7   0.964(12)       0.9622(46)          0.61     0.90
           8   0.9367(86)      0.9590(42)          0.80     0.73
           9   0.9592(94)      0.9591(39)          0.75     0.80
          10   0.952(13)       0.9584(37)          0.72     0.85

                      key/index          value
          ------------------------------------
                          pdf    0.9584 (37)
           ('f(p)*pdf', 'sg')    0.4652 (23)
          ('f(p)*pdf', 'sg2')    0.5073 (33)

          results = {'sg': 0.4854(20), 'sg2': 0.5293(33)} 

          s*g from Bayes integral:  mean = 0.4854(20)   sdev = 0.5420(17)
          s*g from fit: 0.78(66)

          logBF = -9.1505(39)

      The result ``logBF`` for the logarithm of the Bayes Factor from the 
      integral agrees well with ``fit.logGBF``, the log Bayes Factor
      in the Gaussian approximation. This is evidence that the Gaussian
      approximation implicit in the least squares fit is reliable; the product
      of ``s*g``, however, is not so Gaussian because of the large uncertainties
      (compared to the means) in ``s`` and ``g`` separately.

      **Paramters:**
          **p**: Array or dictionary containing values for fit parameters, using 
            the same layout as in the fit function.

      **Returns:**
          ``exp(-(chi**2(p) - fit.chi2)/2)`` where ``chi**2(p)`` is the fit's
          ``chi**2`` for fit parameters ``p`` and ``fit.chi2`` is the ``chi**2``
          value for the best fit.

   .. automethod:: simulated_fit_iter(n=None, pexact=None, add_priornoise=False, **kargs)

   .. automethod:: simulated_data_iter(n=None, pexact=None, add_priornoise=False)

   .. automethod:: bootstrapped_fit_iter(n=None, datalist=None)

   .. automethod:: check_roundoff(rtol=0.25,atol=1e-6)

   .. automethod:: qqplot_residuals(plot=None)

   .. automethod:: plot_residuals(plot=None)

   .. automethod:: set(clear=False, **defaults)

Functions
---------
.. autofunction:: lsqfit.empbayes_fit

.. autofunction:: lsqfit.wavg

.. function:: lsqfit.gammaQ(a, x)

      Return the normalized incomplete gamma function ``Q(a,x) = 1-P(a,x)``.

      ``Q(a, x) = 1/Gamma(a) * \int_x^\infty dt exp(-t) t ** (a-1) = 1 - P(a, x)``

      Note that ``gammaQ(ndof/2., chi2/2.)`` is the probabilty that one could
      get a ``chi**2`` larger than ``chi2`` with ``ndof`` degrees
      of freedom even if the model used to construct ``chi2`` is correct.


:class:`lsqfit.MultiFitter` Classes
-------------------------------------
:class:`lsqfit.MultiFitter` provides a framework for building component
systems to fit multiple pieces of data using a set of custom-designed models,
derived  from :class:`lsqfit.MultiFitterModel`. Each model  encapsulates:
a) a particular fit function; b) a recipe for assembling the corresponding fit
data from a dictionary that contains all of the data; and c) a recipe for
assembling a fit prior drawn from a dictionary containing all the priors.
This allows fit problems to be broken down down into more manageable pieces,
which are then aggregated by :class:`lsqfit.MultiFitter` into a single fit.

This framework was developed to support the :mod:`corrfitter` module which is
used to analyze 2-point and 3-point correlators generated in Monte Carlo
simulations of quantum field theories (like QCD). The :mod:`corrfitter`
module provides two models to describe correlators: :class:`corrfitter.Corr2`
to describe one  2-point correlator, and :class:`corrfitter.Corr3` to describe
one 3-point  correlator. A typical analysis involves fitting data for a mixture of
2-point and 3-point correlators, with sometimes hundreds of correlators in all.
Each correlator is described by either  a ``Corr2`` model or a ``Corr3``
model. A list of models, one for each  correlator, is handed
:class:`corrfitter.CorrFitter` (derived from  :class:`lsqfit.MultiFitter`) to
fit the models to the correlator data. The models for different
correlators typically share many fit parameters.

A simpler example of a model is one that encapsulates
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
           try:
               return p[self.intercept] + p[self.slope] * self.x
           except KeyError:
               # slope parameter marginalized/omitted
               return len(self.x) * [p[self.intercept]]

       def buildprior(self, prior, mopt=None):
           " Extract the model's parameters from prior. "
           newprior = {}
           newprior[self.intercept] = prior[self.intercept]
           if mopt is None:
               # slope parameter marginalized/omitted if mopt is not None
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
parameters.  Assume that we know *a priori* that the intercept and slopes are all
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

Empirical Bayes tuning can be used with a :mod:`MultiFitter` (see :ref:`empirical-bayes`). 
Continuing from the example just above, we may be uncertain about the prior for the 
intercept. The following code varies the width of that prior to maximize 
the Bayes Factor (``logGBF``)::

   def fitargs(z):
       prior = gv.gvar(dict(s1='0(1)', s2='0(1)', s3='0(1)', s4='0(1)'))
       prior['a'] = gv.gvar(0, np.exp(z))       # np.exp => positive std dev
       return dict(prior=prior, data=data, mopt=True)
    fit,z = fitter.empbayes_fit(0, fitargs)
    print(fit)  
    print('intercept =', fit.p['a'])

The output shows that the data prefer a prior of ``0.0(2)`` for the 
intercept (not surprisingly)::

   Least Square Fit:
     chi2/dof [dof] = 0.55 [16]    Q = 0.92    logGBF = 19.917

   Parameters:
               a   0.2009 (78)     [  0.00 (20) ]  

   Settings:
     svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 1/0.0)

   intercept = 0.2009(78)

The increase in the Bayes Factor, however, is not significant, and the 
result is almost unchanged. This confirms that the original choice was 
reasonable.

Another variation is to replace the simultaneous fit of the four models
by a chained fit, where one model is fit at a time and its
results are fed into the next fit through that fit's prior. Replacing the
fit code by ::

   fitter = lsqfit.MultiFitter(models=models)
   fit = fitter.chained_lsqfit(data=data, prior=prior)  
   # same as fit = fitter.lsqfit(data=data, prior=prior, chained=True)
   print(fit.formatall())
   print('intercept =', fit.p['a'])

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
                  a    0.206 (11)     [ 0.213 (16) ]
                 s2   0.4904 (64)     [  0.0 (1.0) ]

    Settings:
      svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

    ========== d3
    Least Square Fit:
      chi2/dof [dof] = 0.66 [4]    Q = 0.62    logGBF = 5.3767

    Parameters:
                  a    0.1995 (90)      [ 0.206 (11) ]
                 s3   -0.0840 (57)      [  0.0 (1.0) ]

    Settings:
      svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

    ========== d4
    Least Square Fit:
      chi2/dof [dof] = 0.41 [4]    Q = 0.81    logGBF = 5.9402

    Parameters:
                  a    0.2012 (78)      [ 0.1995 (90) ]
                 s4   -0.2001 (53)      [   0.0 (1.0) ]

    Settings:
      svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 4/0.0)

    intercept = 0.2012(78)

Note how the value for ``a`` improves with each fit.

Chained fits are most useful
with very large data sets when it is possible to break the data into
smaller, more manageable chunks. There are a variety of options for
organizing the chain of fits; these are discussed in the
:meth:`MultiFitter.chained_lsqfit` documentation.


.. autoclass:: lsqfit.MultiFitter(models, mopt=None, ratio=False, fast=True, **fitterargs)

   .. automethod:: MultiFitter.lsqfit

   .. automethod:: MultiFitter.chained_lsqfit

   .. automethod:: MultiFitter.empbayes_fit

   .. automethod:: MultiFitter.set

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


References
-----------
The :mod:`lsqfit` and :mod:`gvar` modules were originally created to facilitate statistical analyses of data generated by lattice QCD simulations. Background information about the techniques used in these modules can be found in several articles (on lattice QCD applications):

   * For a general discussion of Bayesian fitting (and Empirical Bayes) see: G.P. Lepage et al, Nucl.Phys.Proc.Suppl. 106 (2002) 12-20 [`hep-lat/0110175 <https://arxiv.org/pdf/hep-lat/0110175.pdf>`_].

   * For a discussion of the underlying analysis in a fit and the meaning of the error budget see Appendix A in: C. Bouchard et al, Phys.Rev. D90 (2014) 054506 [`arXiv:1406.2279 <https://arxiv.org/pdf/1406.2279.pdf>`_].

   * For a discussion of marginalization see the appendix in: C. McNeile et al, Phys.Rev. D82, 034512 (2010) [`arXiv:1004.4285 <https://arxiv.org/pdf/1004.4285.pdf>`_]. For another sample application see: K. Hornbostel et al, Phys.Rev. D85 (2012) 031504 [`arXiv:1111.1363 <https://arxiv.org/pdf/1111.1363.pdf>`_].

   * For a discussion of SVD cuts (and goodness-of-fit) see Appendix D in: R.J. Dowdall et al, Phys.Rev. D100 (2019) 9, 094508 [`arXiv:1907.01025 <https://arxiv.org/pdf/1907.01025.pdf>`_].




Requirements
------------
:mod:`lsqfit` relies heavily on the :mod:`gvar`, and :mod:`numpy` modules.
Also the fitting and minimization routines are from
the Gnu Scientific Library (GSL) and/or the Python :mod:`scipy` module.


