:mod:`lsqfit` - Nonlinear Least Squares Fitting
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. automodule:: lsqfit
   :synopsis: Nonlinear least squares fitting.


Formal Background
----------------------
The formal structure structure of a least-squares problem involves 
fitting input data :math:`y_i` with functions :math:`f_i(p)` by adjusting
fit parameters :math:`p_a` to minimize   

.. math::

   \chi^2 &\equiv \sum_{ij} \Delta y(p)_i\,(\mathrm{cov}_y^{-1})_{ij}\,
   \Delta y(p)_j  \\
   &\equiv \Delta y(p)\cdot \mathrm{cov}_y^{-1}\cdot
   \Delta y(p)
   
where :math:`\mathrm{cov}_y` is the covariance matrix for the input data
and

.. math::

    \Delta y(p)_i \equiv f_i(p) - y_i.

There are generally two types of input data --- actual data and
prior information for each fit parameter --- but we lump these together
here since they enter in the same way (that is, the sums over :math:`i`
and :math:`j` are over all data and priors).

The best-fit values :math:`\overline{p}_a` for the fit parameters are those
that minimize :math:`\chi^2`:

.. math::

   (\partial_a \Delta y(\overline{p}))^\mathrm{T}
   \cdot\mathrm{cov}_y^{-1}\cdot
   \Delta y(\overline{p}) = 0

where the derivatives are :math:`\partial_a = \partial/\partial
\overline{p}_a`. The covariance matrix :math:`\mathrm{cov}_p` for these is
obtained (approximately) from
    
.. math::
    
        (\mathrm{cov^{-1}_p})_{ab} \equiv 
    (\partial_a \Delta y(\overline p))^\mathrm{T}
    \cdot \mathrm{cov}^{-1}_y \cdot
    (\partial_b\Delta y(\overline p)).

Consequently the variance for any function :math:`g(\overline p)` of the
best-fit parameters is given by (approximately)

.. math::
   
   \sigma^2_{g} = \partial g(\overline p) \cdot 
   \mathrm{cov}_p \cdot \partial g(\overline p)

The definition of the covariance matrix implies that it and any variance
:math:`\sigma^2_g` derived from it depend linearly (approximately) on the
elements of the input data covariance matrix :math:`\mathrm{cov}_y`, at
least when errors are small: 

.. math::

   \sigma^2_g \approx \sum_{ij} c(\overline p)_{ij} \,
    (\mathrm{cov}_y)_{ij}

This allows us to associate different portions of the output error 
:math:`\sigma^2_g` with different parts of the input error
:math:`\mathrm{cov}_y`, creating an "error budget" for 
:math:`g(\overline p)`. 
Such information helps pinpoint the input errors that most affect the
output errors for any particular quantity  :math:`g(\overline p)`, 
and also indicates how those output errors might change for a given change
in input error.

The relationship between the input and output errors is only
approximately linear because the coefficients in the expansion depend upon
the best-fit values for the parameters, and these depend upon the input
errors --- but only weakly when errors are small. Neglecting such variation 
in the parameters, the error budget for any quantity is easily computed 
using

.. math::

   \frac{\partial (\mathrm{cov}_p)_{ab}}{\partial (\mathrm{cov}_y)_{ij}}
    = D_{ai}\,D_{bj}
   
where

.. math::

   D_{ai} \equiv (\mathrm{cov}_p \cdot \partial \Delta y \cdot 
      \mathrm{cov}_y^{-1})_{ai}

and, trivially,
:math:`\mathrm{cov}_p = D\cdot\mathrm{cov}_y\cdot D^\mathrm{T}`.

This last formula suggests that 

.. math::

   \frac{\partial p_a}{\partial y_i} = D_{ai}.
   
This relationship is true in the limit of small errors, as is easily derived
from the minimum condition for the fit: Differentiating with respect to
:math:`y_i` we obtain

.. math::

   (\partial_a \Delta y(p))^\mathrm{T}\cdot\mathrm{cov}_y^{-1}\cdot
   \frac{\partial\Delta y(p)}{\partial y_i} = 0

where we have ignored terms suppressed by a factor of :math:`\Delta y(p)`.
This leads immediately to the relationship above.

The data's covariance matrix :math:`\mathrm{cov}_y` is sometimes rather
singular, making it difficult to invert. This problem is dealt with using
an *svd* cut: the covariance matrix is diagonalized, some number of the
smallest (and therefore least-well determined) eigenvalues and their
eigenvectors are discarded, and the inverse matrix is reconstituted from
the eigenmodes that remain. (Instead of discarding modes one can replace
their eigenvalues by the smallest eigenvalue that is retained; this is less
conservative and sometimes leads to more accurate results.) Note that the
covariance matrix has at most :math:`N` non-zero eigenvalues when it is
estimated from :math:`N` random samples; zero-modes should always be
discarded.

nonlinear_fit Objects
---------------------  

.. autoclass:: lsqfit.nonlinear_fit

   The results from the fit are accessed through the following attributes
   (of ``fit`` where ``fit = nonlinear_fit(...)``):

   .. attribute:: chi2
   
      The ``chi**2`` for the last fit.
      
   .. attribute:: cov
   
      Covariance matrix from fit.

   .. attribute:: dof
   
      Number of degrees of freedom in fit.
      
   .. attribute:: logGBF
   
      Logarithm of Gaussian Bayes Factor for last fit (larger is better).
      The exponential of ``fit.logGBF`` is proportional to the
      Bayesian posterior probability of the fit in a linearized
      (that is, Gaussian) approximation. Larger values imply larger 
      posterior probabilities.
      
   .. attribute:: p
   
      Best-fit parameters from last fit. Depending upon what was used for the
      prior (or ``p0``), it is either: a dictionary (:class:`gvar.BufferDict`)
      of |GVar|\s and/or arrays of |GVar|\s; or an array
      (:class:`numpy.ndarray`) of |GVar|\s).
      
   .. attribute:: pmean
   
      Means of the best-fit parameters from last fit (dictionary or array).
      
   .. attribute:: psdev
   
      Standard deviations of the best-fit parameters from last fit (dictionary
      or array).
      
   .. attribute:: palt
   
      Same as ``fit.p`` except that the errors are computed directly 
      from ``fit.cov``. This is faster but means that no information about
      correlations with the input data is retained (unlike in ``fit.p``);
      and, therefore, ``fit.palt`` cannot be used to generate error
      budgets. ``fit.p`` and ``fit.palt`` give the same means and normally
      give the same errors for each parameter. They differ only when the
      input data's covariance matrix is too singular to invert accurately
      (because of roundoff error), in which case an *svd* cut is advisable.
      
   .. attribute:: p0
   
      The parameter values used to start the fit.
      
   .. attribute:: Q
   
      Quality factor for last fit (should be >0.1 for good fits).
      
   .. attribute:: svdcorrection
   
      The sum of all *svd* corrections to the input data and priors. Its only
      use is in creating error budgets: its contribution to a fit result's
      error is the contribution due to the *svd* cut. Equals ``[ ]`` if 
      there was no *svd* cut or it had no impact.
      
   .. attribute:: time
   
      CPU time (in secs) taken by last fit.
      
   The input parameters to the fit can be accessed as attributes. Note 
   in particular attributes:
   
   .. attribute:: prior
   
      Prior used in the fit. This may be differ from the input prior if an
      *svd* cut is used (``svdcut>0``). It is either a dictionary
      (:class:`gvar.BufferDict`) or an array (:class:`numpy.ndarray`),
      depending upon the input. Equals ``None`` if no prior was specified.
      
   .. attribute:: x
      
      The first field in the input ``data``. This is sometimes the independent
      variable (as in 'y vs x' plot), but may be anything.
      
   .. attribute:: y
   
      Fit data used in the fit. This may be differ from the input data if an
      *svd* cut is used (``svdcut>0``). It is either a dictionary
      (:class:`gvar.BufferDict`) or an array (:class:`numpy.ndarray`),
      depending upon the input.

   The main methods support doing the fits, printing out detailed
   information about the last fit, doing bootstrap analyses of the fit
   errors, checking the formatting of input data and the fit function,
   and checking for roundoff errors in the final error estimates:

   .. automethod:: format(maxline=0)
   
   .. automethod:: fmt_errorbudget(outputs,inputs,ndigit=2,percent=True)
   
   .. automethod:: fmt_values(outputs,ndigit=3)
      
   .. automethod:: bootstrap_iter
         
   .. automethod:: check_roundoff(rtol=0.25,atol=1e-6)


Functions
---------
.. autofunction:: lsqfit.empbayes_fit

.. autofunction:: lsqfit.wavg


Utility Classes
---------------
.. autoclass:: lsqfit.multifit(x0,n,f,reltol=1e-4,abstol=0,maxit=1000,alg='lmsder',analyzer=None)

.. autoclass:: lsqfit.multiminex(x0,f,tol=1e-4,maxit=1000,step=1,alg='nmsimplex2',analyzer=None)
   
Requirements
------------
:mod:`lsqfit` relies heavily on the :mod:`gvar`, and :mod:`numpy` modules.
Several utility functions are in :mod:`lsqfit_util`. Also the minimization
routines are from the Gnu Scientific Library (*GSL*).


