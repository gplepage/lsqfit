""" Introduction
----------------
This package contains tools for nonlinear least-squares curve fitting of
data. In general a fit has four inputs:

    1) The dependent data ``y`` that is to be fit --- typically ``y``
       is a Python dictionary in an :mod:`lsqfit` analysis. Its values
       ``y[k]`` are either |GVar|\s or arrays (any shape or dimension) of
       |GVar|\s that specify the values of the dependent variables
       and their errors.

    2) A collection ``x`` of independent data --- ``x`` can have any
       structure and contain any data, or it can be omitted.

    3) A fit function ``f(x, p)`` whose parameters ``p`` are adjusted by
       the fit until ``f(x, p)`` equals ``y`` to within ``y``\s errors
       --- parameters `p`` are usually specified by a dictionary whose
       values ``p[k]`` are individual parameters or (:mod:`numpy`)
       arrays of parameters. The fit function is assumed independent
       of ``x`` (that is, ``f(p)``) if ``x = False`` (or if ``x`` is
       omitted from the input data).

    4) Initial estimates or *priors* for each parameter in ``p``
       --- priors are usually specified using a dictionary ``prior``
       whose values ``prior[k]`` are |GVar|\s or arrays of |GVar|\s that
       give initial estimates (values and errors) for parameters ``p[k]``.

A typical code sequence has the structure::

    ... collect x, y, prior ...

    def f(x, p):
        ... compute fit to y[k], for all k in y, using x, p ...
        ... return dictionary containing the fit values for the y[k]s ...

    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
    print(fit)      # variable fit is of type nonlinear_fit

The parameters ``p[k]`` are varied until the ``chi**2`` for the fit is
minimized.

The best-fit values for the parameters are recovered after fitting
using, for example, ``p=fit.p``. Then the ``p[k]`` are |GVar|\s or
arrays of |GVar|\s that give best-fit estimates and fit uncertainties
in those estimates (as well as the correlations between them).
The ``print(fit)`` statement prints a summary of the fit results.

The dependent variable ``y`` above could be an array instead of a
dictionary, which is less flexible in general but possibly more
convenient in simpler fits. Then the approximate ``y`` returned by fit
function ``f(x, p)`` must be an array with the same shape as the dependent
variable. The prior ``prior`` could also be represented by an array
instead of a dictionary.

By default priors are Gaussian/normal distributions, represented by
|GVar|\s. :mod:`lsqfit` also
allows for log-normal and other distributions as well. The
latter are indicated by replacing the prior (in a dictionary prior)
with key ``c``,  for example, by a prior for the parameter's logarithm,
with key ``log(c)``.
:class:`nonlinear_fit` in effect adds parameter ``c`` to the parameter
dictionary, deriving its value from parameter ``log(c)``.
The fit function can be expressed directly in terms of
parameter ``c``  and so is the same no matter which distribution is
used for ``c``. Additional distributions
can be added using :meth:`gvar.BufferDict.add_distribution`.

The :mod:`lsqfit` tutorial contains extended explanations and examples.
The first appendix in the paper at http://arxiv.org/abs/arXiv:1406.2279
provides conceptual background on the techniques used in this
module for fits and, especially, error budgets.
"""

# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2021 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import collections
import functools
import sys
import warnings
import math, pickle, time, copy

import numpy

import gvar as _gvar

# default parameters for nonlinear_fit
_FITTER_DEFAULTS = dict(
    tol=1e-8,
    svdcut=None,
    eps=None,
    debug=False,
    maxit=1000,
    noise=(False, False)
    )

# dictionary containing all fitters available to nonlinear_fit.
_FITTERS = { }

try:
    from ._scipy import scipy_least_squares, scipy_multiminex, gammaQ
    _no_scipy = False
    _FITTERS['scipy_least_squares'] = scipy_least_squares
    _FITTER_DEFAULTS['fitter'] = 'scipy_least_squares'
except:
    _no_scipy = True

try:
    from ._gsl import gsl_multiminex, gammaQ
    from ._gsl import gsl_multifit, gsl_v1_multifit
    _no_gsl = False
    _FITTERS['gsl_multifit'] = gsl_multifit
    _FITTERS['gsl_v1_multifit'] = gsl_v1_multifit
    _FITTER_DEFAULTS['fitter'] = 'gsl_multifit'
except:
    _no_gsl = True

if _no_scipy and _no_gsl:
    raise RuntimeError('neither GSL nor scipy is installed --- need at least one')

_FDATA_attr = 'mean inv_wgts correction logdet nblocks svdn svdcut eps nw niw'.split()

if sys.version_info.major == 2:
    class _FDATA(object):
        __slots__ = _FDATA_attr
        def __init__(self, **kargs):
            for k in kargs:
                setattr(self, k, kargs[k])

        def __getstate__(self):
            return {k:getattr(self, k) for k in self.__slots__}
        
        def __setstate__(self, state):
            for k in state:
                setattr(self, k, state[k])

        def _remove_gvars(self, gvlist):
            newfdata = _FDATA(**{k:getattr(self, k) for k in self.__slots__})
            newfdata.correction = _gvar.remove_gvars(newfdata.correction, gvlist)
            return newfdata 

        def _distribute_gvars(self, gvlist):
            self.correction = _gvar.distribute_gvars(self.correction, gvlist)
else:
    class _FDATA(object):
        __slots__ = _FDATA_attr
        def __init__(self, **kargs):
            for k in kargs:
                setattr(self, k, kargs[k])

        def _remove_gvars(self, gvlist):
            newfdata = _FDATA(**{k:getattr(self, k) for k in self.__slots__})
            newfdata.correction = _gvar.remove_gvars(newfdata.correction, gvlist)
            return newfdata 

        def _distribute_gvars(self, gvlist):
            self.correction = _gvar.distribute_gvars(self.correction, gvlist)


# Internal data type for _unpack_data()

class nonlinear_fit(object):
    """ Nonlinear least-squares fit.

    :class:`lsqfit.nonlinear_fit` fits a (nonlinear) function ``f(x, p)``
    to data ``y`` by varying parameters ``p``, and stores the results: for
    example, ::

        fit = nonlinear_fit(data=(x, y), fcn=f, prior=prior)   # do fit
        print(fit)                               # print fit results

    The best-fit values for the parameters are in ``fit.p``, while the
    ``chi**2``, the number of degrees of freedom, the logarithm of Gaussian
    Bayes Factor, the number of iterations (or function evaluations),  and the
    cpu time needed for the fit are in ``fit.chi2``, ``fit.dof``,
    ``fit.logGBF``, ``fit.nit``, and ``fit.time``, respectively. Results for
    individual parameters in ``fit.p`` are of type |GVar|, and therefore carry
    information about errors and correlations with other parameters. The fit
    data and prior can be recovered using ``fit.x`` (equals ``False`` if there
    is no ``x``), ``fit.y``, and ``fit.prior``; the data and prior are
    corrected for the SVD cut, if there is one (that is, their covariance
    matrices have been modified in accordance with the SVD cut).

    Args:

        data (dict, array or tuple):
            Data to be fit by :class:`lsqfit.nonlinear_fit`
            can have any of the following forms:

                ``data = x, y``
                    ``x`` is the independent data that is passed to the fit
                    function with the fit parameters: ``fcn(x, p)``. ``y`` is a
                    dictionary (or array) of |GVar|\s that encode the means and
                    covariance matrix for the data that is to be fit being fit.
                    The fit function must return a result having the same
                    layout as ``y``.

                ``data = y``
                    ``y`` is a dictionary (or array) of |GVar|\s that encode
                    the means and covariance matrix for the data being fit.
                    There is no independent data so the fit function depends
                    only upon the fit parameters: ``fit(p)``. The fit function
                    must return a result having the same layout as ``y``.

                ``data = x, ymean, ycov``
                    ``x`` is the independent data that is passed to the fit
                    function with the fit parameters: ``fcn(x, p)``. ``ymean``
                    is an array containing the mean values of the fit data.
                    ``ycov`` is an array containing the covariance matrix of
                    the fit data; ``ycov.shape`` equals ``2*ymean.shape``.
                    The fit function must return an array having the same
                    shape as ``ymean``.

                ``data = x, ymean, ysdev``
                    ``x`` is the independent data that is passed to the fit
                    function with the fit parameters: ``fcn(x, p)``. ``ymean``
                    is an array containing the mean values of the fit data.
                    ``ysdev`` is an array containing the standard deviations of
                    the fit data; ``ysdev.shape`` equals ``ymean.shape``. The
                    data are assumed to be uncorrelated. The fit function must
                    return an array having the same shape as ``ymean``.

            Setting ``x=False`` in the first, third or fourth of these formats
            implies that the fit function depends only on the fit parameters:
            that is, ``fcn(p)`` instead of ``fcn(x, p)``. (This is not assumed
            if ``x=None``.)

        fcn (callable): The function to be fit to ``data``. It is either a
            function of the independent data ``x`` and the fit parameters ``p``
            (``fcn(x, p)``), or a function of just the fit parameters
            (``fcn(p)``) when there is no ``x`` data or ``x=False``. The
            parameters are tuned in the fit until the function returns values
            that agree with the ``y`` data to within the ``y``\s' errors. The
            function's return value must have the same layout as the ``y`` data
            (a dictionary or an array). The fit parameters ``p`` are either: 1)
            a dictionary where each ``p[k]`` is a single parameter or an array
            of parameters (any shape); or, 2) a single array of parameters. The
            layout of the parameters is the same as that of prior ``prior`` if
            it is specified; otherwise, it is inferred from of the starting
            value ``p0`` for the fit.

        prior (dict, array, str, gvar.GVar or None): A dictionary (or array)
            containing *a priori* estimates for all parameters ``p`` used by
            fit function ``fcn(x, p)`` (or ``fcn(p)``). Fit parameters ``p``
            are stored in a dictionary (or array) with the same keys and
            structure (or shape) as ``prior``. The default value is ``None``;
            ``prior`` must be defined if ``p0`` is ``None``.

        p0 (dict, array, float, None, or True): Starting values for fit
            parameters in fit. :class:`lsqfit.nonlinear_fit` adjusts ``p0`` to
            make it consistent in shape and structure with ``prior`` when the
            latter is specified: elements missing from ``p0`` are filled in
            using ``prior``, and elements in ``p0`` that are not in ``prior``
            are discarded. If ``p0`` is a string, it is taken as a file name
            and :class:`lsqfit.nonlinear_fit` attempts to read starting values
            from that file; best-fit parameter values are written out to the
            same file after the fit (for priming future fits). If ``p0`` is
            ``None`` or the attempt to read the file fails, starting values
            are extracted from ``prior``. If ``p0`` is ``True``, it
            is replaced by a starting point drawn at random from the
            ``prior`` distribution. The default value is ``None``;
            ``p0`` must be explicitly specified if ``prior`` is ``None``.

        linear (list or None): Optional list of fit parameters that appear
            linearly in the fit function. The fit function can be reexpressed
            (using *variable projection*) as a function that is independent of
            its linear parameters. The resulting fit has fewer fit parameters
            and typically will converge in fewer iterations, but each
            iteration will take longer. Whether or not the fit is faster or
            more robust in any particular application is a matter for
            experiment, but answers should be the same either way. The linear
            parameters are reconstructed from the nonlinear parameters (and
            the data) after the fit. Parameter ``linear`` is either: a list of
            dictionary keys corresponding to linear parameters when the
            parameters are stored in a dictionary (see ``prior``); or, a list
            of indices corresponding to these parameters when they are stored
            in an array. Note that this feature is experimental; the
            interface may change in the future.

        eps (float): If positive, singularities in the correlation matrix 
            for ``g`` are regulated using :func:`gvar.regulate` 
            with cutoff ``eps``. This makes the correlation matrices 
            less singular, which can improve the  stability and accuracy 
            of a fit. Ignored if ``svdcut`` is specified (and 
            not ``None``).

        svdcut (float): If nonzero, singularities in the correlation
            matrix are regulated using :func:`gvar.regulate`
            with an SVD cutoff ``svdcut``. This makes the correlation 
            matrices less singular, which can improve the  stability and 
            accuracy of a fit. Default is ``svdcut=1e-12``.

        noise (tuple or bool): If ``noise[0]=True``, noise is 
            added to the data and prior means corresponding to any 
            additional uncertainties introduced by using ``eps>0`` 
            or ``svdcut>0``. If ``noise[1]=True``, noise is added 
            to the prior means corresponding to the uncertainties
            in the prior. Noise is useful for testing the
            quality of a fit (``chi2``). Setting ``noise=True`` 
            is shorthand for ``noise=(True, True)``, and
            ``noise=False`` means ``noise=(False, False)`` (the default).
        
        udata (dict, array or tuple):
            Same as ``data`` but instructs the fitter to ignore  correlations
            between different pieces of data.  This speeds up the  fit,
            particularly for large amounts of data, but ignores potentially
            valuable information if the data actually are correlated. Only
            one of ``data`` or ``udata`` should be specified. (Default is
            ``None``.)

        fitter (str or None): Fitter code. Options if GSL is installed
            include: ``'gsl_multifit'`` (default) and ``'gsl_v1_multifit'``
            (original fitter). Options if :mod:`scipy` is installed include:
            ``'scipy_least_squares'`` (default if GSL not installed).
            ``gsl_multifit`` has many options, providing extensive user
            control. ``scipy_least_squares`` can be used for fits where the
            parameters are bounded. (Bounded parameters can also be
            implemented, for any of the fitters, using non-Gaussian priors ---
            see the tutorial.)

        tol (float or tuple): Assigning ``tol=(xtol, gtol, ftol)`` causes the
            fit to stop searching for a minimum when any of


                1. ``xtol >=`` relative change in parameters between iterations

                2. ``gtol >=`` relative size of gradient of ``chi**2`` function

                3. ``ftol >=`` relative change in ``chi**2`` between iterations

            is satisfied. See the fitter documentation for detailed
            definitions of these stopping conditions. Typically one sets
            ``xtol=1/10**d`` where ``d`` is the number of digits of precision
            desired in the result, while ``gtol<<1`` and ``ftol<<1``. Setting
            ``tol=delta`` where ``delta`` is a number is equivalent to setting
            ``tol=(delta,1e-10,1e-10)``. Setting ``tol=(delta1,delta2)`` is
            equivalent to setting ``tol=(delta1,delta2,1e-10)``. Default is
            ``tol=1e-8``. (Note: the ``ftol`` option is disabled in some
            versions of the GSL library.)

        maxit (int): Maximum number of algorithm iterations (or function
            evaluations for some fitters) in search for minimum;
            default is 1000.

        debug (bool): Set to ``True`` for extra debugging of the fit function
            and a check for roundoff errors. (Default is ``False``.)

        fitterargs (dict): Dictionary of additional arguments passed through
            to the underlying fitter. Different fitters offer different
            parameters; see the documentation for each.

    Objects of type :class:`lsqfit.nonlinear_fit` have the following
    attributes:

    Attributes:

        chi2 (float): The minimum ``chi**2`` for the fit.
            ``fit.chi2 / fit.dof`` is usually of order one in good fits.
            Values much less than one suggest that actual fluctuations in
            the input data and/or priors might be smaller than suggested
            by the standard deviations (or covariances) used in the fit.

        cov (array): Covariance matrix of the best-fit parameters from
            the fit.

        dof (int): Number of degrees of freedom in the fit, which equals
            the number of pieces of data being fit when priors are specified
            for the fit parameters. Without priors, it is the number of pieces
            of data minus the number of fit parameters.

        error (str): Error message generated by the underlying fitter when
            an error occurs. ``None`` otherwise.

        fitter_results: Results returned by the underlying fitter. Refer to
            the appropriate fitter's documentation for details.

        logGBF (float or None):The logarithm of the probability (density)
            of obtaining the fit data by randomly sampling the parameter model
            (priors plus fit function) used in the fit --- that is, it is
            ``P(data|model)``. This quantity is useful for comparing fits of
            the same data to different models, with different priors and/or
            fit functions. The model with the largest value of ``fit.logGBF``
            is the one preferred by the data. The exponential of the difference
            in ``fit.logGBF`` between two models is the ratio of probabilities
            (Bayes factor) for those models. Differences in ``fit.logGBF``
            smaller than 1 are not very significant. Gaussian statistics are
            assumed when computing ``fit.logGBF``.

        p (dict, array or gvar.GVar): Best-fit parameters from fit. Depending
            upon what was used for the prior (or ``p0``), it is either: a
            dictionary (:class:`gvar.BufferDict`) of |GVar|\s and/or arrays of
            |GVar|\s; or an array (:class:`numpy.ndarray`) of |GVar|\s.
            ``fit.p`` represents a multi-dimensional Gaussian distribution
            which, in Bayesian terminology, is the *posterior* probability
            distribution of the fit parameters.

        pmean (dict, array or float): Means of the best-fit parameters
            from fit.

        psdev (dict, array or float): Standard deviations of the best-fit
            parameters from fit.

        palt (dict, array or gvar.GVar): Same as ``fit.p`` except that the errors
            are computed directly from ``fit.cov``. This is faster but means
            that no information about correlations with the input data is
            retained (unlike in ``fit.p``); and, therefore, ``fit.palt``
            cannot be used to generate error budgets. ``fit.p`` and
            ``fit.palt`` give the same means and normally give the same errors
            for each parameter. They differ only when the input data's
            covariance matrix is too singular to invert accurately (because of
            roundoff error), in which case refitting with a nonzero value 
            for ``eps`` or ``svdcut`` is advisable.

        p0 (dict, array or float): The parameter values used to start the fit.
            This will differ from the input ``p0`` if the latter was
            incomplete.

        prior (dict, array, gvar.GVar or None): Prior used in the fit. This may
            differ  from the input prior if an SVD cut is used. It is either
            a  dictionary (:class:`gvar.BufferDict`) or an array
            (:class:`numpy.ndarray`), depending upon the input. Equals
            ``None`` if no prior was specified.

        Q (float or None): The probability that the ``chi**2`` from the fit
            could have been larger, by chance, assuming the best-fit model
            is correct. Good fits have ``Q`` values larger than 0.1 or so.
            Also called the *p-value* of the fit. The probabilistic
            intrepretation becomes unreliable if the actual fluctuations
            in the input data and/or priors are much smaller than suggested
            by the standard deviations (or covariances) used in the fit
            (leading to an unusually small ``chi**2``).

        residuals: An array containing the fit residuals normalized by the
            corresponding standard deviations. The residuals are projected
            onto the eigenvectors of the correlation matrix and so should 
            be uncorrelated from each other. The residuals include contributions
            from both the fit data and the prior. They are related to the 
            the ``chi**2`` of the fit by:
            ``chi2 = sum(fit.residuals**2)``.

        stopping_criterion (int): Criterion used to
            stop fit:

                0: didn't converge

                1: ``xtol >=`` relative change in parameters between iterations

                2: ``gtol >=`` relative size of gradient of ``chi**2``

                3: ``ftol >=`` relative change in ``chi**2`` between iterations

                4: unable to improve fit further (e.g., already converged)

        correction (gvar.GVar): Sum of all corrections, if any, added
            to the fit data and prior when ``eps>0`` or ``svdcut>0``.

        svdn (int): Number of eigenmodes of the correlation matrix 
            modified (and/or deleted) when ``svdcut>0``.

        time (float): *CPU* time (in secs) taken by fit.

        tol (tuple): Tolerance used in fit. This differs from the input
            tolerance if the latter was incompletely specified.

        x (obj): The first field in the input ``data``. This is sometimes the
            independent variable (as in 'y vs x' plot), but may be anything.
            It is set equal to ``False`` if the ``x`` field is omitted from
            the input ``data``. (This also means that the fit function has no
            ``x`` argument: so ``f(p)`` rather than ``f(x,p)``.)

        y (dict, array or gvar.GVar): Fit data used in the fit. This may differ
            from the input data if an SVD cut is used. It is either a
            dictionary (:class:`gvar.BufferDict`) or an array
            (:class:`numpy.ndarray`), depending upon the input.

        nblocks (dict): ``nblocks[s]`` equals the number of block-diagonal
            sub-matrices of the ``y``--``prior`` covariance matrix that are
            size ``s``-by-``s``. This is sometimes useful for debugging.

    The global defaults used by |nonlinear_fit| can be changed by
    changing entries in dictionary ``lsqfit.nonlinear_fit.DEFAULTS``
    for keys ``'eps'``, ``'svdcut'``, ``'debug'``, ``'tol'``, ``'noise'``,
    ``'maxit'``,  and ``'fitter'``. Additional defaults can be 
    added to that dictionary to be are passed through |nonlinear_fit| 
    to the underlying fitter (via dictionary ``fitterargs``).
    """
    # N.B. If _fdata is specified (set from a previous fit's
    # fit.fdata), then the data and prior correlation matrices are
    # from _fdata and not from the data themselves. The means in
    # _fdata are adjusted to be the same as in data and prior.
    # This was introduced to make simulated_fit_iter a little faster,
    # since now it doesn't have to rediagonalize the correlation matrix
    # before each fit. This is not part of the public interface and
    # could easily disappear in the future.

    DEFAULTS = {}

    FITTERS = _FITTERS

    def __init__(
        self, data=None, fcn=None, prior=None, p0=None, eps=False,
        svdcut=False, debug=None, tol=None, maxit=None, udata=None, _fdata=None,
        noise=None, 
        add_svdnoise=None, add_priornoise=None, # legacy names
        linear=[], fitter=None, **fitterargs
        ):

        # check arguments
        if data is None and udata is None:
            raise ValueError('neither data nor udata is specified')
        if fcn is None:
            raise ValueError('no fit function specified')
        if (p0 is None or p0 is True) and prior is None:
            raise ValueError('neither p0 nor prior is specified')

        # install defaults where needed
        if eps is False:
            eps = nonlinear_fit.DEFAULTS.get(
                'eps', _FITTER_DEFAULTS['eps']
                )
        if svdcut is False:
            svdcut = nonlinear_fit.DEFAULTS.get(
                'svdcut', _FITTER_DEFAULTS['svdcut'],
                )
        if debug is None:
            debug = nonlinear_fit.DEFAULTS.get(
                'debug', _FITTER_DEFAULTS['debug'],
                )
        if tol is None:
            tol = nonlinear_fit.DEFAULTS.get(
                'tol', _FITTER_DEFAULTS['tol'],
                )
        if maxit is None:
            maxit = nonlinear_fit.DEFAULTS.get(
                'maxit', _FITTER_DEFAULTS['maxit'],
                )

        noise_not_specified = noise is None
        if noise is None:
            noise = nonlinear_fit.DEFAULTS.get(
                'noise', _FITTER_DEFAULTS['noise'],
                )
        if isinstance(noise, bool):
            noise = (noise, noise)
        # legacy overwrites -- don't rely on these
        if add_svdnoise is not None and noise_not_specified:
            noise = (add_svdnoise, noise[1])
        if add_priornoise is not None and noise_not_specified:
            noise = (noise[0], add_priornoise)

        if fitter is None:
            fitter = nonlinear_fit.DEFAULTS.get(
                'fitter',  _FITTER_DEFAULTS['fitter'],
                )
        for k in nonlinear_fit.DEFAULTS:
            if k in [
                'eps', 'svdcut', 'debug', 'maxit', 'fitter', 'tol',
                'noise', 'add_svdnoise', 'add_priornoise',
                ]:
                continue
            if k not in fitterargs:
                fitterargs[k] = nonlinear_fit.DEFAULTS[k]

        # capture arguments; initialize parameters
        self.fitterargs = fitterargs
        self.noise = noise
        if data is None:
            self.data = udata
            self.uncorrelated_data = True
        else:
            self.data = data
            self.uncorrelated_data = False
        self.p0file = p0 if isinstance(p0, str) else None
        self.p0 = p0 if self.p0file is None else None
        self.fcn = fcn
        self._p = None
        self.debug = debug
        self.fitter = DEFAULT_FITTER if fitter is None else fitter
        if self.fitter not in nonlinear_fit.FITTERS:
            raise ValueError('unknown fitter: ' + str(self.fitter))

        clock = time.perf_counter if hasattr(time, 'perf_counter') else time.time
        cpu_time = clock()

        if noise[1] and prior is not None:
            prior = prior + (_gvar.sample(prior) - _gvar.mean(prior))

        # unpack prior,data,fcn,p0 to reconfigure for multifit
        if _fdata is None:
            x, y, prior, fdata = _unpack_data(
                data=self.data, prior=prior, svdcut=svdcut, eps=eps,
                uncorrelated_data=self.uncorrelated_data,
                noise=noise, debug=debug
                )
        else:
            x = data[0]
            y = data[1]
            fdata = copy.deepcopy(_fdata)
            fdata.mean[:y.size] = _gvar.mean(y.flat)
            if prior is not None:
                fdata.mean[y.size:] = _gvar.mean(prior.flat)
        self.eps = fdata.eps 
        self.svdcut = fdata.svdcut
        self.x = x
        self.y = y
        self.prior = prior
        self.fdata = fdata
        self.correction = fdata.correction
        self.svdcorrection = fdata.correction  # legacy name
        self.nblocks = fdata.nblocks
        self.svdn = fdata.svdn
        self.p0 = _unpack_p0(
            p0=self.p0, p0file=self.p0file, prior=self.prior,
            )
        p0 = self.p0.flatten()  # only need the buffer for multifit
        flatfcn = _unpack_fcn(
            fcn=self.fcn, p0=self.p0, y=self.y, x=self.x,
            )

        # create fit function chiv for multifit
        self._chiv, self._chivw = _build_chiv_chivw(
            fdata=self.fdata, fcn=flatfcn, prior=self.prior
            )
        nf = self.fdata.nw
        self.dof = nf - self.p0.size

        # check for linear variables
        if linear is not None:
            if self.p0.shape is None:
                self.linear = []
                bufsize = len(self.p0.buf)
                for k in linear:
                    if k not in self.p0:
                        raise ValueError('key {} not in prior'.format(k))
                    sl = self.p0.slice(k)
                    if isinstance(sl, slice):
                        self.linear += range(
                            sl.start, sl.stop,
                            sl.step if sl.step is not None else 1
                            )
                    else:
                        self.linear += [sl]
            elif len(linear) > 0:
                ptmp = numpy.zeros(self.p0.shape, dtype=bool)
                ptmp[linear] = True
                ptmp = ptmp.flatten()
                self.linear = numpy.arange(len(ptmp))[ptmp]
            else:
                self.linear = []
        else:
            self.linear = []

        # trial run if debugging
        if self.debug:
            if self.dof < 0:
                raise RuntimeError('fewer data values than parameters')
            if self.prior is None:
                p0gvar = numpy.array([p0i*_gvar.gvar(1, 1)
                                for p0i in p0.flat])
                nchivw = self.y.size
            else:
                p0gvar = self.prior.flatten() + p0
                nchivw = self.y.size + self.prior.size
            selfp0 = self.p0
            f = self.fcn(selfp0) if self.x is False else self.fcn(self.x, selfp0)
            if not _y_fcn_match(self.y, f):
                raise RuntimeError(_y_fcn_match.msg)
            for p in [p0, p0gvar]:
                f = flatfcn(p)
                if len(f)!=self.y.size:
                    raise RuntimeError(
                        "fcn(x, p) differs in size from y: %s, %s"
                                     % (len(f), y.size)
                        )
                if p is p0 and numpy.any(
                    [isinstance(f_i, _gvar.GVar) for f_i in f]
                    ):
                    raise RuntimeError(
                        "fcn(x, p) returns GVar's when p contains only numbers"
                        )
                v = self._chiv(p)
                if nf != len(v):
                    raise RuntimeError( #
                        "Internal error -- len(chiv): (%s, %s)" % (len(v), nf))
                vw = self._chivw(p)
                if nchivw != len(vw):
                    raise RuntimeError( #
                        "Internal error -- len(chivw): (%s, %s)"
                        %(len(vw), nchivw))

        if self.fitter == 'scipy_least_squares' and 'bounds' in self.fitterargs:
            lower, upper = self.fitterargs['bounds']
            if self.p0.shape is None:
                larray = []
                uarray = []
                for k in self.p0:
                    larray.extend(numpy.reshape(lower[k], -1))
                    uarray.extend(numpy.reshape(upper[k], -1))
                larray = numpy.array(larray)
                uarray = numpy.array(uarray)
            else:
                lower, upper = self.fitterargs['bounds']
                larray = numpy.reshape(lower, -1)
                uarray = numpy.reshape(upper, -1)
            self.fitterargs['bounds'] = (larray, uarray)

        # do the fit and save results
        if maxit > 0:
            if linear is not None and len(linear) > 0:
                fit = self._varpro_fit(p0=p0, nf=nf, tol=tol, maxit=maxit)
            else:
                fit = nonlinear_fit.FITTERS[self.fitter](
                    p0, nf, self._chiv, tol=tol, maxit=maxit, **self.fitterargs
                )
            self.error = fit.error
            self.cov = fit.cov
            self.chi2 = numpy.sum(fit.f**2)
            self.residuals = numpy.array(fit.f)
            self.Q = gammaQ(self.dof/2., self.chi2/2.)
            self.nit = fit.nit
            self.tol = fit.tol
            self.maxit = maxit
            self.stopping_criterion = fit.stopping_criterion
            self.description = getattr(fit, 'description', '')
            self.fitter_results = fit.results
            self._p = None          # lazy evaluation
            self.palt = _reformat(
                self.p0, _gvar.gvar(fit.x.flat, fit.cov),
                )
            self.pmean = _gvar.mean(self.palt)
            self.psdev = _gvar.sdev(self.palt)
        else:
            if self.prior is None:
                pmean = _gvar.mean(self.p0)
                psdev = _gvar.mean(self.p0)
                psdev.flat[:] = numpy.inf
            else:
                pmean = _gvar.mean(self.prior)
                psdev = _gvar.sdev(self.prior)
            self.palt = _gvar.gvar(pmean, psdev)
            self.pmean = _gvar.mean(self.palt)
            self.psdev = _gvar.sdev(self.palt)
            self.error = None
            self.cov = _gvar.evalcov(self.palt.flat)
            self.residuals = self._chiv(self.pmean.flat[:])
            self.chi2 = numpy.sum(self.residuals ** 2)
            self.Q = gammaQ(self.dof/2., self.chi2/2.)
            self.nit = 0
            self.tol = tol
            self.stopping_criterion = 0
            self.description = ''
            self.fitter_results = None
            self._p = self.palt

        # compute logGBF, etc
        self.dchi2 = _fit_dchi2(self)
        self.pdf = _fit_pdf(self)
        if self.prior is None:
            self.logGBF = None
        else:
            def logdet(m):
                (sign, ans) = numpy.linalg.slogdet(m)
                if sign < 0:
                    warnings.warn('det(fit.cov) < 0 --- roundoff errors? Try an svd cut.')
                return ans
                # return numpy.sum(numpy.log(numpy.linalg.svd(m, compute_uv=False)))
            logdet_cov = logdet(self.cov)
            self.logGBF = 0.5*(
                logdet_cov - self.fdata.logdet - self.chi2 -
                self.dof * numpy.log(2. * numpy.pi)
                )

        # archive final parameter values if requested
        if self.p0file is not None:
            with open(self.p0file, "wb") as ofile:
                pickle.dump(self.pmean, ofile)

        self.time = clock() - cpu_time
        if self.debug:
            self.check_roundoff()
        
        # legacy 
        self.svdcorrect = self.correction

    def _varpro_fit(self, p0, nf, tol, maxit):
        def lstsq(M, y):
            try:
                MTM = M.T.dot(M)
                MTy = M.T.dot(y)
                return _gvar.linalg.solve(MTM, MTy)
            except:
                return _gvar.linalg.lstsq(M, y)
        def M_y(p):
            " chiv = y - M @ p defines y and M "
            p[self.linear] *= 0.0
            y = self._chiv(p)
            M = []
            for i in self.linear:
                p[i] -= 1.0
                M.append(self._chiv(p) - y)
                p[i] += 1.0
            return numpy.transpose(M), y
        def fitfcn(p):
            M, y = M_y(p)
            p[self.linear] = lstsq(M, y)
            return self._chiv(p)
        localgvar = _gvar.gvar_factory()
        if len(self.linear) < len(p0) and maxit > 0:
            fit = nonlinear_fit.FITTERS[self.fitter](
                p0, nf, fitfcn, tol=tol, maxit=maxit, **self.fitterargs
                )
            p = localgvar(fit.x, fit.cov)
        else:
            class dummy:
                def __init__(self):
                    pass
            fit = dummy()
            fit.nit = 0
            fit.tol = (1e-8, 1e-10, 1e-10)
            fit.error = None
            fit.results = None
            fit.stopping_criterion = 0
            fit.description = 'no fit - all parameters linear'
            if self.prior is None:
                p = localgvar(p0, p0)
            else:
                p = localgvar(
                    _gvar.mean(self.prior.buf), _gvar.evalcov(self.prior.buf)
                    )
        M, y = M_y(p)
        y += localgvar(len(y) * ['0(1)'])
        p[self.linear] = lstsq(M, y)
        fit.x = _gvar.mean(p)
        fit.cov = _gvar.evalcov(p)
        fit.f = self._chiv(fit.x)
        return fit

    def _remove_gvars(self, gvlist):
        self.p  # need to fill _p
        fit = copy.copy(self)
        try:
            # if can pickle fcn then keep everything
            fit.pickled_fcn = _gvar.dumps((self.fcn, self._chiv, self._chivw, self.pdf, self.dchi2))
        except:
            if self.debug:
                warnings.warn('unable to pickle fit function; it is omitted')
            # fit.fcn = None 
            # fit._chiv = None 
            # fit._chivw = None
            # fit.pdf = None 
            # fit.dchi2 = None
        for k in ['_chiv', '_chivw', 'fcn', 'pdf', 'dchi2']:
            del fit.__dict__[k]
        fit.__dict__ = _gvar.remove_gvars(fit.__dict__, gvlist)
        return fit
    
    def _distribute_gvars(self, gvlist):
        self.__dict__ = _gvar.distribute_gvars(self.__dict__, gvlist)
        try:
            # try restoring fit function
            fcn, chiv, chivw, pdf, dchi2 = _gvar.loads(self.pickled_fcn)
            self.fcn = fcn
            self._chiv = chiv 
            self._chivw = chivw
            self.pdf = pdf 
            self.dchi2 = dchi2
            del self.__dict__['pickled_fcn']
        except:
            if self.debug:
                warnings.warn('unable to unpickle fit function; it is omitted')
        return self 

    @staticmethod
    def set(clear=False, **defaults):
        """ Set default parameters for :class:`lsqfit.nonlinear_fit`.

        Use to set default values for parameters: ``eps``, ``svdcut``,
        ``debug``, ``tol``, ``maxit``, and ``fitter``. Can also set
        parameters specific to the fitter specified by the ``fitter``
        argument.

        Sample usage::

            import lsqfit

            old_defaults = lsqfit.nonlinear_fit.set(
                fitter='gsl_multifit', alg='subspace2D', solver='cholesky',
                tol=1e-10, debug=True,
                )

        ``nonlinear_fit.set()`` without arguments returns a
        dictionary containing the current defaults.

        Args:
            clear (bool): If ``True`` remove earlier settings,
                restoring the original defaults, before adding new
                defaults. The default value is ``clear=False``.
                ``nonlinear_fit.set(clear=True)`` restores the
                original defaults.
            defaults (dict): Dictionary containing new defaults.

        Returns:
            A dictionary containing the old defaults,
            before they were updated. These can be restored using
            ``nonlinear_fit.set(old_defaults)`` where ``old_defaults``
            is the dictionary containint the old defaults.
        """
        old_defaults = dict(nonlinear_fit.DEFAULTS)
        if clear:
            nonlinear_fit.DEFAULTS = {}
        for k in defaults:
            if k == 'fitter' and defaults[k] not in nonlinear_fit.FITTERS:
                raise ValueError('unknown fitter: ' + str(defaults[k]))
            nonlinear_fit.DEFAULTS[k] = defaults[k]
        return old_defaults

    def __str__(self):
        return self.format()

    def check_roundoff(self, rtol=0.25, atol=1e-6):
        """ Check for roundoff errors in fit.p.

        Compares standard deviations from fit.p and fit.palt to see if they
        agree to within relative tolerance ``rtol`` and absolute tolerance
        ``atol``. Generates a warning if they do not (in which
        case an SVD cut might be advisable).
        """
        psdev = _gvar.sdev(self.p.flat)
        paltsdev = _gvar.sdev(self.palt.flat)
        if not numpy.allclose(psdev, paltsdev, rtol=rtol, atol=atol):
            warnings.warn("Possible roundoff errors in fit.p; try svd cut.")

    def _getp(self):
        """ Build :class:`gvar.GVar`\s for best-fit parameters. """
        if self._p is not None:
            return self._p
        # buf = [y,prior]; D[a,i] = dp[a]/dbuf[i]
        pmean = _gvar.mean(self.palt).flat
        buf = (
            self.y.flat[:] if self.prior is None else
            numpy.concatenate((self.y.flat, self.prior.flat))
            )
        D = numpy.zeros((self.cov.shape[0], len(buf)), float)
        for i, chivw_i in enumerate(self._chivw(_gvar.valder(pmean))):
            # for a in range(D.shape[0]):
                # D[a, i] = chivw_i.dotder(self.cov[a])
            D[:, i] = chivw_i.mdotder(self.cov)

        # p[a].mean=pmean[a]; p[a].der[j] = sum_i D[a,i]*buf[i].der[j]
        p = []
        for a in range(D.shape[0]): # der[a] = sum_i D[a,i]*buf[i].der
            p.append(
                _gvar.gvar(pmean[a], _gvar.wsum_der(D[a], buf), buf[0].cov)
                )
        self._p = _reformat(self.palt, p)
        return self._p

    p = property(_getp, doc="Best-fit parameters with correlations.")

    # transformed_p = property(_getp, doc="Same as fit.p --- for legacy code.")  # legacy name

    # fmt_partialsdev = _gvar.fmt_errorbudget  # this is for legacy code
    # fmt_errorbudget = _gvar.fmt_errorbudget
    # fmt_values = _gvar.fmt_values

    def evalchi2(self, p):
        """ Evaluate ``chi**2`` for arbitrary parameters ``p``.

            *Deprecated* Use ``fit.dchi2(p)`` instead.

            Args:
                p: Array or dictionary containing values for fit parameters,
                    using the same layout as in the fit function.

            Returns:
                ``chi**2`` for ``p``.
        """
        if hasattr(p, "keys"):
            p = _gvar.asbufferdict(p)
        else:
            p = numpy.asarray(p)
        return numpy.sum(self._chiv(p.flat[:]) ** 2)

    def logpdf(self, p, normalize=False):
        """ Logarithm of the fit's probability density function at ``p``.

        *Deprecated* Use ``fit.dchi2(p)`` instead.

        The fit's probability density function (PDF) is the product of the 
        Gaussian PDF for the data times the Gaussian PDF for the prior.
        It is proportional to ``exp(-fit.evalchi2(p))``.

        Args:
            p: Array or dictionary containing values for fit parameters,
                using the same layout as in the fit function.

            normalize (bool): If ``True`` the PDF is normalized; otherwise 
                the result is ``log(fit.pdf(p))`` which is unnormalized.
                (See discussion of ``fit.pdf``). Default is ``False``.

        Returns:
            ``-chi**2(p)/2 - log(norm)`` for ``p``.
    
        """
        if not hasattr(self, '_logpdfnorm'):
            self._logpdfnorm = 0.5 * (
                self.fdata.logdet
                + numpy.log(2*numpy.pi) * (self.dof + numpy.size(self.palt))
                )
        # return (self.dchi2(p) + self.chi2) / 2 + self._logpdfnorm
        return - self.evalchi2(p) / 2 - self._logpdfnorm

    def qqplot_residuals(self, plot=None):
        """ QQ plot normalized fit residuals.

        The sum of the squares of the residuals equals ``self.chi2``.
        Individual residuals should be distributed in a Gaussian
        distribution centered about zero. A Q-Q plot orders the 
        residuals and plots them against the value they would have if 
        they were distributed according to a Gaussian distribution.
        The resulting plot will approximate a straight line along
        the diagonal of the plot (dashed black line) if 
        the residuals have a Gaussian distribution with zero mean
        and unit standard deviation.

        The residuals are fit to a straight line and the fit
        is displayed in the plot (solid red line). Residuals that
        fall on a straight line have a distribution that is 
        Gaussian. A nonzero intercept indicates a bias in the mean, away from zero. 
        A slope smaller than 1.0 indicates the actual standard deviation 
        is smaller than suggested by the fit errors, as would be expected if 
        the ``chi2/dof`` is significantly below 1.0 (since ``chi2`` equals
        the sum of the squared residuals).

        One way to display the plot is with::

            fit.qqplot_residuals().show()

        Args:
            plot: a :mod:`matplotlib` plotter. If ``None``, 
                uses ``matplotlib.pyplot``.

        Returns:
            Plotter ``plot``.

        This method requires the :mod:`scipy` and :mod:`matplotlib` modules.
        """
        if _no_scipy:
            warnings.warn('scipy module not installed; needed for qqplot_residuals()')
            return
        if plot is None:
            import matplotlib.pyplot as plot
        from scipy import stats 
        (x, y), (s,y0,r) = stats.probplot(self.residuals, plot=plot, fit=True)
        minx = min(x)
        maxx = max(x)
        plot.plot([minx, maxx], [minx, maxx], 'k:')
        text = (r'residual = {:.2f} + {:.2f} $\times$ theory' '\nr = {:.2f}').format(y0, s, r)
        plot.title('Q-Q Plot')
        plot.ylabel('Ordered fit residuals')
        ylim = plot.ylim()
        plot.text(minx, ylim[0] + (ylim[1] - ylim[0]) * 0.9,text, color='r')
        return plot 

    def plot_residuals(self, plot=None):
        """ Plot normalized fit residuals.

        The sum of the squares of the residuals equals ``self.chi2``.
        Individual residuals should be distributed about one, in
        a Gaussian distribution.

        Args:
            plot: :mod:`matplotlib` plotter. If ``None``, 
                uses ``matplotlib.pyplot``.

        Returns:
            Plotter ``plot``.
        """
        if plot is None:
            import matplotlib.pyplot as plot
        x = numpy.arange(1, len(self.residuals) + 1)
        y = self.residuals
        plot.plot(x, y, 'bo')
        plot.ylabel('normalized residuals')
        xr = [x[0], x[-1]]
        plot.plot([x[0], x[-1]], [0, 0], 'r-')
        plot.fill_between(
            x=xr, y1=[-1,-1], y2=[1,1], color='r', alpha=0.075
            )
        return plot

    def format(self, maxline=0, pstyle='v', nline=None, extend=True):
        """ Formats fit output details into a string for printing.

        The output tabulates the ``chi**2`` per degree of freedom of the fit
        (``chi2/dof``), the number of degrees of freedom, the ``Q``  value of
        the fit (ie, p-value), and the logarithm of the Gaussian Bayes Factor
        for the fit (``logGBF``). At the end it lists the SVD cut, the number
        of eigenmodes modified by the SVD cut, the tolerances used in the fit,
        and the time in seconds needed to do the fit. The tolerance used to
        terminate the fit is marked with an asterisk. It also lists
        information about the fitter used if it is other than the standard
        choice.

        Optionally, ``format`` will also list the best-fit values
        for the fit parameters together with the prior for each (in ``[]`` on
        each line). Lines for parameters that deviate from their prior by more
        than one (prior) standard deviation are marked with asterisks, with
        the number of asterisks equal to the number of standard deviations (up
        to five). Lines for parameters designated as linear (see ``linear``
        keyword) are marked with a minus sign after their prior.

        ``format`` can also list all of the data and the corresponding values
        from the fit, again with asterisks on lines  where there is a
        significant discrepancy.

        Args:
            maxline (int or bool): Maximum number of data points for which
                fit results and input data are tabulated. ``maxline<0``
                implies that only ``chi2``, ``Q``, ``logGBF``, and ``itns``
                are tabulated; no parameter values are included. Setting
                ``maxline=True`` prints all data points; setting it
                equal ``None`` or ``False`` is the same as setting
                it equal to ``-1``. Default is ``maxline=0``.
            pstyle (str or None): Style used for parameter list. Supported
                values are 'vv' for very verbose, 'v' for verbose, and 'm' for
                minimal. When 'm' is set, only parameters whose values differ
                from their prior values are listed. Setting ``pstyle=None``
                implies no parameters are listed.
            extend (bool): If ``True``, extend the parameter list to
                include values derived from log-normal or other
                non-Gaussian parameters. So values for fit parameter
                ``p['log(a)']``, for example, are listed together with
                values ``p['a']`` for the exponential of the fit parameter.
                Setting ``extend=False`` means that only the value
                for ``p['log(a)']`` is listed. Default is ``True``.

        Returns:
            String containing detailed information about fit.
        """
        # unpack arguments
        if nline is not None and maxline == 0:
            maxline = nline         # for legacy code (old name)
        if maxline is True:
            # print all data
            maxline = sys.maxsize
        if maxline is False or maxline is None:
            maxline = -1
        if pstyle is not None:
            if pstyle[:2] == 'vv':
                pstyle = 'vv'
            elif pstyle[:1] == 'v':
                pstyle = 'v'
            elif pstyle[:1] == 'm':
                pstyle = 'm'
            else:
                raise ValueError("Invalid pstyle: "+str(pstyle))

        def collect(v1, v2, style='v', stride=1, extend=False):
            """ Collect data from v1 and v2 into table.

            Returns list of [label,v1fmt,v2fmt]s for each entry in v1 and
            v2. Here v1fmt and v2fmt are strings representing entries in v1
            and v2, while label is assembled from the key/index of the
            entry.
            """
            def nstar(v1, v2):
                sdev = max(v1.sdev, v2.sdev)
                if sdev == 0:
                    nstar = 5
                else:
                    try:
                        nstar = int(abs(v1.mean - v2.mean) / sdev)
                    except:
                        nstar = 5
                if nstar > 5:
                    nstar = 5
                elif nstar < 1:
                    nstar = 0
                return '  ' + nstar * '*'
            ct = 0
            ans = []
            width = [0,0,0]
            stars = []
            if v1.shape is None:
                # BufferDict
                keys = list(v1.keys())
                if extend:
                    v1 = _gvar.BufferDict(v1)
                    v2 = _gvar.BufferDict(v2)
                    ekeys = v1.extension_keys()
                    if len(ekeys) > 0:
                        first_ekey = ekeys[0]
                        keys += ekeys
                    else:
                        extend = False
                for k in keys:
                    if extend and k == first_ekey:
                        # marker indicating beginning of extra keys
                        stars.append(None)
                        ans.append(None)
                    ktag = str(k)
                    if numpy.shape(v1[k]) == ():
                        if ct%stride != 0:
                            ct += 1
                            continue
                        if style in ['v','m']:
                            v1fmt = v1[k].fmt(sep=' ')
                            v2fmt = v2[k].fmt(sep=' ')
                        else:
                            v1fmt = v1[k].fmt(-1)
                            v2fmt = v2[k].fmt(-1)
                        if style == 'm' and v1fmt == v2fmt:
                            ct += 1
                            continue
                        stars.append(nstar(v1[k], v2[k]))
                        ans.append([ktag, v1fmt, v2fmt])
                        w = [len(ai) for ai in ans[-1]]
                        for i, (wo, wn) in enumerate(zip(width, w)):
                            if wn > wo:
                                width[i] = wn
                        ct += 1
                    else:
                        ktag = ktag + " "
                        for i in numpy.ndindex(v1[k].shape):
                            if ct%stride != 0:
                                ct += 1
                                continue
                            ifmt = (len(i)*"%d,")[:-1] % i
                            if style in ['v','m']:
                                v1fmt = v1[k][i].fmt(sep=' ')
                                v2fmt = v2[k][i].fmt(sep=' ')
                            else:
                                v1fmt = v1[k][i].fmt(-1)
                                v2fmt = v2[k][i].fmt(-1)
                            if style == 'm' and v1fmt == v2fmt:
                                ct += 1
                                continue
                            stars.append(nstar(v1[k][i], v2[k][i]))
                            ans.append([ktag+ifmt, v1fmt, v2fmt])
                            w = [len(ai) for ai in ans[-1]]
                            for i, (wo, wn) in enumerate(zip(width, w)):
                                if wn > wo:
                                    width[i] = wn
                            ct += 1
                            ktag = ""
            else:
                # numpy array
                v2 = numpy.asarray(v2)
                for k in numpy.ndindex(v1.shape):
                    # convert array(GVar) to GVar
                    v1k = v1[k] if hasattr(v1[k], 'fmt') else v1[k].flat[0]
                    v2k = v2[k] if hasattr(v2[k], 'fmt') else v2[k].flat[0]
                    if ct%stride != 0:
                        ct += 1
                        continue
                    kfmt = (len(k) * "%d,")[:-1] % k
                    if style in ['v','m']:
                        v1fmt = v1k.fmt(sep=' ')
                        v2fmt = v2k.fmt(sep=' ')
                    else:
                        v1fmt = v1k.fmt(-1)
                        v2fmt = v2k.fmt(-1)
                    if style == 'm' and v1fmt == v2fmt:
                        ct += 1
                        continue
                    stars.append(nstar(v1k, v2k)) ###
                    ans.append([kfmt, v1fmt, v2fmt])
                    w = [len(ai) for ai in ans[-1]]
                    for i, (wo, wn) in enumerate(zip(width, w)):
                        if wn > wo:
                            width[i] = wn
                    ct += 1

            collect.width = width
            collect.stars = stars
            return ans

        # build header
        dof = self.dof
        if dof > 0:
            chi2_dof = self.chi2/self.dof
        else:
            chi2_dof = self.chi2
        try:
            Q = 'Q = %.2g' % self.Q
        except:
            Q = ''
        try:
            logGBF = 'logGBF = %.5g' % self.logGBF
        except:
            logGBF = ''
        if self.prior is None:
            descr = ' (no prior)'
        else:
            descr = ''
        table = ('Least Square Fit%s:\n  chi2/dof [dof] = %.2g [%d]    %s'
                 '    %s\n' % (descr, chi2_dof, dof, Q, logGBF))
        if maxline < 0:
            return table

        # create parameter table
        if pstyle is not None:
            table = table + '\nParameters:\n'
            prior = self.prior
            if prior is None:
                if self.p0.shape is None:
                    prior = _gvar.BufferDict(
                        self.p0, buf=self.p0.flatten() + _gvar.gvar(0,float('inf')))
                else:
                    prior = self.p0 + _gvar.gvar(0,float('inf'))
            data = collect(self.palt, prior, style=pstyle, stride=1, extend=extend)
            w1, w2, w3 = collect.width
            fst = "%%%ds%s%%%ds%s[ %%%ds ]" % (
                max(w1, 15), 3 * ' ',
                max(w2, 10), int(max(w2,10)/2) * ' ', max(w3,10)
                )
            if len(self.linear) > 0:
                spacer = [' ', '-']
            else:
                spacer = ['', '']
            for i, (di, stars) in enumerate(zip(data, collect.stars)):
                if di is None:
                    # marker for boundary between true fit parameters and derived parameters
                    ndashes = (
                        max(w1, 15) + 3 + max(w2, 10) + int(max(w2, 10)/2)
                        + 4 + max(w3, 10)
                        )
                    table += ndashes * '-' + '\n'
                    continue
                table += (
                    (fst % tuple(di)) +
                    spacer[i in self.linear] +
                    stars + '\n'
                    )

        # settings
        settings = "\nSettings:"
        if self.svdcut is not None:
            if not self.noise[0] or self.svdcut < 0:
                settings += "\n  svdcut/n = {svdcut:.2g}/{svdn}".format(
                    svdcut=self.svdcut, svdn=self.svdn
                    )
            else:
                settings += "\n  svdcut/n = {svdcut:.2g}/{svdn}*".format(
                        svdcut=self.svdcut, svdn=self.svdn
                        )
        else:
            if self.noise[0]:
                settings += "\n  eps = {eps:.2g}*".format(
                        eps=self.eps
                        )
            else:
                settings += "\n  eps = {eps:.2g}".format(
                        eps=self.eps
                        )
        criterion = self.stopping_criterion
        try:
            fmtstr = [
                "    tol = ({:.2g},{:.2g},{:.2g})",
                "    tol = ({:.2g}*,{:.2g},{:.2g})",
                "    tol = ({:.2g},{:.2g}*,{:.2g})",
                "    tol = ({:.2g},{:.2g},{:.2g}*)",
                "    tol = ({:.2g},{:.2g},{:.2g})",
            ][criterion if criterion is not None else 0]
            settings += fmtstr.format(*self.tol)
        except:
            pass
        if criterion is not None and criterion == 0:
            settings +="    (itns/time = {itns}*/{time:.1f})".format(
                itns=self.nit, time=self.time
                )
        else:
            settings +="    (itns/time = {itns}/{time:.1f})".format(
                itns=self.nit, time=self.time
                )
        default_line = '\n  fitter = gsl_multifit    methods = lm/more/qr\n'
        newline = "\n  fitter = {}    {}\n".format(
            self.fitter, self.description
            )
        if newline != default_line:
            settings += newline
        else:
            settings += '\n'
        if maxline <= 0 or self.data is None:
            return table + settings
        # create table comparing fit results to data
        ny = self.y.size
        stride = 1 if maxline >= ny else (int(ny/maxline) + 1)
        if hasattr(self, 'fcn_p'):
            f = self.fcn_p
        elif self.x is False:
            f = self.fcn(self.p)
        else:
            f = self.fcn(self.x, self.p)
        if hasattr(f, 'keys'):
            f = _gvar.BufferDict(f)
        else:
            f = numpy.array(f)
        data = collect(self.y, f, style='v', stride=stride, extend=False)
        w1,w2,w3 = collect.width
        clabels = ("key","y[key]","f(p)[key]")
        if self.y.shape is not None and self.x is not False and self.x is not None:
            # use x[k] to label lines in table?
            try:
                x = numpy.array(self.x)
                xlist = []
                ct = 0
                for k in numpy.ndindex(x.shape):
                    if ct%stride != 0:
                        ct += 1
                        continue
                    xlist.append("%g" % x[k])
                assert len(xlist) == len(data)
            except:
                xlist = None
            if xlist is not None:
                for i,(d1,d2,d3) in enumerate(data):
                    data[i] = (xlist[i],d2,d3)
                clabels = ("x[k]","y[k]","f(x[k],p)")

        w1,w2,w3 = max(9,w1+4), max(9,w2+4), max(9,w3+4)
        table += "\nFit:\n"
        fst = "%%%ds%%%ds%%%ds\n" % (w1, w2, w3)
        table += fst % clabels
        table += (w1 + w2 + w3) * "-" + "\n"
        for di, stars in zip(data, collect.stars):
            table += fst[:-1] % tuple(di) + stars + '\n'

        return table + settings

    @staticmethod
    def load_parameters(filename):
        """ Load parameters stored in file ``filename``.

        ``p = nonlinear_fit.load_p(filename)`` is used to recover the
        values of fit parameters dumped using ``fit.dump_p(filename)`` (or
        ``fit.dump_pmean(filename)``) where ``fit`` is of type
        :class:`lsqfit.nonlinear_fit`. The layout of the returned
        parameters ``p`` is the same as that of ``fit.p`` (or
        ``fit.pmean``).
        """
        warnings.warn(
            "nonlinear_fit.load_parameters deprecated; use pickle.load or gvar.load instead",
            DeprecationWarning,
            )
        with open(filename,"rb") as f:
            return pickle.load(f)

    def dump_p(self, filename):
        """ Dump parameter values (``fit.p``) into file ``filename``.

        ``fit.dump_p(filename)`` saves the best-fit parameter values
        (``fit.p``) from a ``nonlinear_fit`` called ``fit``. These values
        are recovered using
        ``p = nonlinear_fit.load_parameters(filename)``
        where ``p``'s layout is the same as that of ``fit.p``.
        """
        warnings.warn(
            "nonlinear_fit.dump_p deprecated; use gvar.dump instead",
            DeprecationWarning
            )
        with open(filename, "wb") as f:
            pickle.dump(self.palt, f) # dump as a dict

    def dump_pmean(self, filename):
        """ Dump parameter means (``fit.pmean``) into file ``filename``.

        ``fit.dump_pmean(filename)`` saves the means of the best-fit
        parameter values (``fit.pmean``) from a ``nonlinear_fit`` called
        ``fit``. These values are recovered using
        ``p0 = nonlinear_fit.load_parameters(filename)``
        where ``p0``'s layout is the same as ``fit.pmean``. The saved
        values can be used to initialize a later fit (``nonlinear_fit``
        parameter ``p0``).
        """
        warnings.warn(
            "nonlinear_fit.dump_pmean deprecated; use pickle.dump instead",
            DeprecationWarning,
            )
        with open(filename, "wb") as f:
            if self.pmean.shape is not None:
                pickle.dump(numpy.array(self.pmean), f)
            else:
                pickle.dump(collections.OrderedDict(self.pmean), f) # dump as a dict

    def simulated_fit_iter(
        self, n=None, pexact=None, add_priornoise=False, bootstrap=None, **kargs
        ):
        """ Iterator that returns simulation copies of a fit.

        Fit reliability is tested using simulated data which
        replaces the mean values in ``self.y`` with random numbers
        drawn from a distribution whose mean equals ``self.fcn(pexact)``
        and whose covariance matrix is the same as ``self.y``'s. Simulated
        data is very similar to the original fit data, ``self.y``,
        but corresponds to a world where the correct values for
        the parameters (*i.e.*, averaged over many simulated data
        sets) are given by ``pexact``. ``pexact`` is usually taken
        equal to ``fit.pmean``.

        Each iteration of the iterator creates new simulated data,
        with different random numbers, and fits it, returning the
        the :class:`lsqfit.nonlinear_fit` that results. The simulated
        data has the same covariance matrix as ``fit.y``.
        Typical usage is::

            ...
            fit = nonlinear_fit(...)
            ...
            for sfit in fit.simulated_fit_iter(n=3):
                ... verify that sfit has a good chi**2 ...
                ... verify that sfit.p agrees with pexact=fit.pmean within errors ...

        Only a few iterations are needed to get a sense of the fit's
        reliability since we know the correct answer in each case. The
        simulated fit's output results should agree with ``pexact``
        (``=fit.pmean`` here) within the simulated fit's errors.

        Setting parameter ``add_priornoise=True`` varies the means of the
        priors as well as the means of the data. This option is useful
        for testing goodness of fit because with it ``chi**2/N`` should
        be ``1 ± sqrt(2/N)``, where ``N`` is the
        number of degrees of freedom. (``chi**2/N`` can be significantly
        smaller than one without added noise in prior means.)

        Simulated fits can also be used to estimate biases in the fit's
        output parameters or functions of them, should non-Gaussian behavior
        arise. This is possible, again, because we know the correct value for
        every parameter before we do the fit. Again only a few iterations
        may be needed for reliable estimates.

        Args:
            n (int or ``None``): Maximum number of iterations (equals
                infinity if ``None``).
            pexact (``None`` or array/dict of numbers): Fit-parameter values
                for the underlying distribution used to generate simulated
                data; replaced by ``self.pmean`` if is ``None`` (default).
            add_priornoise (bool): Vary prior means if ``True``;
                otherwise vary only the means in ``self.y`` (default).
            kargs: Dictionary containing override values for fit parameters.

        Returns:
            An iterator that returns :class:`lsqfit.nonlinear_fit`\s
            for different simulated data.
        """
        pexact = self.pmean if pexact is None else pexact
        # bootstrap is old name for add_priornoise; keep for legacy code
        if bootstrap is not None:
            add_priornoise = bootstrap
        # Note: don't need svdcut since these are built into the data_iter
        fargs = dict(
            fcn=self.fcn, svdcut=0.0, eps=None, p0=pexact, fitter=self.fitter,
            )
        fargs.update(self.fitterargs)
        fargs.update(kargs)
        for ysim, priorsim in self.simulated_data_iter(
            n, pexact=pexact, add_priornoise=add_priornoise
            ):
            fit = nonlinear_fit(
                data=(self.x, ysim), prior=priorsim, _fdata=self.fdata,
                **fargs
                )
            fit.pexact = pexact
            yield fit

    def simulated_data_iter(
        self, n=None, pexact=None, add_priornoise=False, bootstrap=None
        ):
        """ Iterator that returns simulated data based upon a fit's data.

        Simulated data is generated from a fit's data ``fit.y`` by
        replacing the mean values in that data with random numbers
        drawn from a distribution whose mean is ``self.fcn(pexact)``
        and whose covariance matrix is the same as that of ``self.y``.
        Each iteration of the iterator returns new simulated data,
        with different random numbers for the means and a covariance
        matrix equal to that of ``self.y``. This iterator is used by
        ``self.simulated_fit_iter``.

        Typical usage::

            fit = nonlinear_fit(data=(x,y), prior=prior, fcn=fcn)
            ...
            for ysim, priorsim in fit.simulate_data_iter(n=10):
                fitsim = nonlinear_fit(data=(x, ysim), prior=priorsim, fcn=fcn)
                print(fitsim)
                print('chi2 =', gv.chi2(fit.p, fitsim.p))

        This code tests the fitting protocol on simulated data, comparing the
        best fit parameters in each case with the correct values (``fit.p``).
        The loop in this code is functionally the same as (but probably not
        as fast as)::

            for fitsim in fit.simulated_fit_iter(n=10):
                print(fitsim)
                print('chi2 =', gv.chi2(fit.p, fitsim.p))

        Args:
            n (int or None): Maximum number of iterations (equals
                infinity if ``None``).

            pexact (None or dict/array of numbers): Fit-parameter values for
                the underlying distribution used to generate simulated data;
                replaced by ``self.pmean`` if is ``None`` (default).

            add_priornoise (bool): Vary prior means if ``True``; otherwise
                vary only the means in ``self.y`` (default).

        Returns:
            An iterator that returns a 2-tuple containing simulated
            versions of self.y and self.prior: ``(ysim, priorsim)``.
        """
        pexact = self.pmean if pexact is None else pexact
        # bootstrap is old name for add_priornoise; keep for legacy code
        if bootstrap is not None:
            add_priornoise = bootstrap
        f = self.fcn(pexact) if self.x is False else self.fcn(self.x, pexact)
        y = copy.deepcopy(self.y)
        if isinstance(y, _gvar.BufferDict):
            # y,f dictionaries; fresh copy of y, reorder f
            tmp_f = _gvar.BufferDict([(k, f[k]) for k in y])
            y.buf += tmp_f.buf - _gvar.mean(y.buf)
        else:
            # y,f arrays; fresh copy of y
            y += numpy.asarray(f) - _gvar.mean(y)
        prior = copy.deepcopy(self.prior)
        if prior is None or not add_priornoise:
            yiter = _gvar.bootstrap_iter(y, n)
            for ysim in _gvar.bootstrap_iter(y, n):
                yield ysim, prior
        else:
            yp = numpy.empty(y.size + prior.size, object)
            yp[:y.size] = y.flat
            yp[y.size:] = prior.flat
            for ypsim in _gvar.bootstrap_iter(yp, n):
                y.flat = ypsim[:y.size]
                prior.flat = ypsim[y.size:]
                yield y, prior

    # legacy name
    simulate_iter = simulated_fit_iter

    def bootstrapped_fit_iter(self, n=None, datalist=None, **kargs):
        """ Iterator that returns bootstrap copies of a fit.

        A bootstrap analysis involves three steps: 1) make a large number
        of "bootstrap copies" of the original input data and prior that differ
        from each other by random amounts characteristic of the underlying
        randomness in the original data; 2) repeat the entire fit analysis
        for each bootstrap copy of the data, extracting fit results from
        each; and 3) use the variation of the fit results from bootstrap
        copy to bootstrap copy to determine an approximate probability
        distribution (possibly non-gaussian) for the fit parameters and/or
        functions of them: the results from each bootstrap fit are samples
        from that distribution.

        Bootstrap copies of the data for step 2 are provided in
        ``datalist``. If ``datalist`` is ``None``, they are generated
        instead from the means and covariance matrix of the fit data
        (assuming gaussian statistics). The maximum number of bootstrap
        copies considered is specified by ``n`` (``None`` implies no
        limit).

        Variations in the best-fit parameters (or functions of them)
        from bootstrap fit to bootstrap fit define the probability
        distributions for those quantities. For example, one could use the
        following code to analyze the distribution of function ``g(p)``
        of the fit parameters::

            fit = nonlinear_fit(...)

            ...

            glist = []
            for bsfit in fit.bootstrapped_fit_iter(
                n=100, datalist=datalist,
                ):
                glist.append(g(bsfit.pmean))

            ... analyze samples glist[i] from g(p) distribution ...

        This code generates ``n=100`` samples ``glist[i]`` from the
        probability distribution of ``g(p)``. If everything is Gaussian,
        the mean and standard deviation of ``glist[i]`` should agree
        with ``g(fit.p).mean`` and ``g(fit.p).sdev``.

        Args:
            n (int): Maximum number of iterations if ``n`` is not ``None``;
                otherwise there is no maximum.
            datalist (iter): Collection of bootstrap ``data`` sets for fitter.
            kargs (dict): Overrides arguments in original fit.

        Returns:
            Iterator that returns an |nonlinear_fit| object
            containing results from the fit to the next data set in
            ``datalist``.
        """
        fargs = dict(fitter=self.fitter, fcn=self.fcn)
        fargs.update(self.fitterargs)
        fargs['p0'] = self.pmean
        fargs['prior'] = self.prior
        for k in kargs:
            fargs[k] = kargs[k]
        prior = fargs['prior']
        del fargs['prior']
        if datalist is None:
            x = self.x
            y = self.y
            if prior is None:
                for yb in _gvar.bootstrap_iter(y, n):
                    fit = nonlinear_fit(data=(x, yb), prior=None, **fargs)
                    yield fit
            else:
                g = _gvar.BufferDict(y=y.flat, prior=prior.flat)
                for gb in _gvar.bootstrap_iter(g, n):
                    yb = _reformat(y, buf=gb['y'])
                    priorb = _reformat(prior, buf=gb['prior'])
                    fit = nonlinear_fit(data=(x, yb), prior=priorb, **fargs)
                    yield fit
        else:
            if prior is None:
                i = 0
                for datab in datalist:
                    i += 1
                    if n is not None and i > n:
                        break
                    fit = nonlinear_fit(data=datab, prior=None, **fargs)
                    yield fit
            else:
                piter = _gvar.bootstrap_iter(prior)
                i = 0
                for datab in datalist:
                    i += 1
                    if n is not None and i > n:
                        break
                    fit = nonlinear_fit(data=datab, prior=next(piter), **fargs)
                    yield fit

    # legacy name
    bootstrap_iter = bootstrapped_fit_iter # legacy

# implement as classes so pickling works
class _fit_dchi2(object):
    """``chi**2(p) - fit.chi2`` for fit parameters ``p``.

    **Paramters:**
        **p:** Array or dictionary containing values for fit parameters, using
            the same layout as in the fit function.

    **Returns:**
        ``chi**2(p) - fit.chi2`` where ``chi**2(p)`` is the fit's
        ``chi**2`` for fit parameters ``p`` and ``fit.chi2`` is the ``chi**2``
        value for the best fit.
    """
    def __init__(self, fit):
        self._chiv = fit._chiv 
        self.chi2 = fit.chi2

    def __call__(self, p):
        if hasattr(p, "keys"):
            p = _gvar.asbufferdict(p)
        else:
            p = numpy.asarray(p)
        return numpy.sum(self._chiv(p.flat[:]) ** 2) - self.chi2

class _fit_pdf(object):
    """ ``exp(-(chi**2(p) - fit.chi2)/2)`` for fit parameters ``p``.

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
    """
    def __init__(self, fit):
        self._chiv = fit._chiv 
        self.chi2 = fit.chi2
        self.lognorm = 0.5 * (
            fit.fdata.logdet
            + numpy.log(2*numpy.pi) * (fit.dof + numpy.size(fit.palt))
            ) + fit.chi2 / 2

    def __call__(self, p):
        if hasattr(p, "keys"):
            p = _gvar.asbufferdict(p)
        else:
            p = numpy.asarray(p)
        return numpy.exp(-(numpy.sum(self._chiv(p.flat[:]) ** 2) - self.chi2) / 2)

nonlinear_fit.set(**_FITTER_DEFAULTS)

#####################################################################3
# background methods used by nonlinear_fit:

def _reformat(p, buf):
    """ Apply format of ``p`` to data in 1-d array ``buf``. """
    if numpy.ndim(buf) != 1:
        raise ValueError("Buffer ``buf`` must be 1-d.")
    if hasattr(p, 'keys'):
        ans = _gvar.BufferDict(p)
        if ans.size != len(buf):
            raise ValueError(       #
                "p, buf size mismatch: %d, %d"%(ans.size, len(buf)))
        ans = _gvar.BufferDict(ans, buf=buf)
    else:
        if numpy.size(p) != len(buf):
            raise ValueError(       #
                "p, buf size mismatch: %d, %d"%(numpy.size(p), len(buf)))
        ans = numpy.array(buf).reshape(numpy.shape(p))
    return ans

def _unpack_data(data, prior, svdcut, eps, uncorrelated_data, noise, debug):
    """ Unpack data and prior into ``(x, y, prior, fdata)``.

    This routine unpacks ``data`` and ``prior`` into ``x, y, prior, fdata``
    where ``x`` is the independent data, ``y`` is the fit data,
    ``prior`` is the collection of priors for the fit, and ``fdata``
    contains the information about the data and prior needed for the
    fit function. Both ``y`` and ``prior`` are modified to account
    if ``svdcut>0`` or ``eps>0``.

    Allowed layouts for ``data`` are: ``x, y, ycov``, ``x, y, ysdev``,
    ``x, y``, and ``y``. In the last two case, ``y`` can be either an array
    of |GVar|\s or a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s. In the last case it is assumed that the fit function is a
    function of only the parameters: ``fcn(p)`` --- no ``x``. (This is also
    assumed if ``x = False``.)

    Output data in ``fdata`` is: ``fdata.mean`` containing the mean values of
    ``y.flat`` and ``prior.flat`` (if there is a prior);
    ``fdata.correction``  containing the sum of the SVD corrections to
    ``y.flat`` and ``prior.flat``; ``fdata.logdet`` containing  the logarithm
    of the  determinant of the covariance matrix of ``y.flat`` and
    ``prior.flat``; and ``fdata.inv_wgts`` containing a representation of the
    inverse of the covariance matrix, after SVD cuts (see :func:`gvar.svd`
    for a description of the format).
    """
    # unpack data tuple
    if not isinstance(data, tuple):
        x = False                   # no x in fit fcn
        y = _unpack_gvars(data)
    elif len(data) == 3:
        x, ym, ycov = data
        ym = numpy.asarray(ym)
        ycov = numpy.asarray(ycov)
        y = _gvar.gvar(ym, ycov)
    elif len(data) == 2:
        x, y = data
        y = _unpack_gvars(y)
    else:
        raise ValueError("data tuple wrong length: "+str(len(data)))
    if debug:
        if numpy.any(_gvar.sdev(y.flat) == 0):
            raise ValueError('some input data have zero standard deviations')
        if numpy.any(numpy.isnan(_gvar.mean(y.flat))):
            raise ValueError("some input data means are nan's")
        if numpy.any(numpy.isnan(_gvar.sdev(y.flat))):
            raise ValueError("some input data std devs are nan's")

    # clean up
    if prior is not None:
        prior = _unpack_gvars(prior)
        if debug:
            if numpy.any(_gvar.sdev(prior.flat) == 0):
                raise ValueError('some priors have zero standard deviations')
            if numpy.any(numpy.isnan(_gvar.mean(prior.flat))):
                raise ValueError("some prior means are nan's")
            if numpy.any(numpy.isnan(_gvar.sdev(prior.flat))):
                raise ValueError("some prior std devs are nan's")

    def _apply_svd(data, svdcut=svdcut, eps=eps):
        ans, inv_wgts = _gvar.regulate(
            data, wgts=-1, eps=eps, svdcut=svdcut, noise=noise[0],
            )
        fdata = _FDATA(
            mean=_gvar.mean(ans.flat),
            inv_wgts=inv_wgts,
            correction=numpy.sum(ans.correction.flat),
            logdet=ans.logdet,
            nblocks=ans.nblocks,
            svdn=ans.nmod,
            svdcut=ans.svdcut,
            eps=ans.eps,
            nw=sum(len(wgts) for iw, wgts in inv_wgts),
            niw=sum(len(iw) for iw, wgts in inv_wgts),
            )
        del ans.nblocks
        # del ans.correction
        return ans, fdata

    if uncorrelated_data:
        ysdev = _gvar.sdev(y.flat)
        if prior is None:
            pfdata = _FDATA(
                mean=[], inv_wgts=[([],[])], correction=_gvar.gvar(0,0),
                logdet=0.0, nblocks={1:0}, svdn=0, svdcut=0, eps=0, nw=0, niw=0,
                )
        else:
            prior, pfdata = _apply_svd(prior)
        inv_wgts = [(numpy.arange(y.size, dtype=numpy.intp), 1. / ysdev)]
        i, wgt = pfdata.inv_wgts[0]
        if len(i) > 0:
            inv_wgts = [(
                numpy.concatenate((inv_wgts[0][0], i + y.size)),
                numpy.concatenate((inv_wgts[0][1], wgt))
                )]
        for i, wgt in pfdata.inv_wgts[1:]:
            inv_wgts.append((
                i + y.size, wgt
                ))
        pfdata.nblocks[1] = pfdata.nblocks.get(1, 0) + y.size
        fdata = _FDATA(
            mean=numpy.concatenate((_gvar.mean(y.flat), pfdata.mean)),
            inv_wgts=inv_wgts,
            correction=pfdata.correction,
            logdet=2 * numpy.sum(numpy.log(ysdev)) + pfdata.logdet,
            nblocks=pfdata.nblocks,
            svdn=pfdata.svdn,
            svdcut=pfdata.svdcut,
            eps=pfdata.svdcut,
            nw=sum(len(wgts) for iw, wgts in inv_wgts),
            niw=sum(len(iw) for iw, wgts in inv_wgts),
            )
    elif prior is None:
        y, fdata = _apply_svd(y)
    else:
        yp, fdata = _apply_svd(numpy.concatenate((y.flat, prior.flat)))
        y.flat = yp[:y.size]
        prior.flat = yp[y.size:]
    return x, y, prior, fdata

def _unpack_gvars(g):
    """ Unpack collection of GVars to BufferDict or numpy array. """
    if g is not None:
        g = _gvar.gvar(g)
        if not hasattr(g, 'flat'):
            # must be a scalar (ie, not an array and not a dictionary)
            g = numpy.asarray(g)
    return g

def _unpack_p0(p0, p0file, prior):
    """ Create proper p0.

    Try to read from a file. If that doesn't work, try using p0,
    and then, finally, the prior. If the p0 is from the file, it is
    checked against the prior to make sure that all elements have the
    right shape; if not the p0 elements are adjusted (using info from
    the prior) to be the correct shape. If p0 is a dictionary,
    keys in p0 that are not in prior are discarded. If p0 is True,
    then a random p0 is generated from the prior.
    """
    if p0file is not None:
        # p0 is a filename; read in values
        try:
            with open(p0file, "rb") as f:
                p0 = pickle.load(f)
        except (IOError, EOFError):
            if prior is None:
                raise IOError(
                    "No prior and can't read parameters from " + p0file
                    )
            else:
                p0 = None
    if p0 is not None:
        # repackage as BufferDict or numpy array
        if p0 is True:
            p0 = next(_gvar.raniter(prior))
        if hasattr(p0, 'keys'):
            p0 = _gvar.BufferDict(p0)
            if p0.dtype != float:
                p0.buf = numpy.asarray(p0.buf, dtype=float)
        else:
            p0 = numpy.array(p0, float)
    if prior is not None:
        # build new p0 from p0, plus the prior as needed
        pp = _reformat(prior, buf=[x.mean if x.mean != 0.0
                        else x.mean + 0.1 * x.sdev for x in prior.flat])
        if p0 is None:
            p0 = pp
        else:
            if pp.shape is not None:
                # pp and p0 are arrays
                pp_shape = pp.shape
                p0_shape = p0.shape
                if len(pp_shape)!=len(p0_shape):
                    raise ValueError(       #
                        "p0 and prior shapes incompatible: %s, %s"
                        % (str(p0_shape), str(pp_shape)))
                idx = []
                for npp, np0 in zip(pp_shape, p0_shape):
                    idx.append(slice(0, min(npp, np0)))
                idx = tuple(idx)    # overlapping slices in each dir
                pp[idx] = p0[idx]
                p0 = pp
            else:
                # pp and p0 are dicts
                # adjust p0[k] to be compatible with shape of prior[k]
                for k in pp:
                    if k not in p0:
                        continue
                    pp_shape = numpy.shape(pp[k])
                    p0_shape = numpy.shape(p0[k])
                    if len(pp_shape)!=len(p0_shape):
                        raise ValueError("p0 and prior incompatible: "
                                         +str(k))
                    if pp_shape == p0_shape:
                        pp[k] = p0[k]
                    else:
                        # find overlap between p0 and pp
                        pp_shape = pp[k].shape
                        p0_shape = p0[k].shape
                        if len(pp_shape)!=len(p0_shape):
                            raise ValueError(       #
                                "p0 and prior incompatible: "+str(k))
                        idx = []
                        for npp, np0 in zip(pp_shape, p0_shape):
                            idx.append(slice(0, min(npp, np0)))
                        idx = tuple(idx)    # overlapping slices in each dir
                        pp[k][idx] = p0[k][idx]
                p0 = pp
    if p0 is None:
        raise ValueError("No starting values for parameters")
    return p0


def _unpack_fcn(fcn, p0, y, x):
    """ reconfigure fitting fcn so inputs, outputs = flat arrays; hide x """
    if y.shape is not None:
        if p0.shape is not None:
            nfcn = functools.partial(flatfcn_aa, x=x, fcn=fcn, pshape=p0.shape)
        else:
            po = _gvar.BufferDict(p0, buf=numpy.zeros(p0.size, float))
            nfcn = functools.partial(flatfcn_ad, x=x, fcn=fcn, po=po)
    else:
        yo = _gvar.BufferDict(y, buf=y.size*[None])
        if p0.shape is not None:
            nfcn = functools.partial(flatfcn_da, x=x, fcn=fcn, pshape=p0.shape, yo=yo)
        else:
            po = _gvar.BufferDict(p0, buf=numpy.zeros(p0.size, float))
            nfcn = functools.partial(flatfcn_dd, x=x, fcn=fcn, po=po, yo=yo)
    return nfcn

def flatfcn_aa(p, x, fcn, pshape):
    po = p.reshape(pshape)
    ans = fcn(po) if x is False else fcn(x, po)
    if hasattr(ans, 'flat'):
        return ans.flat
    else:
        return numpy.array(ans).flat

def flatfcn_ad(p, x, fcn, po):
    po.buf = p
    ans = fcn(po) if x is False else fcn(x, po)
    if hasattr(ans, 'flat'):
        return ans.flat
    else:
        return numpy.array(ans).flat

def flatfcn_da(p, x, fcn, pshape, yo):
    po = p.reshape(pshape)
    fxp = fcn(po) if x is False else fcn(x, po)
    for k in yo:
        yo[k] = fxp[k]
    return yo.flat

def flatfcn_dd(p, x, fcn, po, yo):
    po.buf = p
    fxp = fcn(po) if x is False else fcn(x, po)
    for k in yo:
        yo[k] = fxp[k]
    return yo.flat

def _y_fcn_match(y, f):
    if hasattr(f,'keys'):
        f = _gvar.BufferDict(f)
    else:
        f = numpy.array(f)
    if y.shape != f.shape:
        _y_fcn_match.msg = ("shape mismatch between y and fcn: "
                            + str(y.shape) + ", " + str(f.shape) )
        return False
    if y.shape is None:
        for k in y:
            if k not in f:
                _y_fcn_match.msg = "key mismatch: " + str(k)
                return False
            if numpy.shape(y[k]) == ():
                if numpy.shape(f[k]) != ():
                    _y_fcn_match.msg = "shape mismatch for key " + str(k)
                    return False
            elif y[k].shape != f[k].shape:
                _y_fcn_match.msg = "shape mismatch for key " + str(k)
                return False
        for k in f:
            if k not in y:
                _y_fcn_match.msg = "key mismatch: " + str(k)
                return False
    return True

# add extras and utilities to lsqfit
from ._extras import empbayes_fit, wavg
from ._extras import MultiFitterModel, MultiFitter
from ._utilities import _build_chiv_chivw
from ._version import __version__

# legacy definitions (obsolete)
class _legacy_constructor:
    def __init__(self, cl, msg):
        self.cl = cl
        self.msg = msg

    def __call__(self, *args, **kargs):
        warnings.warn(self.msg, UserWarning, stacklevel=2)
        return self.cl(*args,**kargs)

BufferDict = _gvar.BufferDict

CGPrior = _legacy_constructor(
    _gvar.BufferDict,"CGPrior is deprecated; use gvar.BufferDict instead.")
GPrior = _legacy_constructor(
    _gvar.BufferDict,"GPrior is deprecated; use gvar.BufferDict instead.")
LSQFit = _legacy_constructor(
    nonlinear_fit,"LSQFit is deprecated; use lsqfit.nonlinear_fit instead.")
