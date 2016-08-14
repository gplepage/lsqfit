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
       structure and contain any data (or no data).

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
in those estimates. The ``print(fit)`` statement prints a summary of
the fit results.

The dependent variable ``y`` above could be an array instead of a
dictionary, which is less flexible in general but possibly more
convenient in simpler fits. Then the approximate ``y`` returned by fit
function ``f(x, p)`` must be an array with the same shape as the dependent
variable. The prior ``prior`` could also be represented by an array
instead of a dictionary.

By default priors are Gaussian/normal distributions, represented by  |GVar|\s.
Setting :class:`nonlinear_fit` parameter ``extend=True``  allows for log-normal
and sqrt-normal distributions as well. The latter are indicated by replacing
the prior (in a dictionary prior) with key ``c``,  for example, by a prior for
the parameter's logarithm or square root, with key ``log(c)`` or ``sqrt(c)``,
respectively.  :class:`nonlinear_fit` adds parameter ``c`` to the parameter
dictionary, deriving its value from parameter ``log(c)`` or ``sqrt(c)``. The
fit function can be expressed directly in terms of parameter ``c``  and so is
the same no matter which distribution is used for ``c``. Note that a
sqrt-normal distribution with zero mean is equivalent to an exponential
distribution. Additional distributions can be added using
:meth:`gvar.add_parameter_distribution`.

The :mod:`lsqfit` tutorial contains extended explanations and examples.
The first appendix in the paper at http://arxiv.org/abs/arXiv:1406.2279
provides conceptual background on the techniques used in this
module for fits and, especially, error budgets.
"""

# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2016 G. Peter Lepage.
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
import inspect
import re
import sys
import warnings
import math, pickle, time, copy

import numpy

import gvar as _gvar

# add extras and utilities to lsqfit
from ._extras import empbayes_fit, wavg
from ._utilities import dot as _util_dot
from ._utilities import _build_chiv_chivw
from ._utilities import multifit
from ._utilities import multiminex, gammaQ
from ._version import version as __version__


_FDATA = collections.namedtuple(
    '_FDATA', 'mean inv_wgts svdcorrection logdet nblocks svdn'
    )
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
    Bayes Factor, the number of iterations, and the cpu time needed for the
    fit are in ``fit.chi2``, ``fit.dof``, ``fit.logGBF``, ``fit.nit``, and
    ``fit.time``, respectively. Results for individual parameters in
    ``fit.p`` are of type |GVar|, and therefore carry information about
    errors and correlations with other parameters. The fit data and prior
    can be recovered using ``fit.x`` (equals ``False`` if there is no ``x``),
    ``fit.y``, and ``fit.prior``; the data and prior are corrected for the
    *svd* cut, if there is one (that is, their covariance matrices have been
    modified in accordance with the *svd* cut).

    :param data: Data to be fit by :class:`lsqfit.nonlinear_fit`. It can
        have any of the following formats:

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
    :param fcn: The function to be fit to ``data``. It is either a
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
    :type fcn: function
    :param prior: A dictionary (or array) containing *a priori* estimates
        for all parameters ``p`` used by fit function ``fcn(x, p)`` (or
        ``fcn(p)``). Fit parameters ``p`` are stored in a dictionary (or
        array) with the same keys and structure (or shape) as ``prior``.
        The default value is ``None``; ``prior`` must be defined if ``p0``
        is ``None``.
    :type prior: dictionary, array, or ``None``
    :param p0: Starting values for fit parameters in fit.
        :class:`lsqfit.nonlinear_fit` adjusts ``p0`` to make it consistent
        in shape and structure with ``prior`` when the latter is
        specified: elements missing from ``p0`` are filled
        in using ``prior``, and elements in
        ``p0`` that are not in ``prior`` are discarded. If ``p0`` is a
        string, it is taken as a file name and
        :class:`lsqfit.nonlinear_fit` attempts to read starting values from
        that file; best-fit parameter values are written out to the same
        file after the fit (for priming future fits). If ``p0`` is ``None``
        or the attempt to read the file fails, starting values are
        extracted from ``prior``. The default value is ``None``; ``p0``
        must be defined if ``prior`` is ``None``.
    :type p0: dictionary, array, string or ``None``
    :param svdcut: If ``svdcut`` is nonzero (not ``None``), *svd* cuts
        are applied to every block-diagonal sub-matrix of the covariance
        matrix for the data ``y`` and ``prior`` (if there is a prior).
        The blocks are first rescaled so that all
        diagonal elements equal 1 -- that is, the blocks are replaced
        by the correlation matrices for the corresponding subsets of variables.
        Then, if ``svdcut > 0``,
        eigenvalues of the rescaled matrices that are smaller than ``svdcut``
        times the maximum eigenvalue are replaced by ``svdcut`` times the
        maximum eigenvalue. This makes the covariance matrix less singular
        and less susceptible to roundoff error. When ``svdcut < 0``,
        eigenvalues smaller than ``|svdcut|`` times the maximum eigenvalue
        are discarded and the corresponding components in ``y`` and
        ``prior`` are zeroed out.
    :type svdcut: ``None`` or ``float``
    :param extend: Log-normal and sqrt-normal distributions can be used
        for fit priors when ``extend=True``, provided the parameters are
        specified by a dictionary (as opposed to an array). To use such a
        distribution  for a parameter ``'c'`` in the fit prior, replace
        ``prior['c']``  with a prior specifying its logarithm or  square
        root, designated by ``prior['log(c)']``  or ``prior['sqrt(c)']``,
        respectively. The dictionaries containing parameters generated by
        |nonlinear_fit| will have entries for both ``'c'``   and ``'log(c)'``
        or ``'sqrt(c)'``, so only the prior need be changed to  switch to
        log-normal/sqrt-normal distributions. Setting ``extend=False`` (the
        default) restricts all parameters to Gaussian distributions.
        Additional distributions can be added using
        :meth:`gvar.add_parameter_distribution`.

    :param debug: Set to ``True`` for extra debugging of the fit function
        and a check for roundoff errors. (Default is ``False``.)
    :type debug: boolean
    :param fitterargs: Dictionary of arguments passed on to
        :class:`lsqfit.multifit`, which does the fitting.
    """
    def __init__(
        self, data, fcn, prior=None, p0=None, extend=False,
        svdcut=1e-15, debug=False, **kargs
        ):
        # capture arguments; initialize parameters
        self.fitterargs = kargs
        if isinstance(svdcut, tuple):
            svdcut = svdcut[0]
            warnings.warn(
                'svdcut = tuple is no longer supported; replace by a single number',
                UserWarning, stacklevel=2
                )
        self.svdcut = svdcut
        self.data = data
        self.p0file = p0 if isinstance(p0, str) else None
        self.p0 = p0 if self.p0file is None else None
        self.fcn = fcn
        self._p = None
        self._palt = None
        self.debug = debug
        if 'svdnum' in kargs:
            del kargs['svdnum']
            warnings.warn(
                'svdnum is no longer supported by lsqfit',
                UserWarning, stacklevel=2
                )
        cpu_time = time.clock()

        # unpack prior,data,fcn,p0 to reconfigure for multifit
        x, y, prior, fdata = _unpack_data(
            data=self.data, prior=prior, svdcut=self.svdcut, extend=extend
            )
        self.x = x
        self.y = y
        self.prior = prior
        self.svdcorrection = fdata.svdcorrection
        self.nblocks = fdata.nblocks
        self.svdn = fdata.svdn
        self.p0 = _unpack_p0(
            p0=self.p0, p0file=self.p0file, prior=self.prior, extend=extend
            )
        self.extend = extend if self.p0.shape is None else False
        p0 = self.p0.flatten()  # only need the buffer for multifit
        flatfcn = _unpack_fcn(
            fcn=self.fcn, p0=self.p0, y=self.y, x=self.x,
            extend=self.extend
            )

        # create fit function chiv for multifit
        self._chiv, self._chivw = _build_chiv_chivw(
            fdata=fdata, fcn=flatfcn, prior=self.prior
            )
        self.dof = self._chiv.nf - self.p0.size
        nf = self._chiv.nf

        # trial run if debugging
        if self.debug:
            if self.prior is None:
                p0gvar = numpy.array([p0i*_gvar.gvar(1, 1)
                                for p0i in p0.flat])
                nchivw = self.y.size
            else:
                p0gvar = self.prior.flatten() + p0
                nchivw = self.y.size + self.prior.size
            selfp0 = self.p0
            if self.extend:
                selfp0 = _gvar.ExtendedDict(selfp0)
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

        # do the fit and save results
        # if self.fitterargs.get('alg', None) == 'scipy':
        #     multifit = LSQFitter
        fit = multifit(p0, nf, self._chiv, **self.fitterargs)
        self.error = fit.error
        self.cov = fit.cov
        self.chi2 = numpy.sum(fit.f**2)
        self.Q = gammaQ(self.dof/2., self.chi2/2.)
        self.nit = fit.nit
        self.tol = fit.tol
        self.stopping_criterion = fit.stopping_criterion
        self.alg = fit.alg
        self._p = None          # lazy evaluation
        try:
            self._trimmed_palt = _reformat(
                self.p0, _gvar.gvar(fit.x.flat, fit.cov), extend=False
                )
        except:
            print('xxx', fit.x , fit.error)
            print('cov', fit.cov)
            raise ValueError('ugh')
        self.palt = _reformat(
            self.p0, _gvar.gvar(fit.x.flat, fit.cov), extend=self.extend
            )
        self.psdev = _gvar.sdev(self.palt)
        self.pmean = _gvar.mean(self.palt)
        # compute logGBF
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
                logdet_cov - fdata.logdet - self.chi2 -
                self.dof * numpy.log(2. * numpy.pi)
                )

        # archive final parameter values if requested
        if self.p0file is not None:
            self.dump_pmean(self.p0file)

        self.time = time.clock()-cpu_time
        if self.debug:
            self.check_roundoff()

    def __str__(self):
        return self.format()

    def check_roundoff(self, rtol=0.25, atol=1e-6):
        """ Check for roundoff errors in fit.p.

        Compares standard deviations from fit.p and fit.palt to see if they
        agree to within relative tolerance ``rtol`` and absolute tolerance
        ``atol``. Generates a warning if they do not (in which
        case an *svd* cut might be advisable).
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
        pmean = _gvar.mean(self._trimmed_palt).flat
        buf = (
            self.y.flat if self.prior is None else
            numpy.concatenate((self.y.flat, self.prior.flat))
            )
        D = numpy.zeros((self.cov.shape[0], len(buf)), float)
        for i, chivw_i in enumerate(self._chivw(_gvar.valder(pmean))):
            for a in range(D.shape[0]):
                D[a, i] = chivw_i.dotder(self.cov[a])

        # p[a].mean=pmean[a]; p[a].der[j] = sum_i D[a,i]*buf[i].der[j]
        p = []
        for a in range(D.shape[0]): # der[a] = sum_i D[a,i]*buf[i].der
            p.append(_gvar.gvar(pmean[a], _gvar.wsum_der(D[a], buf),
                     buf[0].cov))
        self._p = _reformat(self.p0, p, extend=self.extend)
        return self._p

    p = property(_getp, doc="Best-fit parameters with correlations.")

    # transformed_p = property(_getp, doc="Same as fit.p --- for legacy code.")  # legacy name

    fmt_partialsdev = _gvar.fmt_errorbudget  # this is for legacy code
    fmt_errorbudget = _gvar.fmt_errorbudget
    fmt_values = _gvar.fmt_values

    def format(self, maxline=0, pstyle='v', nline=None):
        """ Formats fit output details into a string for printing.

        The output tabulates the ``chi**2`` per degree of freedom of the fit
        (``chi2/dof``), the number of degrees of freedom, the logarithm of the
        Gaussian Bayes Factor for the fit (``logGBF``), and the number of fit-
        algorithm iterations needed by the fit. Optionally, it will also list
        the best-fit values for the fit parameters together with the prior for
        each (in ``[]`` on each line). Lines for parameters that deviate from
        their prior by more than one (prior) standard deviation are marked
        with asterisks, with the number of asterisks equal to the number of
        standard deviations (up to five). ``format`` can also list all of the
        data and the corresponding values from the fit, again with asterisks
        on lines  where there is a significant discrepancy. At the end it
        lists the SVD cut, the number of eigenmodes modified by the SVD cut,
        the tolerances used in the fit, and the time in seconds needed to do
        the fit. The tolerance used to terminate the fit is marked with an
        asterisk.

        :param maxline: Maximum number of data points for which fit
            results and input data are tabulated. ``maxline<0`` implies
            that only ``chi2``, ``Q``, ``logGBF``, and ``itns`` are
            tabulated; no parameter values are included. Setting
            ``maxline=True`` prints all data points; setting it
            equal ``None`` or ``False`` is the same as setting
            it equal to ``-1``. Default is ``maxline=0``.
        :type maxline: integer or bool
        :param pstyle: Style used for parameter list. Supported values are
            'vv' for very verbose, 'v' for verbose, and 'm' for minimal.
            When 'm' is set, only parameters whose values differ from their
            prior values are listed.
        :type pstyle: 'vv', 'v', or 'm'
        :returns: String containing detailed information about fit.
        """
        # unpack arguments
        if nline is not None and maxline == 0:
            maxline = nline         # for legacy code (old name)
        if maxline is True:
            # print all data
            maxline = sys.maxsize
        if maxline is False or maxline is None:
            maxline = -1
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
                nstar = int(abs(v1.mean - v2.mean) / sdev)
                if nstar > 5:
                    nstar = 5
                elif nstar < 1:
                    nstar = 0
                return '  ' + nstar * '*'
            if extend:
                v2 = _gvar.ExtendedDict(v2)
                newkeys = v2.newkeys()
                try:
                    first_newkey = next(newkeys)
                except StopIteration:
                    extend = False
            ct = 0
            ans = []
            width = [0,0,0]
            stars = []
            if v1.shape is None:
                # BufferDict
                for k in v1:
                    if extend and k == first_newkey:
                        # marker indicating beginning of extra keys
                        stars.append(None)
                        ans.append(None)
                    ktag = str(k)
                    if v1.isscalar(k):
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
                    if ct%stride != 0:
                        ct += 1
                        continue
                    kfmt = (len(k)*"%d,")[:-1] % k
                    if style in ['v','m']:
                        v1fmt = v1[k].fmt(sep=' ')
                        v2fmt = v2[k].fmt(sep=' ')
                    else:
                        v1fmt = v1[k].fmt(-1)
                        v2fmt = v2[k].fmt(-1)
                    if style == 'm' and v1fmt == v2fmt:
                        ct += 1
                        continue
                    stars.append(nstar(v1[k], v2[k])) ###
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
            Q = '%.2g' % self.Q
        except:
            Q = '?'
        try:
            logGBF = '%.5g' % self.logGBF
        except:
            logGBF = str(self.logGBF)
        if self.prior is None:
            descr = " (no prior)"
        else:
            descr = ""
        table = ('Least Square Fit%s:\n  chi2/dof [dof] = %.2g [%d]    Q = %s'
                 '    logGBF = %s\n' % (descr, chi2_dof, dof, Q, logGBF))
        if maxline < 0:
            return table

        # create parameter table
        table = table + '\nParameters:\n'
        prior = self.prior
        if prior is None:
            if self.p0.shape is None:
                prior = _gvar.BufferDict(
                    self.p0, buf=self.p0.flatten() + _gvar.gvar(0,float('inf')))
            else:
                prior = self.p0 + _gvar.gvar(0,float('inf'))
        data = collect(self.palt, prior, style=pstyle, stride=1, extend=self.extend)
        w1, w2, w3 = collect.width
        fst = "%%%ds%s%%%ds%s[ %%%ds ]" % (
            max(w1, 15), 3 * ' ',
            max(w2, 10), int(max(w2,10)/2) * ' ', max(w3,10)
            )
        for di, stars in zip(data, collect.stars):
            if di is None:
                # marker for boundary between true fit parameters and derived parameters
                ndashes = (
                    max(w1, 15) + 3 + max(w2, 10) + int(max(w2, 10)/2)
                    + 4 + max(w3, 10)
                    )
                table += ndashes * '-' + '\n'
                continue
            table += (fst % tuple(di)) + stars + '\n'

        settings = "\nSettings:\n  svdcut/n = {svdcut}/{svdn}".format(
            svdcut=self.svdcut, svdn=self.svdn
            )
        if len(self.tol) == 2:
            settings += "    reltol/abstol = {:.2g}/{:.2g}".format(*self.tol)
            if self.stopping_criterion == 1:
                settings += '*'
        else:
            fmtstr = [
                "    tol = ({:.2g},{:.2g},{:.2g})",
                "    tol = ({:.2g}*,{:.2g},{:.2g})",
                "    tol = ({:.2g},{:.2g}*,{:.2g})",
                "    tol = ({:.2g},{:.2g},{:.2g}*)",
                ][self.stopping_criterion]
            settings += fmtstr.format(*self.tol)
        settings +="    (itns/time = {itns}/{time:.1f})\n".format(
            itns=self.nit, time=self.time
            )
        if pstyle == 'm':
            settings = ""
        elif self.alg != "lmsder":
            settings += "  alg = %s\n" % self.alg

        if maxline <= 0 or self.data is None:
            return table + settings
        # create table comparing fit results to data
        ny = self.y.size
        stride = 1 if maxline >= ny else (int(ny/maxline) + 1)
        f = self.fcn(self.p) if self.x is False else self.fcn(self.x, self.p)
        if hasattr(f, 'keys'):
            f = _gvar.BufferDict(f)
        else:
            f = numpy.array(f)
        data = collect(self.y, f, style='v', stride=stride)
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
        with open(filename, "wb") as f:
            if self.p0.shape is not None:
                pickle.dump(numpy.array(self.pmean), f)
            else:
                pickle.dump(collections.OrderedDict(self.pmean), f) # dump as a dict

    def simulated_fit_iter(self, n, pexact=None, bootstrap=False, **kargs):
        """ Iterator that returns simulation copies of a fit.

        Fit reliability can be tested using simulated data which
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
                ... verify that sfit.p agrees with pexact=fit.pmean within errors ...

        Only a few iterations are needed to get a sense of the fit's
        reliability since we know the correct answer in each case. The
        simulated fit's output results should agree with ``pexact``
        (``=fit.pmean`` here) within the simulated fit's errors.

        Simulated fits can also be used to estimate biases in the fit's
        output parameters or functions of them, should non-Gaussian behavior
        arise. This is possible, again, because we know the correct value for
        every parameter before we do the fit. Again only a few iterations
        may be needed for reliable estimates.

        The (possibly non-Gaussian) probability distributions for parameters,
        or functions of them, can be explored in more detail by setting option
        ``bootstrap=True`` and collecting results from a large number of
        simulated fits. With ``bootstrap=True``, the means of the priors are
        also varied from fit to fit, as in a bootstrap simulation; the new
        prior means are chosen at random from the prior distribution.
        Variations in the best-fit parameters (or functions of them)
        from fit to fit define the probability distributions for those
        quantities. For example, one would use the following code to
        analyze the distribution of function ``g(p)`` of the fit parameters::

            fit = nonlinear_fit(...)

            ...

            glist = []
            for sfit in fit.simulated_fit_iter(n=100, bootstrap=True):
                glist.append(g(sfit.pmean))

            ... analyze samples glist[i] from g(p) distribution ...

        This code generates ``n=100`` samples ``glist[i]`` from the
        probability distribution of ``g(p)``. If everything is Gaussian,
        the mean and standard deviation of ``glist[i]`` should agree
        with ``g(fit.p).mean`` and ``g(fit.p).sdev``.

        The only difference between simulated fits with ``bootstrap=True``
        and ``bootstrap=False`` (the default) is that the prior means are
        varied. It is essential that they be varied in a bootstrap analysis
        since one wants to capture the impact of the priors on the final
        distributions, but it is not necessary and probably not desirable
        when simply testing a fit's reliability.

        :param n: Maximum number of iterations (equals infinity if ``None``).
        :type n: integer or ``None``
        :param pexact: Fit-parameter values for the underlying distribution
            used to generate simulated data; replaced by ``self.pmean`` if
            is ``None`` (default).
        :type pexact: ``None`` or array or dictionary of numbers
        :param bootstrap: Vary prior means if ``True``; otherwise vary only
            the means in ``self.y`` (default).
        :type bootstrap: bool
        :returns: An iterator that returns :class:`lsqfit.nonlinear_fit`\s
            for different simulated data.

        Note that additional keywords can be added to overwrite keyword
        arguments in :class:`lsqfit.nonlinear_fit`.
        """
        pexact = self.pmean if pexact is None else pexact
        # Note: don't need svdcut since these are built into the data_iter
        fargs = dict(fcn=self.fcn, svdcut=None, p0=pexact)
        fargs.update(self.fitterargs)
        fargs.update(kargs)
        for ysim, priorsim in self._simulated_data_iter(
            n, pexact=pexact, bootstrap=bootstrap
            ):
            fit = nonlinear_fit(
                data=(self.x, ysim), prior=priorsim, extend=self.extend,
                **fargs
                )
            fit.pexact = pexact
            yield fit

    def _simulated_data_iter(self, n, pexact=None, bootstrap=True):
        """ Iterator that returns simulated data based upon a fit's data.

        Simulated data is generated from a fit's data ``fit.y`` by
        replacing the mean values in that data with random numbers
        drawn from a distribution whose mean is ``self.fcn(pexact)``
        and whose covariance matrix is the same as that of ``self.y``.
        Each iteration of the iterator returns new simulated data,
        with different random numbers for the means and a covariance
        matrix equal to that of ``self.y``. This iterator is used by
        ``self.simulated_fit_iter``.

        :param n: Maximum number of iterations (equals infinity if ``None``).
        :type n: integer or ``None``
        :param pexact: Fit-parameter values for the underlying distribution
            used to generate simulated data; replaced by ``self.pmean`` if
            is ``None`` (default).
        :type pexact: ``None`` or array or dictionary of numbers
        :param bootstrap: Vary prior means if ``True``; otherwise vary only
            the means in ``self.y`` (default).
        :type bootstrap: bool
        :returns: An iterator that returns a 2-tuple containing simulated
            versions of self.y and self.prior: ``(ysim, priorsim)``.
        """
        pexact = self.pmean if pexact is None else pexact
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
        if prior is None or not bootstrap:
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

    simulate_iter = simulated_fit_iter

    def bootstrapped_fit_iter(self, n=None, datalist=None):
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
            for sfit in fit.bootstrapped_fit_iter(
                n=100, datalist=datalist, bootstrap=True
                ):
                glist.append(g(sfit.pmean))

            ... analyze samples glist[i] from g(p) distribution ...

        This code generates ``n=100`` samples ``glist[i]`` from the
        probability distribution of ``g(p)``. If everything is Gaussian,
        the mean and standard deviation of ``glist[i]`` should agree
        with ``g(fit.p).mean`` and ``g(fit.p).sdev``.

        :param n: Maximum number of iterations if ``n`` is not ``None``;
            otherwise there is no maximum.
        :type n: integer
        :param datalist: Collection of bootstrap ``data`` sets for fitter.
        :type datalist: sequence or iterator or ``None``
        :returns: Iterator that returns an |nonlinear_fit| object
            containing results from the fit to the next data set in
            ``datalist``
        """
        fargs = {}
        fargs.update(self.fitterargs)
        fargs['fcn'] = self.fcn
        prior = self.prior
        if datalist is None:
            x = self.x
            y = self.y
            if n is None:
                raise ValueError("datalist, n can't both be None.")
            if prior is None:
                for yb in _gvar.bootstrap_iter(y, n):
                    fit = nonlinear_fit(
                        data=(x, yb), prior=None, p0=self.pmean,
                        extend=self.extend, **fargs
                        )
                    yield fit
            else:
                g = _gvar.BufferDict(y=y.flat, prior=prior.flat)
                for gb in _gvar.bootstrap_iter(g, n):
                    yb = _reformat(y, buf=gb['y'])
                    priorb = _reformat(prior, buf=gb['prior'])
                    fit = nonlinear_fit(
                        data=(x, yb), prior=priorb, p0=self.pmean,
                        extend=self.extend, **fargs
                        )
                    yield fit
        else:
            if prior is None:
                for datab in datalist:
                    fit = nonlinear_fit(
                        data=datab, prior=None, p0=self.pmean,
                        extend=self.extend, **fargs
                        )
                    yield fit
            else:
                piter = _gvar.bootstrap_iter(prior)
                for datab in datalist:
                    fit = nonlinear_fit(
                        data=datab, prior=next(piter), p0=self.pmean,
                        extend=self.extend, **fargs
                        )
                    yield fit

    bootstrap_iter = bootstrapped_fit_iter

def _reformat(p, buf, extend=False):
    """ Apply format of ``p`` to data in 1-d array ``buf``. """
    if numpy.ndim(buf) != 1:
        raise ValueError("Buffer ``buf`` must be 1-d.")
    if hasattr(p, 'keys'):
        ans = _gvar.BufferDict(p)
        if ans.size != len(buf):
            raise ValueError(       #
                "p, buf size mismatch: %d, %d"%(ans.size, len(buf)))
        if extend:
            ans = _gvar.ExtendedDict(ans, buf=buf)
        else:
            ans = _gvar.BufferDict(ans, buf=buf)
    else:
        if numpy.size(p) != len(buf):
            raise ValueError(       #
                "p, buf size mismatch: %d, %d"%(numpy.size(p), len(buf)))
        ans = numpy.array(buf).reshape(numpy.shape(p))
    return ans

def _unpack_data(data, prior, svdcut, extend):
    """ Unpack data and prior into ``(x, y, prior, fdata)``.

    This routine unpacks ``data`` and ``prior`` into ``x, y, prior, fdata``
    where ``x`` is the independent data, ``y`` is the fit data,
    ``prior`` is the collection of priors for the fit, and ``fdata``
    contains the information about the data and prior needed for the
    fit function. Both ``y`` and ``prior`` are modified to account
    for *svd* cuts if ``svdcut>0``.

    Note that redundant keys (eg, 'c' if 'logc' is a key) are removed
    from the prior if extend=True (and the prior is a dictionary).

    Allowed layouts for ``data`` are: ``x, y, ycov``, ``x, y, ysdev``,
    ``x, y``, and ``y``. In the last two case, ``y`` can be either an array
    of |GVar|\s or a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s. In the last case it is assumed that the fit function is a
    function of only the parameters: ``fcn(p)`` --- no ``x``. (This is also
    assumed if ``x = False``.)

    Output data in ``fdata`` is: ``fdata.mean`` containing the mean values of
    ``y.flat`` and ``prior.flat`` (if there is a prior);
    ``fdata.svdcorrection``  containing the sum of the *svd* corrections to
    ``y.flat`` and ``prior.flat``; ``fdata.logdet`` containing  the logarithm
    of the  determinant of the covariance matrix of ``y.flat`` and
    ``prior.flat``; and ``fdata.inv_wgts`` containing a representation of the
    inverse of the covariance matrix, after *svd* cuts (see :func:`gvar.svd`
    for a description of the format).
    """
    # unpack data tuple
    data_is_3tuple = False
    if not isinstance(data, tuple):
        x = False                   # no x in fit fcn
        y = _unpack_gvars(data)
    elif len(data) == 3:
        x, ym, ycov = data
        ym = numpy.asarray(ym)
        ycov = numpy.asarray(ycov)
        y = _gvar.gvar(ym, ycov)
        data_is_3tuple = True
    elif len(data) == 2:
        x, y = data
        y = _unpack_gvars(y)
    else:
        raise ValueError("data tuple wrong length: "+str(len(data)))

    def _apply_svd(data, svdcut=svdcut):
        ans, inv_wgts = _gvar.svd(data, svdcut=svdcut, wgts=-1)
        fdata = _FDATA(
            mean=_gvar.mean(data.flat),
            inv_wgts=inv_wgts,
            svdcorrection=numpy.sum(_gvar.svd.correction),
            logdet=_gvar.svd.logdet,
            nblocks=_gvar.svd.nblocks,
            svdn=_gvar.svd.nmod,
            )
        return ans, fdata

    if prior is None:
        y, fdata = _apply_svd(y)
    else:
        prior = _unpack_gvars(prior)
        if extend and hasattr(prior, 'keys'):
            # remove redundant keys, if any
            prior = _gvar.trim_redundant_keys(prior)
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

def _unpack_p0(p0, p0file, prior, extend):
    """ Create proper p0.

    Try to read from a file. If that doesn't work, try using p0,
    and then, finally, the prior. If the p0 is from the file, it is
    checked against the prior to make sure that all elements have the
    right shape; if not the p0 elements are adjusted (using info from
    the prior) to be the correct shape. If p0 is a dictionary,
    keys in p0 that are not in prior are discarded.

    Note that redundant keys (eg, 'c' if 'logc' is also a key) are
    removed from p0 if extend=True and there is no prior. There is
    no need to remove them if there is a prior, since its keys are
    used (and it has no redundant keys at this point).
    """
    if p0file is not None:
        # p0 is a filename; read in values
        try:
            p0 = nonlinear_fit.load_parameters(p0file)
            # with open(p0file, "rb") as f:
            #     p0 = pickle.load(f)
        except (IOError, EOFError):
            if prior is None:
                raise IOError(
                    "No prior and can't read parameters from " + p0file
                    )
            else:
                p0 = None
    if p0 is not None:
        # repackage as BufferDict or numpy array
        if hasattr(p0, 'keys'):
            p0 = _gvar.BufferDict(p0)
        else:
            p0 = numpy.array(p0)
        if p0.dtype != float:
            raise ValueError(
                'p0 must contain elements of type float, not type ' +
                str(p0.dtype)
                )
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
                # if set(pp.keys()) != set(p0.keys()):
                #     # mismatch in keys between prior and p0
                #     raise ValueError("Key mismatch between prior and p0: "
                #                      + ' '.join(str(k) for k in
                #                      set(prior.keys()) ^ set(p0.keys())))
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
    if extend and prior is None and p0.shape is None:
        # remove redundant keys (eg, remove a if loga present)
        # don't bother if there is a prior -- its keys are used to build p0
        p0 = _gvar.trim_redundant_keys(p0)
    return p0


def _unpack_fcn(fcn, p0, y, x, extend):
    """ reconfigure fitting fcn so inputs, outputs = flat arrays; hide x """
    if y.shape is not None:
        if p0.shape is not None:
            def nfcn(p, x=x, fcn=fcn, pshape=p0.shape):
                po = p.reshape(pshape)
                ans = fcn(po) if x is False else fcn(x, po)
                if hasattr(ans, 'flat'):
                    return ans.flat
                else:
                    return numpy.array(ans).flat
        else:
            if extend:
                po = _gvar.ExtendedDict(p0, buf=numpy.zeros(p0.size, float))
            else:
                po = _gvar.BufferDict(p0, buf=numpy.zeros(p0.size, float))
            def nfcn(p, x=x, fcn=fcn, po=po):
                po.buf = p
                ans = fcn(po) if x is False else fcn(x, po)
                if hasattr(ans, 'flat'):
                    return ans.flat
                else:
                    return numpy.array(ans).flat
    else:
        yo = _gvar.BufferDict(y, buf=y.size*[None])
        if p0.shape is not None:
            def nfcn(p, x=x, fcn=fcn, pshape=p0.shape, yo=yo):
                po = p.reshape(pshape)
                fxp = fcn(po) if x is False else fcn(x, po)
                for k in yo:
                    yo[k] = fxp[k]
                return yo.flat
        else:
            if extend:
                po = _gvar.ExtendedDict(p0, buf=numpy.zeros(p0.size, float))
            else:
                po = _gvar.BufferDict(p0, buf=numpy.zeros(p0.size, float))
            def nfcn(p, x=x, fcn=fcn, po=po, yo=yo):
                po.buf = p
                fxp = fcn(po) if x is False else fcn(x, po)
                for k in yo:
                    yo[k] = fxp[k]
                return yo.flat
    return nfcn

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
            if y.isscalar(k):
                if not f.isscalar(k):
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

class BayesPDF(_gvar.PDF):
    """ Bayesian probability density function corresponding to :class:`nonlinear_fit` ``fit``.

    The probability density function is the exponential of
    ``-1/2`` times the ``chi**2`` function (data and priors) used
    in ``fit`` divided by ``norm``.

    Args:
        fit: Fit from :class:`nonlinear_fit`.

        svdcut (non-negative float or None): If not ``None``, replace
            covariance matrix of ``g`` with a new matrix whose
            small eigenvalues are modified: eigenvalues smaller than
            ``svdcut`` times the maximum eigenvalue ``eig_max`` are
            replaced by ``svdcut*eig_max``. This can ameliorate
            problems caused by roundoff errors when inverting the
            covariance matrix. It increases the uncertainty associated
            with the modified eigenvalues and so is conservative.
            Setting ``svdcut=None`` or ``svdcut=0`` leaves the
            covariance matrix unchanged. Default is ``1e-15``.
    """
    def __init__(self, fit, svdcut=1e-15, norm=1.0):
        super(BayesPDF, self).__init__(fit.p, svdcut=svdcut)
        self.chiv = fit._chiv
        self.chi2 = fit.chi2
        self.log_norm = numpy.log(norm)

    def logpdf(self, p, lognorm=None):
        """ Logarithm of the probability density function evaluated at ``p``. """
        if lognorm is None:
            lognorm = self.log_norm
        if hasattr(p, 'keys'):
            if self.extend:
                p = _gvar.trim_redundant_keys(p).buf
            else:
                if isinstance(p, _gvar.BufferDict):
                    p = p.buf
                else:
                    p = _gvar.BufferDict(p).buf
        else:
            p = numpy.asarray(p).reshape(-1)
        return (
            -0.5 * (numpy.sum(self.chiv(p) ** 2) - self.chi2)
            - self.log_gnorm - self.log_norm
            )


try:
    import vegas
    class BayesIntegrator(_gvar.PDFIntegrator):
        """ :mod:`vegas` integrator for Bayesian fit integrals.

        Args:
            fit: Fit from :class:`nonlinear_fit`.

            limit (positive float): Limits the integrations to a finite
                region of size ``limit`` times the standard deviation on
                either side of the mean. This can be useful if the
                functions being integrated misbehave for large parameter
                values (e.g., ``numpy.exp`` overflows for a large range of
                arguments). Default is ``1e15``.

            scale (positive float): The integration variables are
                rescaled to emphasize parameter values of order
                ``scale`` times the corresponding standard deviations.
                The rescaling does not change the value of the integral but it
                can reduce uncertainties in the :mod:`vegas` estimates.
                Default is ``1.0``.

            pdf (function): Probability density function ``pdf(p)`` of the
                fit parameters to use in place of the normal PDF associated
                with the least-squares fit used to create the integrator.

            svdcut (non-negative float or None): If not ``None``, replace
                covariance matrix of ``g`` with a new matrix whose
                small eigenvalues are modified: eigenvalues smaller than
                ``svdcut`` times the maximum eigenvalue ``eig_max`` are
                replaced by ``svdcut*eig_max``. This can ameliorate
                problems caused by roundoff errors when inverting the
                covariance matrix. It increases the uncertainty associated
                with the modified eigenvalues and so is conservative.
                Setting ``svdcut=None`` or ``svdcut=0`` leaves the
                covariance matrix unchanged. Default is ``1e-15``.


        ``BayesIntegrator(fit)`` is a :mod:`vegas` integrator that evaluates
        expectation values for the multi-dimensional Bayesian distribution
        associated with :class:`nonlinear_fit` ``fit``: the probability
        density function is the exponential of the ``chi**2`` function
        (times ``-1/2``), for data and priors, used in the fit.
        For linear fits, it is equivalent to ``gvar.PDFIntegrator(fit.p)``,
        since the ``chi**2`` function is  quadratic in the fit parameters;
        but they can differ significantly for nonlinear fits.

        ``BayesIntegrator`` integrates over the entire parameter space but
        first reexpresses the integrals in terms of variables that
        diagonalize the covariance matrix of the best-fit parameters
        ``fit.p`` from :class:`nonlinear_fit` and are centered at the
        best-fit values. This greatly facilitates the integration using
        :mod:`vegas`, making integrals over 10s or more of parameters feasible.
        (The :mod:`vegas` module must be installed separately in order
        to use ``BayesIntegrator``.)

        A simple illustration of ``BayesIntegrator`` is given by the following
        code, which we use to evaluate the mean and standard deviation for
        ``s*g`` where ``s`` and ``g`` are fit parameters::

            import lsqfit
            import gvar as gv
            import numpy as np

            # least-squares fit
            x = np.array([0.1, 1.2, 1.9, 3.5])
            y = gv.gvar(['1.2(1.0)', '2.4(1)', '2.0(1.2)', '5.2(3.2)'])
            prior = gv.gvar(dict(a='0(5)', s='0(2)', g='2(2)'))
            def f(x, p):
                return p['a'] + p['s'] * x ** p['g']
            fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=f, debug=True)
            print(fit)

            # Bayesian integral to evaluate expectation value of s*g
            def g(p):
                sg = p['s'] * p['g']
                return [sg, sg**2]

            expval = lsqfit.BayesIntegrator(fit, limit=20.)
            warmup = expval(neval=4000, nitn=10)
            results = expval(g, neval=4000, nitn=15, adapt=False)
            print(results.summary())
            print('results =', results, '\\n')
            sg, sg2 = results
            sg_sdev = (sg2 - sg**2) ** 0.5
            print('s*g from Bayes integral:  mean =', sg, '  sdev =', sg_sdev)
            print('s*g from fit:', fit.p['s'] * fit.p['g'])

        where the ``warmup`` calls to the integrator are used to adapt it to
        probability density function from the fit, and then the integrator
        is used to evaluate the expectation value of ``g(p)``, which is
        returned in array ``results``.  Here ``neval`` is the (approximate)
        number of function calls per iteration of the :mod:`vegas` algorithm
        and ``nitn`` is the number of iterations. We use the integrator to
        calculated the expectation value of ``s*g`` and ``(s*g)**2`` so we can
        compute a mean and standard deviation.

        The output from this code shows that the Gaussian approximation
        for ``s*g`` (0.76(66)) is somewhat different from the result
        obtained from a Bayesian integral (0.48(54))::

            Least Square Fit:
              chi2/dof [dof] = 0.32 [4]    Q = 0.87    logGBF = -9.2027

            Parameters:
                          a    1.61 (90)     [  0.0 (5.0) ]
                          s    0.62 (81)     [  0.0 (2.0) ]
                          g    1.2 (1.1)     [  2.0 (2.0) ]

            Settings:
              svdcut/n = 1e-15/0    reltol/abstol = 0.0001/0*    (itns/time = 10/0.0)

            itn   integral        average         chi2/dof        Q
            -------------------------------------------------------
              1   1.034(21)       1.034(21)           0.00     1.00
              2   1.034(21)       1.034(15)           0.56     0.64
              3   1.024(18)       1.030(12)           0.37     0.90
              4   1.010(18)       1.0254(98)          0.47     0.89
              5   1.005(17)       1.0213(85)          0.55     0.88
              6   1.013(19)       1.0199(78)          0.69     0.80
              7   0.987(16)       1.0152(70)          0.78     0.72
              8   1.002(18)       1.0135(66)          0.90     0.59
              9   1.036(20)       1.0160(62)          0.86     0.66
             10   1.060(20)       1.0204(60)          0.94     0.55

            results = [0.4837(32) 0.5259(47)]

            s*g from Bayes integral:  mean = 0.4837(32)   sdev = 0.5403(25)
            s*g from fit: 0.78(66)

        The table shows estimates of the probability density function's
        normalization from each of the :mod:`vegas` iterations used
        by the integrator to estimate the final results.

        In general functions being integrated can return a number, or an array
        of numbers, or a dictionary whose values are numbers or arrays of
        numbers. This allows multiple expectation values to be evaluated
        simultaneously.

        See the documentation with the :mod:`vegas` module for more details on
        its use, and on the attributes and methods associated with
        integrators. The example above sets ``adapt=False`` when  computing
        final results. This gives more reliable error estimates  when
        ``neval`` is small. Note that ``neval`` may need to be much larger
        (tens or hundreds of thousands) for more difficult high-dimension
        integrals.
        """
        def __init__(self, fit, limit=1e15, scale=1.0, pdf=None, svdcut=1e-15):
            super(BayesIntegrator, self).__init__(
                BayesPDF(fit, svdcut=svdcut),
                scale=scale, svdcut=svdcut, limit=limit,
                )
            self.pdf_function = pdf

        def __call__(self, f=None, mpi=False, pdf=None, **kargs):
            """ Estimate expectation value of function ``f(p)``.

            Uses multi-dimensional integration modules :mod:`vegas`  to
            estimate the expectation value of ``f(p)`` with  respect to
            the probability density function  associated with
            :class:`nonlinear_fit` ``fit``. Setting ``mpi=True``
            configures vegas for multi-processor running using MPI.

            Args:
                f (function): Function ``f(p)`` to integrate. Integral is
                    the expectation value of the function with respect
                    to the distribution. The function can return a number,
                    an array of numbers, or a dictionary whose values are
                    numbers or arrays of numbers.

                mpi (bool): If ``True`` configure for use with multiple processors
                    and MPI. This option requires module :mod:`mpi4py`. A
                    script ``xxx.py`` using an MPI integrator is run
                    with ``mpirun``: e.g., ::

                        mpirun -np 4 -output-filename xxx.out python xxx.py

                    runs on 4 processors. Setting ``mpi=False`` (default) does
                    not support multiple processors. The MPI processor
                    rank can be obtained from the ``mpi_rank``
                    attribute of the integrator.

                pdf (function): Probability density function ``pdf(p)`` of the
                    fit parameters to use in place of the normal PDF associated
                    with the least-squares fit used to create the integrator.

            All other keyword arguments are passed on to a :mod:`vegas`
            integrator; see the :mod:`vegas` documentation for further
            information.
            """

            if pdf is None:
                pdf = self.pdf_function
            self._buffer = None
            def _fstd(p, pdf=pdf):
                """ convert output to an array """
                fp = [] if f is None else f(p)
                if pdf is None:
                    pdf = numpy.exp(self.pdf.logpdf(p))
                else:
                    pdf = pdf(p)
                if self._buffer is None:
                    # setup --- only on first call
                    self._is_dict = hasattr(fp, 'keys')
                    if self._is_dict:
                        self._fp = _gvar.BufferDict(fp)
                        self._is_bdict = isinstance(fp, _gvar.BufferDict)
                        bufsize = self._fp.buf.size + 1
                    else:
                        self._is_bdict = False
                        self._fp = numpy.asarray(fp)
                        bufsize = self._fp.size + 1
                    self._buffer = numpy.empty(bufsize, float)
                    self._buffer[0] = 1.
                if self._is_bdict:
                    self._buffer[1:] = fp.buf
                elif self._is_dict:
                    self._buffer[1:] = BufferDict(fp).buf
                else:
                    self._buffer[1:] = numpy.asarray(fp).flat[:]
                return self._buffer * pdf
            results = super(BayesIntegrator, self).__call__(
                _fstd=_fstd, nopdf=True, mpi=mpi, **kargs
                )
            self.norm = results[0]
            if self._fp.shape is None:
                return _gvar._RAvgDictWrapper(self._fp, results)
            elif self._fp.shape != ():
                return _gvar._RAvgArrayWrapper(self._fp.shape, results)
            else:
                return _gvar._RAvgWrapper(results)
except ImportError:
    pass

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

# from scipy import optimize
# class LSQFitter(object):
#     # have this choose itself unless alg=lmder or lmsder in which case it takes the other
#     def __init__(self, p0, nf, f, reltol=0.0001, abstol=0.0, maxit=1000, alg="scipy", analyzer=None):
#         def func(x, ans=numpy.empty(nf, float)):
#             ans[:] = f(x)
#             return ans
#         def Dfun(x, ans=numpy.empty((nf, len(p0)), float)):
#             xv = _gvar.valder(x)
#             fx = f(xv)
#             for i, fxi in enumerate(fx):
#                 ans[i] = fxi.der
#             return ans
#         x, cov_x, infodict, mesg, ier = optimize.leastsq(
#             func=func,
#             x0=p0,
#             Dfun=Dfun,
#             full_output=True,
#             xtol=reltol,
#             maxfev=maxit,
#             # diag=numpy.ones(len(p0), float),
#             )
#         self.x = x
#         self.error = None if ier in [1,2,3,4] else mesg  # error message or None
#         self.cov = cov_x         # cov matrix from fit
#         self.nit = infodict['nfev']         # number of iterations used in fit
#         self.f = infodict['fvec']       # value of f at min point
#         self.J = None               # df_i / dx[j]
#         self.abstol = 0.0
#         self.reltol = reltol
#         self.alg = alg

# # multifit = LSQFitter
