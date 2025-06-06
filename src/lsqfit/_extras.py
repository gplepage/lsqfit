""" part of lsqfit module: extra functions  """

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-21 G. Peter Lepage.
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
import copy
import functools
import pickle
import time
import warnings

import numpy

import lsqfit
import gvar

_multiminex = getattr(lsqfit, 'gsl_multiminex', lsqfit.scipy_multiminex)

def empbayes_fit(z0, fitargs, p0=None, fitter=lsqfit.nonlinear_fit, **minargs):
    """ Return ``fit`` and ``z`` corresponding to the fit
    ``lsqfit.nonlinear_fit(**fitargs(z))`` that maximizes ``fit.logGBF``.

    This function maximizes the logarithm of the Bayes Factor from
    fit  ``lsqfit.nonlinear_fit(**fitargs(z))`` by varying ``z``,
    starting at ``z0``. The fit is redone for each value of ``z``
    that is tried, in order to determine ``fit.logGBF``.

    The Bayes Factor is proportional to the probability that the data
    came from the model (fit function and priors) used in the fit.
    :func:`empbayes_fit` finds the model or data that maximizes this
    probability.

    One application is illustrated by the following code::

        import numpy as np
        import gvar as gv
        import lsqfit

        # fit data
        x = np.array([1., 2., 3., 4.])
        y = np.array([3.4422, 1.2929, 0.4798, 0.1725])

        # prior
        prior = gv.gvar(['10(1)', '1.0(1)'])

        # fit function
        def fcn(x, p):
            return p[0] * gv.exp( - p[1] * x)

        # find optimal dy
        def fitargs(z):
            dy = y * z
            newy = gv.gvar(y, dy)
            return dict(data=(x, newy), fcn=fcn, prior=prior)

        fit, z = lsqfit.empbayes_fit(0.1, fitargs)
        print(fit.format(True))

    Here we want to fit data ``y`` with fit function ``fcn`` but we don't know
    the uncertainties in our ``y`` values. We assume that the relative errors
    are ``x``-independent and uncorrelated. We add the error ``dy`` that
    maximizes the Bayes Factor, as this is the most likely choice. This fit
    gives the following output::

        Least Square Fit:
          chi2/dof [dof] = 0.58 [4]    Q = 0.67    logGBF = 7.4834

        Parameters:
                      0     9.44 (18)     [ 10.0 (1.0) ]
                      1   0.9979 (69)     [  1.00 (10) ]

        Fit:
             x[k]           y[k]      f(x[k],p)
        ---------------------------------------
                1     3.442 (54)     3.481 (45)
                2     1.293 (20)     1.283 (11)
                3    0.4798 (75)    0.4731 (41)
                4    0.1725 (27)    0.1744 (23)

        Settings:
          svdcut/n = 1e-12/0    tol = (1e-08*,1e-10,1e-10)    (itns/time = 3/0.0)


    We have, in effect, used the variation in the data relative to the best
    fit curve to estimate that the uncertainty in each data point is
    of order 1.6%.

    :func:`empbayes_fit` can be used with other fitters: for example, 
    to use ``lsqfit.vegas_int`` instead of ``lsqfit.nonlinear_fit`` 
    (the default) for the fits, replace the next to last line in 
    the code above with ::
    
        fit, z = lsqfit.empbayes_fit(0.1, fitargs, fitter=lsqfit.vegas_fit).

    See also :meth:`MultiFitter.empbayes_fit`.

    Args:
        z0 (number, array or dict): Starting point for search.
        fitargs (callable): Function of ``z`` that returns a
            dictionary ``args`` containing the :class:`lsqfit.nonlinear_fit`
            arguments corresponding to ``z``. ``z`` should have
            the same layout (number, array or dictionary) as ``z0``.
            ``fitargs(z)`` can instead return a tuple ``(args, plausibility)``,
            where ``args`` is again the dictionary for
            :class:`lsqfit.nonlinear_fit`. ``plausibility`` is the logarithm
            of the *a priori* probabilitiy that ``z`` is sensible. When
            ``plausibility`` is provided, :func:`lsqfit.empbayes_fit`
            maximizes the sum ``logGBF + plausibility``. Specifying
            ``plausibility`` is a way of steering selections away from
            completely implausible values for ``z``.
        p0: Fit-parameter starting values for the first fit. ``p0``
            for subsequent fits is set automatically to optimize fitting
            unless a value is specified by ``fitargs``.
        fitter: Fitter to be used. Default is ``lsqfit.nonlinear_fit``;
            also works with ``lsqfit.vegas_fit``.
        minargs (dict): Optional argument dictionary, passed on to
            :class:`lsqfit.gsl_multiminex` (or
            :class:`lsqfit.scipy_multiminex`), which finds the minimum.

    Returns:
        A tuple containing the best fit (object of type
        :class:`lsqfit.nonlinear_fit`) and the
        optimal value for parameter ``z``.
    """
    save = dict(lastz=None, lastp0=p0)
    if hasattr(z0, 'keys'):
        # z is a dictionary
        z0 = gvar.BufferDict(z0, dtype=float)
        z0buf = z0.buf
        def convert(zbuf):
            return gvar.BufferDict(z0, buf=zbuf)
    elif numpy.shape(z0) == ():
        # z is a number
        z0buf = numpy.array([z0], dtype=float)
        def convert(zbuf):
            return zbuf[0]
    else:
        # z is an array
        z0buf = numpy.asarray(z0, dtype=float)
        def convert(zbuf):
            return zbuf
    def minfcn(zbuf, save=save, convert=convert):
        z = convert(zbuf)
        args = fitargs(z)
        if not hasattr(args, 'keys'):
            args, plausibility = args
        else:
            plausibility = 0.0
        if save['lastp0'] is not None and 'p0' not in args and fitter != lsqfit.vegas_fit:
            args['p0'] = save['lastp0']
        fit = fitter(**args)
        if hasattr(fit, 'logBF'):
            logGBF = gvar.mean(fit.logBF)
        elif hasattr(fit, 'logGBF'):
            logGBF = fit.logGBF 
        else:
            raise RuntimeError('fit has no logGBF or logBF')
        if numpy.isnan(logGBF):
            raise ValueError('logGBF undefined - nan')
        else:
            save['lastz'] = z
            save['lastp0'] = fit.pmean
        return -logGBF - plausibility
    try:
        z = convert(_multiminex(z0buf, minfcn, **minargs).x)
    except ValueError:
        print('*** empbayes_fit warning: null logGBF')
        z = save['lastz']
    args = fitargs(z)
    if not hasattr(args, 'keys'):
        args, plausibility = args
    if save['lastp0'] is not None and 'p0' not in args and fitter != lsqfit.vegas_fit:
        args['p0'] = save['lastp0']
    return fitter(**args), z


class GVarWAvg(gvar.GVar):
    r""" Result from weighted average :func:`lsqfit.wavg`.

    :class:`GVarWAvg` objects are |GVar|\s with extra
    attributes:

    .. attribute:: chi2

        ``chi**2`` for weighted average.

    .. attribute:: dof

        Effective number of degrees of freedom.

    .. attribute:: Q

        The probability that the ``chi**2`` could have been larger,
        by chance, assuming that the data are all Gaussian and consistent
        with each other. Values smaller than 0.1 or suggest that the
        data are not Gaussian or are inconsistent with each other. Also
        called the *p-value*.

        Quality factor `Q` (or *p-value*) for fit.

    .. attribute:: time

        Time required to do average.

    .. attribute:: correction

        The *svd* corrections added to the data in the average.

    .. attribute:: fit

        Fit result from average.
    """
    def __init__(self, avg, fit):
        super(GVarWAvg, self).__init__(*avg.internaldata)
        if fit is None:
            self.chi2 = 0
            self.dof = 1
            self.Q = 1
            self.correction = gvar.gvar(0,0)
            self.time = 0
            self.fit = None
        else:
            self.chi2 = fit.chi2
            self.dof = fit.dof
            self.Q = fit.Q
            self.time = fit.time
            self.correction = fit.correction
            self.fit = fit
        self.svdcorrection = self.correction # legacy name

class ArrayWAvg(numpy.ndarray):
    r""" Result from weighted average :func:`lsqfit.wavg`.

    :class:`ArrayWAvg` objects are :mod:`numpy` arrays (of |GVar|\s) with extra
    attributes:

    .. attribute:: chi2

        ``chi**2`` for weighted average.

    .. attribute:: dof

        Effective number of degrees of freedom.

    .. attribute:: Q

        The probability that the ``chi**2`` could have been larger,
        by chance, assuming that the data are all Gaussian and consistent
        with each other. Values smaller than 0.1 or suggest that the
        data are not Gaussian or are inconsistent with each other. Also
        called the *p-value*.

        Quality factor `Q` (or *p-value*) for fit.

    .. attribute:: time

        Time required to do average.

    .. attribute:: correction

        The *svd* corrections added to the data in the average.

    .. attribute:: fit

        Fit result from average.
    """
    def __new__(cls, g, fit):
        obj = numpy.ndarray.__new__(cls, g.shape, g.dtype, g.flatten())
        if fit is None:
            obj.chi2 = 0
            obj.dof = 1
            obj.Q = 1
            obj.time = 0
            obj.correction = gvar.gvar(0, 0)
        else:
            obj.chi2 = fit.chi2
            obj.dof = fit.dof
            obj.Q = fit.Q
            obj.time = fit.time
            obj.correction = fit.correction
            obj.fit = fit
        return obj

    # def __array_finalize__(self, obj):   # don't want this
    #     if obj is None:                   # since it attaches Q, etc
    #         return                        # to any array descended from
    #     self.chi2 = obj.chi2              # this one
    #     self.dof = obj.dof                # really just want Q, etc.
    #     self.Q = obj.Q                    # for initial output of wavg
    #     self.time = obj.time
    #     self.fit = obj.fit

class BufferDictWAvg(gvar.BufferDict):
    r""" Result from weighted average :func:`lsqfit.wavg`.

    :class:`BufferDictWAvg` objects are :class:`gvar.BufferDict`\s (of |GVar|\s)
    with extra attributes:

    .. attribute:: chi2

        ``chi**2`` for weighted average.

    .. attribute:: dof

        Effective number of degrees of freedom.

    .. attribute:: Q

        The probability that the ``chi**2`` could have been larger,
        by chance, assuming that the data are all Gaussian and consistent
        with each other. Values smaller than 0.1 or suggest that the
        data are not Gaussian or are inconsistent with each other. Also
        called the *p-value*.

        Quality factor `Q` (or *p-value*) for fit.

    .. attribute:: time

        Time required to do average.

    .. attribute:: correction

        The *svd* corrections added to the data in the average.

    .. attribute:: fit

        Fit result from average.
    """
    def __init__(self, g, fit):
        super(BufferDictWAvg, self).__init__(g)
        if fit is None:
            self.chi2 = 0
            self.dof = 1
            self.Q = 1
            self.correction = gvar.gvar(0,0)
            self.time = 0
            self.fit = None
        else:
            self.chi2 = fit.chi2
            self.dof = fit.dof
            self.Q = fit.Q
            self.time = fit.time
            self.correction = fit.correction
            self.fit = fit
        self.svdcorrection = self.correction  # legacy name

def wavg(datalist, fast=False, prior=None, **fitterargs):
    r""" Weighted average of |GVar|\s or arrays/dicts of |GVar|\s.

    The weighted average of ``N`` |GVar|\s ::

        xavg = wavg([g1, g2 ... gN])

    is what one obtains from a weighted least-squares fit of the
    collection of |GVar|\s to the one-parameter fit function ::

        def f(p):
            return N * [p]

    The average is the best-fit value for fit parameter ``p``.  |GVar|\s
    with smaller standard deviations carry more weight than those with
    larger standard deviations; and the averages take account of
    correlations between the |GVar|\s.

    ``wavg`` also works when each ``gi`` is an array of |GVar|\s or a
    dictionary whose values are |GVar|\s or arrays of |GVar|\s.
    Corresponding arrays in different ``gi``\s must have the same dimension,
    but can have different shapes (the overlapping components are
    averaged).  When the ``gi`` are dictionaries, they need not all have
    the same keys.

    Weighted averages can become costly when the number of random samples
    being averaged is large (100s or more). In such cases it might be useful
    to set parameter ``fast=True``. This causes ``wavg`` to estimate the
    weighted average by incorporating the random samples one at a time into a
    running average::

        result = datalist[0]
        for di in datalist[1:]:
            result = wavg([result, di], ...)

    This method can be much faster when ``len(datalist)`` is large, and gives
    the exact result when there are no correlations between different elements
    of list ``datalist``. The results are approximately correct when
    ``datalist[i]`` and ``datalist[j]`` are correlated for ``i!=j``.

    Args:
        datalist (list): The |GVar|\s to be averaged. ``datalist`` is
            a one-dimensional sequence of |GVar|\s, or of arrays of |GVar|\s,
            or of dictionaries containing |GVar|\s  and/or arrays of |GVar|\s.
            Corresponding arrays in different ``datalist[i]``\s must have the
            same dimension.
        fast (bool): If ``fast=True``, ``wavg`` averages the ``datalist[i]``
            sequentially. This can be much faster when averaging a large
            number of sampes but is only approximate if the different
            elements of ``datalist`` are correlated. Default is ``False``.
        fitterargs (dict): Additional arguments (e.g., ``svdcut`` or ``eps``) 
            for the :class:`lsqfit.nonlinear_fit` fitter used to do the 
            averaging.

    Returns:
        The weighted average is returned as a |GVar| or an array of
        |GVar|\s or a dictionary of |GVar|\s and arrays of |GVar|\s.
        Results have the following extra attributes:

        **chi2** - ``chi**2`` for weighted average.

        **dof** - Effective number of degrees of freedom.

        **Q** - Quality factor `Q` (or *p-value*) for fit:
            the probability that the ``chi**2`` could have been larger,
            by chance, assuming that the data are all Gaussian and consistent
            with each other. Values smaller than 0.1 or so suggest that the
            data are not Gaussian or are inconsistent with each other. Also
            called the *p-value*.

        **time** - Time required to do average.

        **correction** - The corrections made to the data
            when ``svdcut>0`` or ``eps>0``.

        **fit** - Fit returned by :class:`lsqfit.nonlinear_fit`.
    """
    # if prior is not None:
    #     datalist = list(datalist) + [prior]
    #     warnings.warn(
    #         'use of prior in lsqfit.wavg is deprecated',
    #         DeprecationWarning
    #         )
    if len(datalist) <= 0:
        return None
    elif len(datalist) == 1:
        if hasattr(datalist[0], 'keys'):
            return BufferDictWAvg(datalist[0], None)
        if numpy.shape(datalist[0]) == ():
            return GVarWAvg(datalist[0], None)
        else:
            return ArrayWAvg(numpy.asarray(datalist[0]), None)
    if fast:
        chi2 = dof = time = correction = 0
        ans = datalist[0]
        for i, di in enumerate(datalist[1:]):
            ans = wavg([ans, di], fast=False, **fitterargs)
            chi2 += ans.chi2
            dof += ans.dof
            time += ans.time
            correction += ans.correction
        ans.fit.dof = dof
        ans.fit.Q = lsqfit.gammaQ(dof / 2., chi2 / 2.)
        ans.fit.chi2 = chi2
        ans.fit.time = time
        ans.fit.correction = correction
        return ans
    if hasattr(datalist[0], 'keys'):
        datashape = None
    else:
        datashape = numpy.shape(datalist[0])
        datalist = [{None:di} for di in datalist]
    # repack as a single dictionary
    p0shape = {}
    p0index = {}
    data = gvar.BufferDict()
    for i, di in enumerate(datalist):
        for k in di:
            data[k, i] = di[k]
            shape = numpy.shape(di[k])
            p0index[k, i] = tuple(slice(0, j) for j in shape)
            if k not in p0shape:
                p0shape[k] = shape
            elif p0shape[k] != shape:
                p0shape[k] = tuple(
                    max(j1, j2) for j1, j2 in zip(shape, p0shape[k])
                    )
    # calculate p0
    p0 = gvar.BufferDict()
    p0count = {}
    for k, i in data:
        if k not in p0:
            p0[k] = numpy.zeros(p0shape[k], float)
            p0count[k] = numpy.zeros(p0shape[k], float)
        if p0index[k, i] == ():
            p0[k] += data[k, i].mean
            p0count[k] += 1
        else:
            p0[k][p0index[k, i]] += gvar.mean(data[k, i])
            p0count[k][p0index[k, i]] += 1.
    for k in p0:
        p0[k] /= p0count[k]
    # set up fit
    def fcn(p):
        ans = gvar.BufferDict()
        for k, i in data:
            shape = data[k, i].shape
            if shape == ():
                ans[k, i] = p[k]
            else:
                ans[k, i] = p[k][p0index[k, i]]
        return ans
    fit = lsqfit.nonlinear_fit(data=data, fcn=fcn, p0=p0, **fitterargs)
    if datashape is None:
        return BufferDictWAvg(fit.p, fit)
    elif datashape == ():
        return GVarWAvg(fit.p[None], fit)
    else:
        return ArrayWAvg(fit.p[None], fit)


class MultiFitterModel(object):
    """ Base class for MultiFitter models.

    Derived classes must define methods ``fitfcn``, ``buildprior``, and
    ``builddata``, all of which are described below. In addition they
    have attributes:

    .. attribute:: datatag

       :class:`lsqfit.MultiFitter` builds fit data for the correlator by
       extracting the data labelled by ``datatag`` (eg, a string) from an
       input data set (eg, a dictionary). This label is stored in the
       ``MultiFitterModel`` and must be passed to its constructor. It must be
       a hashable quantity, like a string or number or tuple of strings and
       numbers.

    .. attribute:: ncg

        When ``ncg>1``, fit data and functions are coarse-grained by
        breaking them up into bins of of ``ncg`` values and replacing
        each bin by its average. This can increase the fitting speed,
        because there is less data, without much loss of precision
        if the data elements within a bin are highly correlated.

    Args:
        datatag: Label used to identify model's data.
        ncg (int): Size of bins for coarse graining (default is ``ncg=1``).
    """
    def __init__(self, datatag, ncg=1):
        super(MultiFitterModel, self).__init__()
        self.datatag = datatag
        self.ncg = ncg

    def fitfcn(self, p):
        """ Compute fit function fit for parameters ``p``.

        Results are returned in a 1-dimensional array the
        same length as (and corresponding to) the fit data
        returned by ``self.builddata(data)``.

        If marginalization is supported, ``fitfcn`` must work
        with or without the marginalized parameters.

        Args:
            p: Dictionary of parameter values.
        """
        raise NotImplementedError("fitfcn not defined")

    def builddataset(self, dataset):
        r""" Extract fit dataset from :class:`gvar.dataset.Dataset` ``dataset``.

        The code  ::

            import gvar as gv

            data = gv.dataset.avg_data(m.builddataset(dataset))

        that builds data for model ``m`` should be functionally
        equivalent to ::

            import gvar as gv

            data = m.builddata(gv.dataset.avg_data(dataset))

        This method is optional. It is used only by
        :meth:`MultiFitter.process_dataset`.

        Args:
            dataset: :class:`gvar.dataset.Dataset` (or similar dictionary)
                dataset containing the fit data for all models. This is
                typically a dictionary, whose keys are the ``datatag``\s of
                the models.

        """
        raise NotImplementedError("builddataset not defined")

    def builddata(self, data):
        r""" Extract fit data corresponding to this model from data set ``data``.

        The fit data is returned in a 1-dimensional array;
        the fitfcn must return arrays of the same length.

        Args:
            data: Data set containing the fit data for all models. This
                is typically a dictionary, whose keys are the ``datatag``\s
                of the models.
        """
        raise NotImplementedError("builddata not defined")

    def buildprior(self, prior, mopt=None):
        """ Extract fit prior from ``prior``.

        Returns a dictionary containing the  part of dictionary
        ``prior`` that is relevant to this model's fit. The code could
        be as simple as collecting the appropriate pieces: e.g., ::

            def buildprior(self, prior, mopt=None):
                mprior = gv.BufferDict()
                model_keys = [...]
                for k in model_keys:
                    mprior[k] = prior[k]
                return mprior

        where ``model_keys`` is a list of keys corresponding to
        the model's parameters. Supporting non-Gaussian distributions
        requires a slight modification: e.g., ::

            def buildprior(self, prior, mopt=None):
                mprior = gv.BufferDict()
                model_keys = [...]
                for k in gv.get_dictkeys(prior, model_keys):
                    mprior[k] = prior[k]
                return mprior

        Marginalization involves omitting some of the fit parameters from the
        model's prior. ``mopt=None`` implies no marginalization. Otherwise
        ``mopt`` will typically contain information about what and how much
        to marginalize.

        Args:
            prior: Dictionary containing *a priori* estimates of all
                fit parameters.
            mopt (object): Marginalization options. Ignore if ``None``.
                Otherwise marginalize fit parameters as specified by ``mopt``.
                ``mopt`` can be any type of Python object; it is used only
                in ``buildprior`` and is passed through to it unchanged.
        """
        raise NotImplementedError("buildprior not defined")

class unchained_nonlinear_fit(lsqfit.nonlinear_fit):
    def __init__(self, fname, fitter_args_kargs, *args, **kargs):
        super(unchained_nonlinear_fit, self).__init__(*args, **kargs)
        self.chained_fits = collections.OrderedDict([(fname, self)])    
        fitter_args_kargs[1]['p0'] = self.pmean    
        self.fitter_args_kargs = fitter_args_kargs

    def _remove_gvars(self, gvlist):
        self.p  # need to fill _p
        fit = copy.copy(self)
        fitter,args,kargs = fit.fitter_args_kargs
        try:
            fit.pickled_models = pickle.dumps(args['models'])
        except:
            if self.debug:
                warnings.warn('unable to pickle fit function; it is omitted')
        fit.chained_fits = list(fit.chained_fits.keys())[0]
        for k in ['_chiv', '_chivw', 'fcn', 'fitter_args_kargs']:
            del fit.__dict__[k]
        fit.__dict__ = gvar.remove_gvars(fit.__dict__, gvlist)
        return fit
    
    def _distribute_gvars(self, gvlist):
        self.__dict__ = gvar.distribute_gvars(self.__dict__, gvlist)
        self.chained_fits = collections.OrderedDict([(self.chained_fits, self)])
        try:
            models = pickle.loads(self.pickled_models)
            self.fcn = MultiFitter(models).buildfitfcn()
            del self.__dict__['pickled_models']
        except:
            if self.debug:
                warnings.warn('unable to unpickle fit function; it is omitted')
        return self 

    def formatall(self, *args, **kargs):
        " Add-on method for fits returned by chained_lsqfit. "
        ans = ''
        for x in self.chained_fits:
            ans += 10 * '=' + ' ' + str(x) + '\n'
            ans += self.chained_fits[x].format(*args, **kargs)
            ans += '\n'
        return ans[:-1]

    def show_plots(self, save=False, view='ratio'):
        fitdata = collections.OrderedDict()
        fitval = collections.OrderedDict()
        for k in self.chained_fits:
            # there is only one k in chained_fits but this is easiest way
            if k[:5] == 'wavg(' and k[-1] == ')':
                continue
            fit = self.chained_fits[k]
            # need OrderedDict conversion because otherwise
            # converts to dict, which doesn't work (BufferDict issue)
            fitdata.update(collections.OrderedDict(fit.data))
            fitval.update(collections.OrderedDict(fit.fcn(self.p)))
        MultiFitter.show_plots(
            fitdata=fitdata, fitval=fitval, save=save, view=view,
            )

    def bootstrapped_fit_iter(
        self, n=None, datalist=None, pdatalist=None, **kargs
        ):
        return MultiFitter._bootstrapped_fit_iter(
            self.fitter_args_kargs,
            n=n, datalist=datalist, pdatalist=pdatalist, **kargs
            )

class chained_nonlinear_fit(lsqfit.nonlinear_fit):
    " Fit results from chained fit. "
    def __init__(self, p, chained_fits, multifitter, prior, fitter_args_kargs):
        if len(chained_fits) <= 0:
            raise ValueError('no chained fits')
        self._p = p
        self.palt = p
        self.pmean = gvar.mean(p)
        self.psdev = gvar.sdev(p)
        self.chained_fits = chained_fits
        self.fitter_args_kargs = fitter_args_kargs
        self.fitter_args_kargs[1]['p0'] = self.pmean

        # extract fcn, fcn values, and data from fits and fitter
        # (for format(...))
        self.fcn = multifitter.buildfitfcn()
        self.data = collections.OrderedDict()
        self.prior = prior
        self.fcn_p = collections.OrderedDict()
        for k in self.chained_fits:
            if k[:5] == 'wavg(' and k[-1] == ')':
                continue
            fit = self.chained_fits[k]
            self.data.update(fit.data)
            self.fcn_p.update(fit.fcn(fit.p))
        self.data = gvar.BufferDict(self.data)
        self.x = False
        self.y = self.data

        self.linear = []
        self.correction = 0
        self.svdn = 0
        self.dof = 0
        self.chi2 = 0
        self.nit = 0
        self.time = 0
        self.svdcut = None
        self.eps = None
        self.tol = [0., 0., 0.]
        self.error = []
        self.logGBF = 0.
        self.noise = (False, False)
        self.fitter = 'chained fit'
        self.description = ''
        self.stopping_criterion = None
        self.residuals = []
        self.p0 = []
        for k in self.chained_fits:
            self.correction += self.chained_fits[k].correction
            if k[:5] == 'wavg(' and k[-1] == ')':
                continue
            self.residuals.extend(self.chained_fits[k].residuals)
            self.svdn += self.chained_fits[k].svdn
            self.dof += self.chained_fits[k].dof
            self.chi2 += self.chained_fits[k].chi2
            self.nit += self.chained_fits[k].nit
            self.time += self.chained_fits[k].time
            self.p0.append(self.chained_fits[k].p0)
            svdcut = self.chained_fits[k].svdcut
            eps = self.chained_fits[k].eps
            if self.chained_fits[k].noise[0]:
                self.noise = (True, self.noise[1])
            if self.chained_fits[k].noise[1]:
                self.noise = (self.noise[0], True)
            if svdcut is not None:
                if self.svdcut is None:
                    self.svdcut = svdcut
                elif abs(svdcut) > abs(self.svdcut):
                    self.svdcut = svdcut
            if eps is not None:
                if self.eps is None:
                    self.eps = eps 
                elif eps > self.eps:
                    self.eps = eps
            tol = self.chained_fits[k].tol
            for i in range(3):
                self.tol[i] = max(self.tol[i], tol[i])
            error = self.chained_fits[k].error
            if error is not None:
                self.error.append(error)
            logGBF = self.chained_fits[k].logGBF
            if logGBF is not None:
                self.logGBF += logGBF
            if self.chained_fits[k].stopping_criterion == 0:
                self.stopping_criterion = 0
        self.residuals = numpy.array(self.residuals)
        if len(self.error) == 0:
            self.error = None
        self.tol = tuple(self.tol)
        self.Q = lsqfit.gammaQ(self.dof/2., self.chi2/2.)
        if self.logGBF == 0:
            self.logGBF = None

        # others
        self.cov = None
        self.fitter_results = None
        self.nblocks = None
        self.svdcorrection = self.correction # legacy name

    def _remove_gvars(self, gvlist):
        self.p  # need to fill _p
        fit = copy.copy(self)
        fitter,args,kargs = fit.fitter_args_kargs
        try:
            fit.pickled_models = pickle.dumps(args['models'])
        except:
            if self.debug:
                warn.warnings('unable to pickle fit function; it is omitted')
        for k in ['fcn', 'fitter_args_kargs']:
            del fit.__dict__[k]
        fit.__dict__ = gvar.remove_gvars(fit.__dict__, gvlist)
        return fit
    
    def _distribute_gvars(self, gvlist):
        self.__dict__ = gvar.distribute_gvars(self.__dict__, gvlist)
        try:
            models = pickle.loads(self.pickled_models)
            self.fcn = MultiFitter(models).buildfitfcn()
            del self.__dict__['pickled_models']
        except:
            if self.debug:
                warn.warnings('unable to unpickle fit function; it is omitted')
        return self 

    def simulated_fit_iter(self, **kargs):
        raise NotImplementedError('use with individual fits in self.chained_fits')

    def show_plots(self, save=False, view='ratio'):
        fitdata = collections.OrderedDict()
        fitval = collections.OrderedDict()
        for k in self.chained_fits:
            if k[:5] == 'wavg(' and k[-1] == ')':
                continue
            fit = self.chained_fits[k]
            # need OrderedDict conversion because otherwise
            # converts to dict, which doesn't work (BufferDict issue)
            fitdata.update(collections.OrderedDict(fit.data))
            fitval.update(collections.OrderedDict(fit.fcn(self.p)))
        MultiFitter.show_plots(
            fitdata=fitdata, fitval=fitval, save=save, view=view,
            )

    def format(self, *args, **kargs):
        def make_line(k, f):
            try:
                Q = '{:.2f}'.format(f.Q)
            except:
                Q = ''
            try:
                logGBF = '{:.5g}'.format(f.logGBF)
            except:
                logGBF = ''
            k = str(k)
            return '\n' + fmt.format(
                '{:10.2g} [{}]'.format(f.chi2 / f.dof, f.dof),
                Q, logGBF, f.svdn,
                '{}/{:.1f}'.format(f.nit, f.time), k
                )
        ans = super(chained_nonlinear_fit, self).format(*args, **kargs)
        fmt = '{:>16}{:>8}{:>11}  {:>5}   {:>10}  {:<}'
        header = fmt.format(
            'chi2/dof [dof]', 'Q', 'logGBF', 'svd-n', 'itns/time','fit            '
            )
        ans += '\nChained Fits:\n' + header
        ans += '\n' + len(header) * '-'
        fmt = '{:>16}{:>8}{:>11}  {:5} {:>12}  {:<}'
        for k in self.chained_fits:
            ans += make_line(k, self.chained_fits[k])
        ans += '\n' + len(header) * '-'
        ans += make_line('all', self)
        ans += '\n'
        return ans

    def formatall(self, *args, **kargs):
        " Add-on method for fits returned by chained_nonlinear_fit. "
        ans = ''
        for x in self.chained_fits:
            ans += 10 * '=' + ' ' + str(x) + '\n'
            ans += self.chained_fits[x].format(*args, **kargs)
            ans += '\n'
        return ans[:-1]

    def bootstrapped_fit_iter(
        self, n=None, datalist=None, pdatalist=None, **kargs
        ):
        return MultiFitter._bootstrapped_fit_iter(
            self.fitter_args_kargs,
            n=n, datalist=datalist, pdatalist=pdatalist, **kargs
            )

class MultiFitter(object):
    """ Nonlinear least-squares fitter for a collection of models.

    Fits collections of data that are modeled by collections of models.
    Fits can be simultaneous (:meth:`lsqfit.MultiFitter.lsqfit`) or chained
    (:meth:`lsqfit.MultiFitter.chained_lsqfit`).

    Args:
        models: List of models, derived from :mod:`lsqfit.MultiFitterModel`,
            to be fit to the data. Individual models in the list can
            be replaced by lists of models or tuples of models; see below.
        mopt (object): Marginalization options. If not ``None``,
            marginalization is used to reduce the number of fit parameters.
            Object ``mopt`` is passed to the models when constructing the
            prior for a fit; it typically indicates the degree of
            marginalization (in a model-dependent fashion). Setting
            ``mopt=None`` implies no marginalization.
        ratio (bool): If ``True``, implement marginalization using
            ratios: ``data_marg = data * fitfcn(prior_marg) / fitfcn(prior)``.
            If ``False`` (default), implement using differences:
            ``data_marg = data + (fitfcn(prior_marg) - fitfcn(prior))``.
        fast (bool): Setting ``fast=True`` (default) strips any variable
            not required by the fit from the prior. This speeds
            fits but loses information about correlations between
            variables in the fit and those that are not. Setting
            ``wavg_all=True`` can restore some of the correlations, but
            is somewhat slower.
        wavg_all (bool): If ``True`` and ``fast=True``, the final result of a
            chained fit is the weighted average of all the fits in the chain.
            This can restore correlations lost in the chain because
            ``fast=True``. This step is omitted if ``wavg_all=False`` or
            ``fast=False``. Default is ``False``.
        fitname (callable or ``None``): Individual fits in a chained fit are
            assigned default names, constructed from the datatags of
            the corresponding models, for access and reporting. These names
            get unwieldy when lots of models are involved. When ``fitname``
            is not ``None`` (default), each default name ``dname`` is
            replaced by ``fitname(dname)`` which should return a string.
        wavg_kargs (dict): Keyword arguments for :meth:`lsqfit.wavg` when
            used to combine results from parallel sub-fits in a chained fit.
        fitterargs (dict): Additional arguments for the
            :class:`lsqfit.nonlinear_fit` object used to do the fits.
            These can be collected in a dictionary (e.g.,
            ``fitterargs=dict(tol=1e-6, maxit=500))``) or listed as
            separate arguments (e.g., ``tol=1e-6, maxit=500``).
    """

    def __init__(
        self, models, mopt=None, ratio=False, fast=True, wavg_all=False,
        wavg_kargs=dict(eps=1e-12), fitname=None, fitterargs={},
        **more_fitterargs
        ):
        super(MultiFitter, self).__init__()
        models = [models] if isinstance(models, MultiFitterModel) else models
        self.models = models
        self.fit = None         # last fit
        self.ratio = ratio
        self.mopt = mopt
        self.fast = fast
        self.wavg_all = wavg_all
        self.wavg_kargs = wavg_kargs
        self.fitterargs = dict(fitterargs)
        self.fitterargs.update(more_fitterargs)
        self.tasklist = self._compile_models(models)
        self.flatmodels = self._flatten_models(self.tasklist)
        self.fitname = (
            fitname if fitname is not None else
            lambda x : str(x)
            )

    def set(self, **kargs):
        """ Reset default keyword parameters.

        Assigns new default values from dictionary ``kargs`` to the fitter's
        keyword parameters. Keywords for the underlying :mod:`lsqfit` fitters
        can also be  included (or grouped together in dictionary
        ``fitterargs``).

        Returns tuple ``(kargs, oldkargs)`` where ``kargs`` is a dictionary
        containing all :class:`lsqfit.MultiFitter` keywords after they have
        been updated, and ``oldkargs`` contains the  original values for these
        keywords. Use ``fitter.set(**oldkargs)`` to restore the original
        values.
        """
        kwords = set([
            'mopt', 'fast', 'ratio', 'wavg_kargs', 'wavg_all',
            'fitterargs', 'fitname',
            ])
        kargs = dict(kargs)
        oldkargs = {}
        fargs = {}
        # changed
        for k in list(kargs.keys()):  # list() needed since changing kargs
            if k in kwords:
                oldkargs[k] = getattr(self, k)
                setattr(self, k, kargs[k])
                kwords.remove(k)
            else:
                fargs[k] = kargs[k]
                del kargs[k]
        # unchanged
        for k in kwords:
            kargs[k] = getattr(self, k)
        # manage fitterargs
        if 'fitterargs' in kwords:
            # means wasn't in kargs initially
            oldkargs['fitterargs'] = self.fitterargs
            self.fitterargs = dict(self.fitterargs)
        if len(fargs) > 0:
            self.fitterargs.update(fargs)
        kargs['fitterargs'] = dict(self.fitterargs)
        return kargs, oldkargs

    def buildfitfcn(self):
        """ Create fit function to fit models in list ``models``. """
        # def _fitfcn(p, flatmodels=self.flatmodels):
        #     ans = gvar.BufferDict()
        #     for m in flatmodels:
        #         ans[m.datatag] = (
        #             m.fitfcn(p) if m.ncg <= 1 else
        #             MultiFitter.coarse_grain(m.fitfcn(p), m.ncg)
        #             )
        #     return ans
        return _multifitfcn(self.flatmodels)

    def builddata(self, mopt=None, data=None, pdata=None, prior=None):
        """ Rebuild pdata to account for marginalization. """
        if pdata is None:
            if data is None:
                raise ValueError('no data or pdata')
            pdata = gvar.BufferDict()
            for m in self.flatmodels:
                pdata[m.datatag] = (
                    m.builddata(data) if m.ncg <= 1 else
                    MultiFitter.coarse_grain(m.builddata(data), m.ncg)
                    )
        else:
            npdata = gvar.BufferDict()
            for m in self.flatmodels:
                npdata[m.datatag] = pdata[m.datatag]
            pdata = npdata
        if mopt is not None:
            fitfcn = self.buildfitfcn()
            p_all = self.buildprior(prior=prior, mopt=None)
            f_all = fitfcn(p_all)

            # fcn with part we want to keep
            p_trunc = self.buildprior(prior=prior, mopt=mopt)
            f_trunc = fitfcn(p_trunc)

            # correct pdata
            pdata = gvar.BufferDict(pdata)
            if not self.ratio:
                for m in self.flatmodels:
                    pdata[m.datatag] += f_trunc[m.datatag] - f_all[m.datatag]
            else:
                for m in self.flatmodels:
                    ii = (gvar.mean(f_all[m.datatag]) != 0)
                    ratio = f_trunc[m.datatag][ii] / f_all[m.datatag][ii]
                    pdata[m.datatag][ii] *= ratio
        return pdata

    def buildprior(self, prior, mopt=None):
        """ Create prior to fit models in list ``models``. """
        nprior = gvar.BufferDict()
        for m in self.flatmodels:
            nprior.update(m.buildprior(
                prior, mopt=mopt,
                ))
        if not self.fast:
            for k in prior:
                if k not in nprior:
                    nprior[k] = prior[k]
        return nprior

    @staticmethod
    def _flatten_models(tasklist):
        " Create 1d-array containing all disctinct models from ``tasklist``. "
        ans = gvar.BufferDict()
        for task, mlist in tasklist:
            if task != 'fit':
                continue
            for m in mlist:
                id_m = id(m)
                if id_m not in ans:
                    ans[id_m] = m
        return ans.buf.tolist()

    @staticmethod
    def flatten_models(models):
        " Create 1d-array containing all disctinct models from ``models``. "
        if isinstance(models, MultiFitterModel):
            ans = [models]
        else:
            tasklist = MultiFitter._compile_models(models)
            ans = MultiFitter._flatten_models(tasklist)
        return ans

    def lsqfit(self, data=None, pdata=None, prior=None, p0=None, chained=False, **kargs):
        """ Compute least-squares fit of models to data.

        :meth:`MultiFitter.lsqfit` fits all of the models together, in
        a single fit. It returns the |nonlinear_fit| object from the fit.

        To see plots of the fit data divided by the fit function
        with the best-fit parameters use

            fit.show_plots()

        This method has optional keyword arguments ``save`` and ``view``;
        see documentation for :class:`lsqfit.MultiFitter.show_plots`
        for more information. Plotting requires module :mod:`matplotlib`.

        To bootstrap a fit, use ``fit.bootstrapped_fit_iter(...)``;
        see :meth:`lsqfit.nonlinear_fit.bootstrapped_fit_iter` for more
        information.

        Args:
            data: Input data. One of ``data`` or ``pdata`` must be
                specified but not both. ``pdata`` is obtained from ``data``
                by collecting the output from ``m.builddata(data)``
                for each model ``m`` and storing it in a dictionary
                with key ``m.datatag``.
            pdata: Input data that has been processed by the
                models using :meth:`MultiFitter.process_data` or
                :meth:`MultiFitter.process_dataset`. One of
                ``data`` or ``pdata`` must be  specified but not both.
            prior (dict): Bayesian prior for fit parameters used by the models.
            p0: Dictionary , indexed by parameter labels, containing
                initial values for the parameters in the fit. Setting
                ``p0=None`` implies that initial values are extracted from the
                prior. Setting ``p0="filename"`` causes the fitter to look in
                the file with name ``"filename"`` for initial values and to
                write out best-fit parameter values after the fit (for the
                next call to ``self.lsqfit()``).
            chained (bool): Use :meth:`MultiFitter.chained_lsqfit`
                instead of :meth:`MultiFitter.lsqfit` if ``chained=True``. 
                Ignored otherwise. Default is ``chained=False``.
            kargs: Arguments that (temporarily) override parameters specified
                when the :class:`MultiFitter` was created. Can also include
                additional arguments to be passed through to the :mod:`lsqfit`
                fitter.
        """
        # chained?
        if chained:
            return self.chained_lsqfit(data=data, pdata=pdata, prior=prior, p0=p0, **kargs)
        # gather parameters
        if prior is None:
            raise ValueError('no prior')
        kargs, oldargs = self.set(**kargs)

        # save parameters for bootstrap (in case needed)
        fitter_args_kargs = (
            self.lsqfit,
            dict(data=data, prior=prior, pdata=pdata, models=self.models),
            dict(kargs),
            )

        # build prior, data and function
        fitprior = self.buildprior(prior=prior, mopt=self.mopt)
        fitdata = self.builddata(
            mopt=self.mopt, data=data, pdata=pdata, prior=prior
            )
        fitfcn = self.buildfitfcn()

        # build name
        if len(self.flatmodels) > 1:
            fname = self.fitname(
                '(' +
                ','.join([self.fitname(k.datatag) for k in self.flatmodels])
                + ')'
                )
        else:
            fname = self.fitname(self.flatmodels[0].datatag)

        # read in p0 if in file (can't leave to nonlinear_fit)
        if isinstance(p0, str):
            p0file = p0 
        else:
            # check whether p0 is a list of strings (from chained fits)
            try:
                if isinstance(p0[0], str):
                    p0file = p0[0]
                else:
                    p0file = None
            except:
                p0file = None
        if p0file is not None:
            try:
                with open(p0file, 'rb') as ifile:
                    _p0 = pickle.load(ifile)
            except (IOError, EOFError):
                _p0 = None
        else:
            _p0 = p0 
        if _p0 is not None and not hasattr(_p0, 'keys'):
            # _p0 is a list from a chained fit
            _p0list = _p0 
            _p0 = _p0list[0]
            for p in _p0list[1:]:
                _p0.update(p)
        
        # fit
        self.fit = unchained_nonlinear_fit(
            fname=fname, fitter_args_kargs=fitter_args_kargs,
            data=fitdata, prior=fitprior, fcn=fitfcn, p0=_p0,
            **self.fitterargs
            )

        # manage p0
        if p0file is not None:
            _p0 = self.fit.pmean
            with open(p0file, 'wb') as ofile:
                pickle.dump(_p0, ofile)

        # restore default keywords
        self.set(**oldargs)
        return self.fit

    def chained_lsqfit(
        self, data=None, pdata=None, prior=None, p0=None,
        **kargs
        ):
        """ Compute chained least-squares fit of models to data. Equivalent to::
        
            self.lsqfit(data, pdata, prior, p0, chained=True, **kargs).

        In a chained fit to models ``[s1, s2, ...]``, the models are fit one
        at a time, with the fit output from one being fed into the prior for
        the next. This can be much faster than  fitting the models together,
        simultaneously. The final result comes from the last fit in the chain,
        and includes parameters from all of the models.

        The most general chain has the structure ``[s1, s2, s3 ...]``
        where each ``sn`` is one of:

            1) A model (derived from :class:`multifitter.MultiFitterModel`).

            2) A tuple ``(m1, m2, m3)`` of models, to be fit together in
                a single fit (i.e., simultaneously). Simultaneous fits
                are useful for closely related models.

            3) A list ``[p1, p2, p3 ...]`` where each ``pn`` is either
                a model, a tuple of models (see #2), or a dictionary (see #4).
                The ``pn`` are fit separately: the fit output from one fit is
                *not* fed into the prior of the next (i.e., the fits are
                effectively in parallel). Results from the separate fits are
                averaged at the end to provide a single composite result for
                the collection of fits. Parallel fits are effective (and fast)
                when the different fits have few or no fit parameters in
                common.

            4) A dictionary that (temporarily) resets default values for
                fitter keywords. The new values, specified in the dictionary,
                apply to subsequent fits in the chain. Any number of such
                dictionaries can be included in the model chain.


        Fit results are returned in a 
        :class:`lsqfit.MultiFitter.chained_nonlinear_fit` object ``fit``,
        which is very similar to a :class:`nonlinear_fit`
        object (see documentation for more information). Object ``fit`` has an
        extra attribute ``fit.chained_fits`` which is an ordered dictionary
        containing fit results for each link in the chain of fits, indexed by
        fit names built from the corresponding data tags.

        To list results from all of the fits in the chain, use ::

            print(fit.formatall())

        This method has optional keyword arguments ``maxline``,
        ``pstyle``, and ``nline``; see the documentation for
        :meth:`lsqfit.nonlinear_fit.format` for more
        information.

        To view plots of each fit use

            fit.show_plots()

        This method has optional keyword arguments ``save`` and ``view``;
        see documentation for :class:`lsqfit.MultiFitter.show_plots`
        for more information. Plotting requires module :mod:`matplotlib`.

        To bootstrap a fit, use ``fit.bootstrapped_fit_iter(...)``;
        see :meth:`lsqfit.nonlinear_fit.bootstrapped_fit_iter` for more
        information.

        Args:
            data: Input data. One of ``data`` or ``pdata`` must be
                specified but not both. ``pdata`` is obtained from ``data``
                by collecting the output from ``m.builddata(data)``
                for each model ``m`` and storing it in a dictionary
                with key ``m.datatag``.
            pdata: Input data that has been processed by the
                models using :meth:`MultiFitter.process_data` or
                :meth:`MultiFitter.process_dataset`. One of
                ``data`` or ``pdata`` must be  specified but not both.
            prior: Bayesian prior for fit parameters used by the models.
            p0: Dictionary , indexed by parameter labels, containing
                initial values for the parameters in the fit. Setting
                ``p0=None`` implies that initial values are extracted from the
                prior. Setting ``p0="filename"`` causes the fitter to look in
                the file with name ``"filename"`` for initial values and to
                write out best-fit parameter values after the fit (for the
                next call to ``self.chained_lsqfit()``). Finally,
                ``p0`` can be a list containing a different ``p0`` for each 
                fit in the chain: for example, ::

                    p0 = [f.pmean for f in fit.chained_fits.values()]

                might be a good starting point for the next fit.
            kargs: Arguments that override parameters specified when
                the :class:`MultiFitter` was created. Can also include
                additional arguments to be passed through to
                the :mod:`lsqfit` fitter.
        """
        if prior is None:
            raise ValueError('no prior')
        # prior is shared by many subfits so need to handle prior noise here
        if 'noise' in kargs:
            if isinstance(kargs['noise'], bool):
                kargs['noise'] = (kargs['noise'], kargs['noise'])
            if kargs['noise'][1]:
                prior = prior + (gvar.sample(prior) - gvar.mean(prior))
                kargs['noise'] = (kargs['noise'][0], False)
        elif kargs.get('add_priornoise', False):
            prior = prior + (gvar.sample(prior) - gvar.mean(prior))
            kargs['add_priornoise'] = False
        kargs, oldargs = self.set(**kargs)

        # parameters for bootstrap (see below)
        fitter_args_kargs = (
            self.chained_lsqfit,
            dict(data=data, prior=prior, pdata=pdata, models=self.models),
            dict(kargs),
            )

        # local copy of prior
        if self.fast:
            prior = self.buildprior(prior)
        else:
            prior = gvar.BufferDict(prior)

        # read in p0 if in file
        p0file = p0 if isinstance(p0, str) else None
        if p0file is not None:
            try:
                with open(p0file, 'rb') as ifile:
                    _p0 = pickle.load(ifile)
            except (IOError, EOFError):
                _p0 = None
        else:
            _p0 = p0 
        p0_iter = iter([_p0]) if (hasattr(_p0, 'keys') or _p0 is None) else iter(_p0)
        next_p0 = None

        # execute tasks in self.tasklist
        chained_fits = collections.OrderedDict()
        all_fnames = []
        all_fitp = []
        for tasktype, taskdata in self.tasklist:
            if tasktype == 'fit':
                fitter = self.__class__(models=taskdata, **kargs)
                next_p0 = next(p0_iter, next_p0)
                fit = fitter.lsqfit(
                    data=data, pdata=pdata, prior=prior, p0=next_p0
                    )
                fname = list(fit.chained_fits.keys())[0]
                if fname in chained_fits:
                    raise ValueError('duplicate fits in chain: ' + str(fname))
                elif fname[:5] == 'wavg(' and fname[-1] == ')':
                    raise ValueError('bad fit name: ' + fname)
                else:
                    all_fnames.append(fname)
                    chained_fits[fname] = fit
                    all_fitp.append(fit.p)
            elif tasktype == 'update-prior':
                lastfit = chained_fits[all_fnames[-1]]
                lastfit_p = lastfit.p
                for k in lastfit_p:
                    idx = tuple(
                        slice(None, i) for i in numpy.shape(lastfit.p[k])
                        )
                    if idx != ():
                        prior[k][idx] = lastfit.p[k]
                    else:
                        prior[k] = lastfit.p[k]
            elif tasktype == 'wavg':
                    if taskdata <= 1:
                        continue
                    nlist = all_fnames[-taskdata:]
                    plist = [chained_fits[k].p for k in nlist]
                    wavg_kargs = kargs.get('wavg_kargs', self.wavg_kargs)
                    fit = lsqfit.wavg(plist, **wavg_kargs).fit
                    fname = self.fitname('wavg({})'.format(','.join(nlist)))
                    all_fnames.append(fname)
                    chained_fits[fname] = fit
            elif tasktype == 'update-kargs':
                kargs.update(taskdata)
            else:
                raise RuntimeError('unknown task: ' + tasktype)

        if self.fast and self.wavg_all:
            wavg_kargs = kargs.get('wavg_kargs', self.wavg_kargs)
            fit = lsqfit.wavg(all_fitp, **wavg_kargs).fit
            fname = self.fitname('wavg(all)')
            chained_fits[fname] = fit
            prior = fit.p

        # p0 management
        if p0file is not None:
            _p0 = []
            for k in chained_fits:
                _p0.append(chained_fits[k].pmean)
            with open(p0file, 'wb') as ofile:
                pickle.dump(_p0, ofile)

        # build output class
        self.fit = chained_nonlinear_fit(
            p=prior, chained_fits=chained_fits,
            multifitter=self, 
            prior=fitter_args_kargs[1]['prior'],
            fitter_args_kargs=fitter_args_kargs,
            )

        # restore default keywords
        self.set(**oldargs)
        return self.fit

    def empbayes_fit(self, z0, fitargs, p0=None, **minargs):
        """ Return fit and ``z`` corresponding to the fit
        ``self.lsqfit(**fitargs(z))`` that maximizes ``logGBF``.

        This function maximizes the logarithm of the Bayes Factor from
        fit  ``self.lsqfit(**fitargs(z))`` by varying ``z``,
        starting at ``z0``. The fit is redone for each value of ``z``
        that is tried, in order to determine ``logGBF``.

        The Bayes Factor is proportional to the probability that the data
        came from the model (fit function and priors) used in the fit.
        :meth:`MultiFitter.empbayes_fit` finds the model or data that maximizes this
        probability. See :func:`lsqfit.empbayes_fit` for more information.

        Include ``chained=True`` in the dictionary returned by ``fitargs(z)``
        if chained fits are desired. See documentation 
        for :meth:`MultiFitter.lsqfit`.

        Args:
            z0 (number, array or dict): Starting point for search.
            fitargs (callable): Function of ``z`` that returns a
                dictionary ``args`` containing the :meth:`MultiFitter.lsqfit`
                arguments corresponding to ``z``. ``z`` should have
                the same layout (number, array or dictionary) as ``z0``.
                ``fitargs(z)`` can instead return a tuple ``(args, plausibility)``,
                where ``args`` is again the dictionary for
                :meth:`MultiFitter.lsqfit`. ``plausibility`` is the logarithm
                of the *a priori* probabilitiy that ``z`` is sensible. When
                ``plausibility`` is provided, :func:`MultiFitter.empbayes_fit`
                maximizes the sum ``logGBF + plausibility``. Specifying
                ``plausibility`` is a way of steering selections away from
                completely implausible values for ``z``.
            p0: Fit-parameter starting values for the first fit. ``p0``
                for subsequent fits is set automatically to optimize fitting
                unless a value is specified by ``fitargs``.
            minargs (dict): Optional argument dictionary, passed on to
                :class:`lsqfit.gsl_multiminex` (or
                :class:`lsqfit.scipy_multiminex`), which finds the minimum.

        Returns:
            A tuple containing the best fit (a fit object) and the
            optimal value for parameter ``z``.
        """
        return empbayes_fit(z0, fitargs, p0=p0, fitter=self.lsqfit, **minargs)

    @staticmethod
    def _compile_models(models):
        """ Convert ``models`` into a list of tasks.

        Each task is tuple ``(name, data)`` where ``name`` indicates the task
        task and ``data`` is the relevant data for that task.

        Supported tasks and data:

            - ``'fit'`` and list of models
            -  ``'update-kargs'`` and ``None``
            - ``'update-prior'`` and ``None``
            - ``'wavg'`` and number of (previous) fits to average

        """
        tasklist = []
        for m in models:
            if isinstance(m, MultiFitterModel):
                tasklist += [('fit', [m])]
                tasklist += [('update-prior', None)]
            elif hasattr(m, 'keys'):
                tasklist += [('update-kargs', m)]
            elif isinstance(m, tuple):
                tasklist += [('fit', list(m))]
                tasklist += [('update-prior', None)]
            elif isinstance(m, list):
                nfit = 0
                for sm in m:
                    if isinstance(sm, MultiFitterModel):
                        tasklist += [('fit', [sm])]
                        nfit += 1
                    elif isinstance(sm, tuple):
                        tasklist += [('fit', list(sm))]
                        nfit += 1
                    elif hasattr(sm, 'keys'):
                        tasklist += [('update-kargs', sm)]
                    else:
                        raise ValueError(
                            'type {} not allowed in sublists '.format(
                                str(type(sm))
                                )
                            )
                if nfit > 0:
                    tasklist += [('wavg', nfit)]
                    tasklist += [('update-prior', None)]
            else:
                raise RuntimeError('bad model list')
        return tasklist

    def bootstrapped_fit_iter(
        self, n=None, datalist=None, pdatalist=None, **kargs
        ):
        # for legacy code; use fit.bootstrapped_fit_iter instead
        warnings.warn(
            'MultiFitter.bootstrapped_fit_iter is deprecated; use fit.bootstrapped_fit_iter instead',
            DeprecationWarning
            )
        return self.fit.bootstrapped_fit_iter(
            n=n, datalist=datalist, pdatalist=pdatalist, **kargs
            )

    @staticmethod
    def _bootstrapped_fit_iter(
        fitter_args_kargs, n=None, datalist=None, pdatalist=None, **kargs
        ):
        """ Iterator that returns bootstrap copies of a fit.

        Bootstrap iterator for |MultiFitter| fits analogous to
        :meth:`lsqfit.bootstrapped_fit_iter`. The bootstrap uses the
        same parameters as the last fit done by the fitter unless they
        are overridden by ``kargs``.

        Args:
            n (int): Maximum number of iterations if ``n`` is not ``None``;
                otherwise there is no maximum. Default is ``None``.
            datalist (iter): Collection of bootstrap data sets for fitter.
            pdatalist (iter): Collection of bootstrap processed data sets for
                fitter.
            kargs (dict): Overrides arguments in original fit.

        Returns:
            Iterator that returns an |nonlinear_fit| object
            containing results from the fit to the next data set in
            ``datalist``.

        """
        fitter, args, okargs = fitter_args_kargs
        for k in okargs:
            if k not in kargs:
                kargs[k] = okargs[k]
        if 'p0' not in kargs:
            kargs['p0'] = args['p0']
        if datalist is not None:
            pdatalist = (
                MultiFitter.process_data(d, args['models']) for d in datalist
                )
        elif pdatalist is None:
            pdata = args['pdata']
            if pdata is None:
                pdata = MultiFitter.process_data(args['data'], args['models'])
            pdatalist = gvar.bootstrap_iter(pdata, n)
        i = 0
        for pdata in pdatalist:
            i += 1
            if n is not None and i > n:
                break
            fit = fitter(pdata=pdata, prior=args['prior'], **kargs)
            yield fit

    bootstrap_iter = bootstrapped_fit_iter   # legacy

    @staticmethod
    def coarse_grain(G, ncg):
        """ Coarse-grain last index of array ``G``.

        Bin the last index of array ``G`` in bins of width ``ncg``, and
        replace each bin by its average. Return the binned results.

        Args:
            G: Array to be coarse-grained.
            ncg: Bin width for coarse-graining.
        """
        if ncg <= 1:
            return G
        G = numpy.asarray(G)
        nbin, remainder = divmod(G.shape[-1], ncg)
        if remainder != 0:
            nbin += 1
        return numpy.transpose([
            numpy.sum(G[..., i:i+ncg], axis=-1) / G[..., i:i+ncg].shape[-1]
            for i in numpy.arange(0, ncg * nbin, ncg)
            ])

    @staticmethod
    def process_data(data, models):
        """ Convert ``data`` to processed data using ``models``.

        Data from dictionary ``data`` is processed by each model
        in list ``models``, and the results collected into a new
        dictionary ``pdata`` for use in :meth:`MultiFitter.lsqfit`
        and :meth:`MultiFitter.chained_lsqft`.
        """
        pdata = gvar.BufferDict()
        for m in MultiFitter.flatten_models(models):
            pdata[m.datatag] = (
                m.builddata(data) if m.ncg <= 1 else
                MultiFitter.coarse_grain(m.builddata(data), ncg=m.ncg)
                )
        return pdata

    @staticmethod
    def process_dataset(dataset, models, **kargs):
        """ Convert ``dataset`` to processed data using ``models``.

        :class:`gvar.dataset.Dataset` (or similar dictionary) object
        ``dataset`` is processed by each model in list ``models``,
        and the results collected into a new dictionary ``pdata`` for use in
        :meth:`MultiFitter.lsqfit` and :meth:`MultiFitter.chained_lsqft`.
        Assumes that the models have defined method
        :meth:`MultiFitterModel.builddataset`. Keyword arguments
        ``kargs`` are passed on to :func:`gvar.dataset.avg_data` when
        averaging the data.
        """
        dset = collections.OrderedDict()
        for m in MultiFitter.flatten_models(models):
            dset[m.datatag] = (
                m.builddataset(dataset) if m.ncg <= 1 else
                MultiFitter.coarse_grain(m.builddataset(dataset), ncg=m.ncg)
                )
        return gvar.dataset.avg_data(dset, **kargs)

    @staticmethod
    def show_plots(fitdata, fitval, x=None, save=False, view='ratio'):
        """ Show plots comparing ``fitdata[k],fitval[k]`` for each key ``k`` in ``fitval``.

        Assumes :mod:`matplotlib` is installed (to make the plots). Plots
        are shown for one correlator at a time. Press key ``n`` to see the
        next correlator; press key ``p`` to see the previous one; press key
        ``q`` to quit the plot and return control to the calling program;
        press a digit to go directly to one of the first ten plots. Zoom,
        pan and save using the window controls.

        There are several different views available for each plot,
        specified by parameter ``view``:

            ``view='ratio'``: Data divided by fit (default).

            ``view='diff'``: Data minus fit, divided by data's standard deviation.

            ``view='std'``: Data and fit.

            ``view='log'``: ``'std'`` with log scale on the vertical axis.

            ``view='loglog'``: `'std'`` with log scale on both axes.

        Press key ``v`` to cycle through these  views; or press keys
        ``r``, ``d``, or ``l`` for the ``'ratio'``, ``'diff'``,
        or ``'log'`` views, respectively.

        Copies of the plots that are viewed can be saved by setting parameter
        ``save=fmt`` where ``fmt`` is a string used to create
        file names: the file name for the plot corresponding to key
        ``k`` is ``fmt.format(k)``. It is important that the
        filename end with a suffix indicating the type of plot file
        desired: e.g., ``fmt='plot-{}.pdf'``.
        """
        import matplotlib.pyplot as plt
        # collect plotinfo
        plotinfo = collections.OrderedDict()
        for tag in fitval:
            d = fitdata[tag]
            f = fitval[tag]
            plotinfo[tag] = (
                numpy.arange(len(d))+1 if x is None else x[tag],
                gvar.mean(d), gvar.sdev(d),  gvar.mean(f), gvar.sdev(f)
                )
        plotinfo_keys = list(plotinfo.keys())
        fig = plt.figure()
        viewlist = ['ratio', 'diff', 'std', 'log', 'loglog']
        def onpress(event):
            if event is not None:
                try:    # digit?
                    onpress.idx = int(event.key)
                except ValueError:
                    if event.key == 'n':
                        onpress.idx += 1
                    elif event.key == 'p':
                        onpress.idx -= 1
                    elif event.key == 'v':
                        onpress.view = (onpress.view + 1) % len(viewlist)
                    elif event.key == 'r':
                        onpress.view = viewlist.index('ratio')
                    elif event.key == 'd':
                        onpress.view = viewlist.index('diff')
                    elif event.key == 'l':
                        onpress.view = viewlist.index('log')
                    # elif event.key == 'q':  # unnecessary
                    #     plt.close()
                    #     return
                    else:
                        return
            else:
                onpress.idx = 0

            # do the plot
            if onpress.idx >= len(plotinfo_keys):
                onpress.idx = len(plotinfo_keys)-1
            elif onpress.idx < 0:
                onpress.idx = 0
            i = onpress.idx
            k = plotinfo_keys[i]
            x, g, dg, gth, dgth = plotinfo[k]
            fig.clear()
            plt.title("%d) %s   (press 'n', 'p', 'q', 'v' or a digit)"
                        % (i, k))
            dx = (max(x) - min(x)) / 50.
            plt.xlim(min(x)-dx, max(x)+dx)
            plotview = viewlist[onpress.view]
            if plotview in ['std', 'log', 'loglog']:
                if plotview in ['log', 'loglog']:
                    plt.yscale('log', nonpositive='clip')
                if plotview == 'loglog':
                    plt.xscale('log', nonpositive='clip')
                plt.ylabel(str(k) + '   [%s]' % plotview)
                if len(x) > 0:
                    if len(x) > 1:
                        plt.plot(x, gth, 'r-')
                        plt.fill_between(
                            x, y2=gth + dgth, y1=gth - dgth,
                            color='r', alpha=0.075,
                            )
                    else:
                        extra_x = [x[0] * 0.5, x[0] * 1.5]
                        plt.plot(extra_x, 2 * [gth[0]], 'r-')
                        plt.fill_between(
                            extra_x, y2=2 * [gth[0] + dgth[0]],
                            y1=2 * [gth[0] - dgth[0]],
                            color='r', alpha=0.075,
                            )
                    plt.errorbar(x, g, dg, fmt='o')
            elif plotview == 'ratio':
                plt.ylabel(str(k)+' / '+'fit'  + '   [%s]' % plotview)
                ii = (gth != 0.0)       # check for exact zeros (eg, antiperiodic)
                if len(x[ii]) > 0:
                    if len(x[ii]) > 1:
                        plt.fill_between(
                            x[ii], y2=1 + dgth[ii] / gth[ii],
                            y1=1 - dgth[ii] / gth[ii],
                            color='r', alpha=0.075,
                            )
                        plt.plot(x, numpy.ones(len(x), float), 'r-')
                    else:
                        extra_x = [x[ii][0] * 0.5, x[ii][0] * 1.5]
                        plt.fill_between(
                            extra_x, y2=2 * [1 + dgth[ii][0]/gth[ii][0]],
                            y1=2 * [1 - dgth[ii][0]/gth[ii][0]],
                            color='r', alpha=0.075,
                            )
                        plt.plot(extra_x, numpy.ones(2, float), 'r-')
                    plt.errorbar(x[ii], g[ii]/gth[ii], numpy.fabs(dg[ii]/gth[ii]), fmt='o')
            elif plotview == 'diff':
                plt.ylabel('({} - fit) / sigma'.format(str(k))  + '   [%s]' % plotview)
                ii = (dg != 0.0)       # check for exact zeros
                if len(x[ii]) > 0:
                    if len(x[ii]) > 1:
                        plt.fill_between(
                            x[ii], y2=dgth[ii] / dg[ii],
                            y1=-dgth[ii] / dg[ii],
                            color='r', alpha=0.075
                            )
                        plt.plot(x, numpy.zeros(len(x), float), 'r-')
                    else:
                        extra_x = [x[ii][0] * 0.5, x[ii][0] * 1.5]
                        plt.fill_between(
                            extra_x, y2=2 * [dgth[ii][0] / dg[ii][0]],
                            y1=2 * [-dgth[ii][0] / dg[ii][0]],
                            color='r', alpha=0.075
                            )
                        plt.plot(extra_x, numpy.zeros(2, float), 'r-')
                    plt.errorbar(
                        x[ii], (g[ii] - gth[ii]) / dg[ii], dg[ii] / dg[ii],
                        fmt='o'
                        )
            if save:
                plt.savefig(save.format(k), bbox_inches='tight')
            else:
                plt.draw()
        onpress.idx = 0
        try:
            onpress.view = viewlist.index(view)
        except ValueError:
            raise ValueError('unknow view: ' + str(view))
        fig.canvas.mpl_connect('key_press_event', onpress)
        onpress(None)
        plt.show()


class _multifitfcn(object):
    " MultiFitter fit function. "
    def __init__(self, flatmodels):
        self.flatmodels = flatmodels 
    
    def __call__(self, p):
        # def _fitfcn(p, flatmodels=self.flatmodels):
        ans = gvar.BufferDict()
        for m in self.flatmodels:
            ans[m.datatag] = (
                m.fitfcn(p) if m.ncg <= 1 else
                MultiFitter.coarse_grain(m.fitfcn(p), m.ncg)
                )
        return ans


class vegas_fit(object):
    r""" Least-squares fit using Bayesian integrals. 

    :class:`lsqfit.vegas_fit` fits a (nonlinear) function 
    ``f(x,p)`` (or ``f(p)``) to data ``y`` using Bayesian 
    integrals over fit parameters ``p``. Typical usage is ::

        vfit = vegas_fit(data=(x,y), fcn=f, prior=prior)
        print(vfit)
        print('best-fit parameters =', vfit.p)

    The fitter calculates the means and (co)variances of 
    the fit parameters (``vfit.p``) assuming that the parameters 
    are described by a probability density function (PDF) 
    proportional to :math:`\exp(-\chi^2(p)/2)` where

    .. math::

        \chi^2(p) = \Delta y \cdot\mathrm{cov}^{-1}_y \cdot \Delta y
        \: + \: 
        \Delta p \cdot\mathrm{cov}^{-1}_\mathrm{prior}\cdot\Delta p

    and :math:`\Delta y_i \equiv \overline y_i - f(x_i,p)` 
    and :math:`\Delta p_i\equiv \overline p_i - p_i`. This involves a 
    multi-dimensional integration over the parameter space using
    :class:`vegas.PDFIntegrator` from the :mod:`vegas` module  
    (which must be installed separately). :mod:`vegas` uses 
    adaptive Monte Carlo integration to obtain estimates for the 
    integrals; see its documentation for more information.

    When the PDF is sufficiently peaked around its maximum, 
    :math:`\chi^2(p)` is (usually) well approximated by a quadratic
    expansion around its minimum, and results obtained from 
    :mod:`lsqfit.vegas_fit` will agree with those obtained from 
    :mod:`lsqfit.nonlinear_fit` --- the latter is the Gaussian 
    approximation to the former. The output from ``nonlinear_fit``
    can often be used to improve significantly the accuracy of 
    the numerical integrals used by ``vegas_fit``, particularly 
    if the PDF is sharply peaked and there are lots of parameters: 
    for example, by setting  ``param=fit.p`` in  ::

        fit = nonlinear_fit(data=(x,y), fcn=f, prior=prior)
        vfit = vegas_fit(data=(x,y), fcn=f, prior=prior, param=fit.p)
        print(vfit)

    we direct ``vegas_fit`` to re-express the integrals over 
    ``p`` in terms of variables that emphasize the region 
    indicated by ``fit.p``. This facilitates the integration 
    and can greatly reduce the numerical uncertainties in the 
    results. Note that the second line in this code snippet can 
    be written more succinctly as ::

        vfit = vegas_fit(fit=fit)

    :mod:`vegas` adapts iteratively to the PDF, averaging results 
    over the iterations. By default it uses 10 iterations to 
    train the integrator on the PDF, and then 10 more, without 
    further adaptation, to estimate the means and covariances 
    of the fit parameters. During the training stage, the 
    integrator remaps the integrals to variables that emphasize
    regions where the PDF is large, refining the map after 
    each iteration. Integral results from the training 
    stage are often unreliable and so are discarded. 
    (Adaptation is turned off for the latter iterations 
    to provide more robust estimates; see the :mod:`vegas` 
    documentation for more information.) The number of evaluations
    of ``f(x,p)`` is limited to at most 1000 by default. Both 
    the number of iterations and the number of function evaluations
    can be specified using parameters ``nitn`` and ``neval``, 
    respectively: for example ::

        vfit = vegas_fit(fit=fit, nitn=(5,10), neval=10_000)

    specifies 5 iterations for adapting to the PDF (training) 
    followed by 10 more for computing ``vfit.p``, with at most 
    10,000 function evaluations per iteration. The number of 
    function evaluations needed depends upon the number of 
    parameters and how sharply peaked the PDF is. 

    Having obtained a fit ``vfit`` from ``vegas_fit``,  
    expectation values with respect to the PDF can be obtained
    for any function ``g(p)`` of the parameters, where ``g(p)`` 
    typically returns an array of numbers or a dictionary whose 
    values are numbers or arrays of numbers, so that multiple 
    expectation values can be computed together::

        s = vfit.stats(g)
    
    where ``s`` is an array or dictionary of |GVar|\s
    giving the means and covariances of the components of ``g(p)``
    with respect to the PDF. Result ``s`` agrees well with 
    ``g(vfit.p)`` when the Gaussian approximation is valid
    (for the PDF and ``g(p)``), but could be quite different 
    otherwise. ``vfit.stats`` can also calculate other 
    statistical parameters (e.g., skewness) and/or histograms
    for the distribution of ``g(p)`` values.

    Args:

        data (dict, array or tuple):
            Data to be fit by :class:`lsqfit.vegas_fit`
            can have either of the following forms:

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

            Setting ``x=False`` in the first of these formats implies
            that the fit function depends only on the fit parameters:
            that is, ``fcn(p)`` instead of ``fcn(x, p)``. (This is not assumed
            if ``x=None``.) Ignored if parameter ``fit`` is specified.

        fcn (callable): The function to be fit to ``data``. It is either a
            function of the independent data ``x`` and the fit parameters ``p``
            (``fcn(x, p)``), or a function of just the fit parameters
            (``fcn(p)``) when there is no ``x`` data or ``x=False``. The
            function's return value must have the same layout as the ``y`` data
            (a dictionary or an array). The fit parameters ``p`` are either: 1)
            a dictionary where each ``p[k]`` is a single parameter or an array
            of parameters (any shape); or, 2) a single array of parameters. The
            layout of the parameters is the same as that of prior ``prior`` if
            it is specified; otherwise, it is inferred from ``param``.

            ``vegas_fit`` is usually much faster if ``fcn`` is designed to 
            process a large batch of integration points all at once. See
            the :mod:`vegas` documentation on ``rbatchintegrand`` and 
            ``lbatchintegrand``.

        prior: A dictionary or array of |GVar|\s representing  
            *a priori* estimates for all parameters ``p`` used by
            fit function ``fcn(x, p)`` (or ``fcn(p)``). Fit parameters ``p``
            are stored in a dictionary or array with the same keys and
            structure (or shape) as ``prior``. The default value is ``None``;
            ``prior`` must be defined if ``param`` is ``None``.
            Ignored if parameter ``fit`` is specified.
    
        param: A dictionary or array of |GVar|\s that specifies the fit 
            parameters ``p`` used by ``fcn(x,p)`` (or ``fcn(p)``), and 
            indicates where in that parameter space the integrator should 
            focus its attention. Fit parameters ``p`` are stored in a 
            dictionary or array with the same keys and structure (or shape) 
            as ``param`` (and ``prior``, if specified). ``vegas_fit`` 
            re-expresses the parameter integrals in terms of variables that 
            emphasize the region of parameter space covered by 
            ``param``. Setting ``param=None`` (the default) is 
            equivalent to setting ``param=prior``; ``prior`` 
            must be defined if ``param=None``.
            Ignored if parameter ``fit`` is specified.

        fit: Fit results from either :class:`lsqfit.nonlinear_fit` or 
            :class:`lsqfit.vegas_fit`. When ``fit`` is specified, the 
            data, prior, and fit function are take from ``fit`` and 
            ``param=fit.p`` is set. The fit function from ``fit`` can 
            be replaced by setting the ``fcn`` parameter (for example, 
            to replace ``fit.fcn`` by an equivalent batch function). 

        svdcut (float): If nonzero, singularities in the correlation
            matrix for ``y`` and ``prior`` are regulated using 
            :func:`gvar.regulate` with an SVD cutoff ``svdcut``. This 
            makes the correlation matrices less singular, which 
            can improve the  stability and accuracy of a fit. 
            Default is ``svdcut=1e-12``. Ignored if parameter ``fit``
            is specified.

        eps (float): If positive, singularities in the correlation 
            matrix for ``y`` and ``prior`` are regulated using 
            :func:`gvar.regulate` with cutoff ``eps``. This makes 
            the correlation matrices less singular, which can 
            improve the  stability and accuracy of a fit. 
            Ignored if ``svdcut`` is specified (and not ``None``).
            Ignored if parameter ``fit`` is specified.

        noise (tuple or bool): If ``noise[0]=True``, noise is 
            added to the data means commensurate with the additional
            uncertainties (if any) introduced by using ``svdcut>0`` 
            or ``eps>0``. If ``noise[1]=True``, noise is added 
            to the prior means commensurate with the uncertainties
            in the prior. Noise is useful for testing the
            quality of a fit (``chi2``). Setting ``noise=True`` 
            is shorthand for ``noise=(True, True)``, and
            ``noise=False`` means ``noise=(False, False)`` (the default).
            Ignored if parameter ``fit`` is specified.

        vegasargs: Any additional keyword argments are passed to 
            the integrator, :class:`vegas.PDFIntegrator`. The most 
            important of these arguments are the number of ``vegas`` 
            interations ``nitn`` and the maximum number ``neval`` of 
            integrand evaluations per iteration. Default values for 
            these are ``nitn=(10,10)`` and ``neval=1000``, where 
            nitn[0] is the number of iterations used to train the
            integrator to the PDF, and nitn[1] is the number of 
            iterations used to determine the means and covariances
            of the parameters. (``vegas`` adapts to the PDF in 
            the first set of (training) iterations; adaptation is 
            turned off for the second.)

    Objects of type :class:`lsqfit.nonlinear_fit` have the following
    attributes:

    Attributes:

        chi2: :math:`\chi^2(p)` evaluated at ``vfit.p``.
            ``fit.chi2 / fit.dof`` is usually of order one in good fits.
            Values much less than one suggest that actual fluctuations in
            the input data and/or priors might be smaller than suggested
            by the standard deviations (or covariances) used in the fit.

        dof: Number of degrees of freedom in the fit, which equals
            the number of pieces of data being fit when priors are specified
            for the fit parameters. Without priors, it is the number of pieces
            of data minus the number of fit parameters.

        integrator: The :class:`vegas.PDFIntegrator` used to do the 
            integrals.

        logBF: The logarithm of the probability (density)
            of obtaining the fit data by randomly sampling the parameter model
            (priors plus fit function) used in the fit --- that is, it is
            the logarithm of ``P(data|model)``. This quantity is useful for 
            comparing fits of the same data to different models, with 
            different priors and/or fit functions. The model with the 
            largest value of ``fit.logBF`` is the one preferred by the 
            data. The exponential of the difference in ``fit.logBF`` between 
            two models is the ratio of probabilities (Bayes factor) for 
            those models. Differences in ``fit.logBF`` smaller than 1 
            are not very significant.

        p: Best-fit parameters from fit in the same format as the 
            prior (array or dictionary containing |GVar|\s). The 
            means and uncertainties (standard deviations/covariances) 
            are calculated from the Bayesian integrals. The 
            uncertainties come from the distribution and from 
            uncertainties in ``vegas``'s estimates of the 
            means (added in quadrature). 
            
            ``vfit.p`` is output from :meth:`vegas.PDFIntegrator.stats`.
            It has additional attributes that provide more information
            about the integrals. See the documentation for 
            :class:`vegas.PDFEV`, :class:`PDFEVArray`, and 
            :class:`vegas.PDFEVDict` for more information.

        pmean: An array or dictionary containing the means of the 
            best-fit parameters from fit.

        psdev: Standard deviations of the best-fit
            parameters from fit.

        palt: Same as ``vfit.p``.
        
        prior: Prior used in the fit. This may differ from the 
            input prior if an SVD cut is set (it is the prior 
            after the SVD cut). It is either a  dictionary 
            (:class:`gvar.BufferDict`) or an array 
            (:class:`numpy.ndarray`), depending upon the input. 
            Equals ``None`` if no prior is specified.

        Q: The probability that the ``chi**2`` from the fit
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
            be uncorrelated from each other. The residuals include 
            contributions from both the fit data and the prior. 
            They are related to the the ``chi**2`` of the fit by:
            ``chi2 = sum(fit.residuals**2)``.

        training: Results returned by the :class:`vegas.PDFIntegrator` 
            after evaluating the integrals used to train the integrator
            to the PDF. These results are not included in the final 
            averages.            
        
        correction: Sum of all corrections, if any, added
            to the fit data and prior when ``eps>0`` or ``svdcut>0``.

        svdn: Number of eigenmodes of the correlation matrix 
            modified (and/or deleted) when ``svdcut>0``.

        time: CPU time (in secs) taken by fit.

        x: The first field in the input ``data``. This is sometimes the
            independent variable (as in 'y vs x' plot), but may be anything.
            It is set equal to ``False`` if the ``x`` field is omitted from
            the input ``data``. (This also means that the fit function has no
            ``x`` argument: so ``f(p)`` rather than ``f(x,p)``.)

        y: Fit data used in the fit, with |GVar| for each data point. 
            This may differ rom the input data if an SVD cut is used 
            (it is ``y`` after the SVD cut). It is either a dictionary 
            (:class:`gvar.BufferDict`) or an array (:class:`numpy.ndarray`), 
            depending upon the input.

        nblocks: ``nblocks[s]`` equals the number of block-diagonal
            sub-matrices of the ``y``--``prior`` covariance matrix that are
            size ``s``-by-``s``. This is sometimes useful for debugging.

    """
    def __init__(
            self, data=None, fcn=None, prior=None, fit=None, param=None, 
            svdcut=False, eps=False, noise=None, 
            nitn=(10,10), **kargs
            ):
        clock = time.perf_counter if hasattr(time, 'perf_counter') else time.time
        cpu_time = clock()
        try:
            import vegas 
        except ImportError:
            raise ImportError('Install the vegas module to use vegas_fit.')
        if numpy.size(nitn) == 1:
            nitn = 2 * (nitn,)
        self.nitn = nitn
        if fit is None:
            fcn = self._build_pdf(
                data=data, fcn=fcn, prior=prior, param=param, svdcut=svdcut, 
                eps=eps, noise=noise
                )
        elif isinstance(fit, vegas_fit):
            # flatfcn already defined so don't need fcn
            self._extract_pdf_vegas_fit(fit=fit, param=param)
        elif isinstance(fit, lsqfit.nonlinear_fit):
            fcn = self._extract_pdf_nonlinear_fit(fit=fit, fcn=fcn, param=param)

        # create integrator and fit function
        
        self.integrator = vegas.PDFIntegrator(self.param, pdf=None, **kargs)
        if not hasattr(self, 'flatfcn'):
            self.flatfcn = self.integrator._make_std_integrand(fcn, xsample=gvar.mean(self.param)) 

        # pdf
        self.chivkargs = {}
        for k in ['inv_wgts', 'nw', '_flat_y_prior_mean', 'minchi2']:
            self.chivkargs[k] = getattr(self, k) 
        self.chivkargs['jac'] = None 
        self.chivkargs['flatfcn_eval'] = self.flatfcn.eval
        pdf = vegas.lbatchintegrand(functools.partial(
            vegas_fit._pdf, **self.chivkargs
            ))
        self.integrator.set(pdf=pdf)

        self.training = self.integrator(nitn=nitn[0], adapt=True)
        self.p = self.integrator.stats(
            f=None, nitn=nitn[1], adapt=kargs.get('adapt', False),
            )
        # collect results
        self.pmean = gvar.mean(self.p) 
        self.psdev = gvar.sdev(self.p)
        self.palt = self.p 
        # self.pdfnorm = self.p.pdfnorm

        self.fitter = 'vegas_fit'
        self.description = '({})'.format(gvar.fmt_chi2(self.p))

        # calculate chi2
        if hasattr(self.pmean, 'keys'):
            # remove extra keys for chi**2 calculation
            tmp = gvar.BufferDict(self.pmean)
            extrakeys = []
            for k in tmp:
                if k not in self.param:
                    extrakeys.append(k)
            for k in extrakeys:
                del tmp[k]
            p = gvar.BufferDict(tmp, lbatch_buf=tmp.buf.reshape(1, -1))
        elif numpy.ndim(self.pmean) == 0:
            p = numpy.array([self.pmean])
        else:
            p = self.pmean.reshape(1, -1)
        self.residuals = self.chiv(p)[0]
        self.chi2 = numpy.sum(self.residuals ** 2)
        self.Q = lsqfit.gammaQ(self.dof/2., self.chi2/2.)
        if self._delta_logBF is not None:
            self.logBF = numpy.log(self.p.pdfnorm) + self._delta_logBF
        self.time = clock() - cpu_time

    def _extract_pdf_vegas_fit(self, fit, param, **kargs):
        for v in ['data', 'fcn', 'x', 'y', 'prior', 'minchi2', '_delta_logBF', 'dof', 'svdcut', 'svdn', 'correction', 'eps', 'noise', 'nw', 'nblocks', 'flatfcn', '_flat_y_prior_mean', 'inv_wgts']:
            if hasattr(fit, v):
                setattr(self, v, getattr(fit, v))
        if param is None:
            if fit.param.shape is None:
                self.param = gvar.BufferDict()
                for k in fit.param:
                    self.param[k] = fit.p[k]
            else:
                self.param = fit.p
        else:
            self.param = lsqfit._unpack_gvars(param)
    
    def _extract_pdf_nonlinear_fit(self, fit, fcn, param, **kargs):
        for v in ['data', 'x', 'y', 'prior', 'dof', 'svdcut', 'correction', 'svdn', 'eps', 'noise', 'nblocks']:
            if hasattr(fit, v):
                setattr(self, v, getattr(fit, v))
        self.minchi2 = fit.chi2
        if param is None:
            self.param = fit.p
        else:
            self.param = lsqfit._unpack_gvars(param)
        self.nw = fit.yp_pdf.nchiv
        self.inv_wgts = fit.yp_pdf.i_invwgts 
        self._flat_y_prior_mean = fit.yp_pdf.mean
        fcn = fit.fcn if fcn is None else fcn
        self.fcn = fcn
        if self.fcn is None:
            raise ValueError('must specify fcn')
        # attach x to the function
        if self.x is not False:
            fcn = functools.partial(self.fcn, self.x)
            fcn.fcntype = getattr(self.fcn, 'fcntype', 'scalar')

        self._delta_logBF = - 0.5 * (
            fit.yp_pdf.logdet
            + numpy.log(2*numpy.pi) * numpy.size(self._flat_y_prior_mean)
            + self.minchi2
            ) 
        return fcn 
    
    def _build_pdf(self, data, fcn, prior, param, svdcut, eps, noise, **kargs):
        """ pdf=None means must construct it here """
        if eps is False:
            eps = lsqfit.nonlinear_fit.DEFAULTS.get(
                'eps', lsqfit._FITTER_DEFAULTS['eps']
                )
        if svdcut is False:
            svdcut = lsqfit.nonlinear_fit.DEFAULTS.get(
                'svdcut', lsqfit._FITTER_DEFAULTS['svdcut'],
                )
        if noise is None:
            noise = lsqfit.nonlinear_fit.DEFAULTS.get(
                'noise', lsqfit._FITTER_DEFAULTS['noise'],
                )
        if isinstance(noise, bool):
            noise = (noise, noise)
        self.noise = noise
        self.eps = eps 
        self.svdcut = svdcut 
        self.minchi2 = 0.0

        if data is None:
            raise ValueError('must specify data')
        self.data = data
        if isinstance(self.data, tuple):
            self.x, self.y = self.data 
        else:
            self.x = False 
            self.y = self.data
        self.y = lsqfit._unpack_gvars(self.y)

        self.fcn = fcn 
        if self.fcn is None:
            raise ValueError('must specify fcn')
        # attach x to the function
        if self.x is not False:
            fcn = functools.partial(fcn, self.x)
            fcn.fcntype = getattr(self.fcn, 'fcntype', 'scalar')

        self.prior = lsqfit._unpack_gvars(prior)
        if param is None:
            if self.prior is None:
                raise ValueError('must specify one of prior or param')
            self.param = self.prior
        else:
            self.param = lsqfit._unpack_gvars(param)
        if noise[1]:
            self.prior = self.prior + gvar.sample(self.prior) - gvar.mean(self.prior) 

        # create PDF components
        # 1) create the data array (incl prior)
        if self.prior is not None:
            flat_y_prior = numpy.concatenate((self.y.flat[:], self.prior.flat[:]))
        else:
            flat_y_prior = self.y.flat[:]

        # 2) apply svd cut, and calculate chi**2 weights
        flat_y_prior, self.inv_wgts = gvar.regulate(flat_y_prior, svdcut=svdcut, eps=eps, wgts=-1, noise=noise[0])
        self.nw = sum([len(wgts) for _,wgts in self.inv_wgts])
        self.nblocks = flat_y_prior.nblocks
        self.dof = self.nw - self.param.size
        self.svdcut = flat_y_prior.svdcut 
        self.eps = flat_y_prior.eps
        self.svdn = flat_y_prior.nmod
        self.correction = flat_y_prior.correction

        if self.svdn > 0:
            if hasattr(self.y, 'keys'):
                self.y = gvar.BufferDict(self.y, buf=flat_y_prior[:self.y.size])
            else:
                self.y = flat_y_prior[:numpy.size(self.y)].reshape(self.y.shape)
            if hasattr(self.prior, 'keys'):
                self.prior = gvar.BufferDict(self.prior, buf=flat_y_prior[self.y.size:])
            elif self.prior is not None:
                self.prior = flat_y_prior[numpy.size(self.y):].reshape(self.prior.shape)
                
        # 3) separate mean and set up logBF
        self._flat_y_prior_mean = gvar.mean(flat_y_prior)
        self._delta_logBF = - 0.5 * (
            flat_y_prior.logdet
            + numpy.log(2*numpy.pi) * numpy.size(flat_y_prior)
            )
        return fcn

    def __str__(self):
        return self.format()
    
    def format(self, maxline=0, pstyle='v'):
        """ Format the output from a :class:`lsqfit.vegas_fit`.
        
        See the documentation for :meth:`lsqfit.nonlinear_fit.format` 
        for more information.
        """
        return lsqfit.nonlinear_fit.format(self, maxline=maxline, pstyle='v')
    
    def qqplot_residuals(self, plot=None):
        """ Create QQ plot of the fit residuals.
        
        See the documentation for :meth:`lsqfit.nonlinear_fit.qqplot_residuals` 
        for more information.
        """
        return lsqfit.nonlinear_fit.qqplot_residuals(self, plot=plot)

    def plot_residuals(self, plot=None):
        """ Create QQ plot of the fit residuals.
        
        See the documentation for :meth:`lsqfit.nonlinear_fit.plot_residuals` 
        for more information.        
        """
        return lsqfit.nonlinear_fit.plot_residuals(self, plot=plot)

    def _remove_gvars(self, gvlist):
        fit = copy.copy(self)
        try:
            # if can pickle fcn then keep everything
            fit.pickled_fcn = gvar.dumps((self.fcn, self.flatfcn))
            for k in ['fcn', 'flatfcn']:
                del fit.__dict__[k]
        except:
            warnings.warn('unable to pickle fit function; it is omitted')
        fit.__dict__ = gvar.remove_gvars(fit.__dict__, gvlist)
        return fit
    
    def _distribute_gvars(self, gvlist):
        self.__dict__ = gvar.distribute_gvars(self.__dict__, gvlist)
        try:
            # try restoring fit function
            self.fcn, self.flatfcn = gvar.loads(self.pickled_fcn)
            del self.__dict__['pickled_fcn']
        except:
            warnings.warn('unable to unpickle fit function; it is omitted')
        return self 
        
    def stats(self, f=None, *args, **kargs):
        """ Means and standard deviations (and covariances) of function ``f(p)``.
        
        If ``f`` is set to ``None`` or omitted, the means and 
        standard deviations (and covariances) of the fit parameters 
        are recalculated.

        See documentation for :meth:`vegas.PDFIntegrator.stats` for 
        further options.
        """
        return self.integrator.stats(f, *args, **kargs)  

    def sample(self, nbatch, mode='rbatch'):
        """ Generate random samples from PDF used in fit.

        See documentation for :meth:`vegas.PDFIntegrator.sample` for more 
        information.

        Args:
            nbatch (int): The integrator will return
                at least ``nbatch`` samples drawn from its PDF. The 
                actual number of samples is the smallest multiple of 
                ``self.last_neval`` that is equal to or larger than ``nbatch``.
                Results are packaged in arrays or dictionaries
                whose elements have an extra index labeling the different 
                samples in the batch. The batch index is 
                the rightmost index if ``mode='rbatch'``; it is 
                the leftmost index if ``mode`` is ``'lbatch'``. 
            mode (bool): Batch mode. Allowed 
                modes are ``'rbatch'`` or ``'lbatch'``,
                corresponding to batch indices that are on the 
                right or the left, respectively. 
                Default is ``mode='rbatch'``.
        
        Returns:
            A tuple ``(wgts,samples)`` containing samples drawn from the integrator's
            PDF, together with their weights ``wgts``. The weighted sample points 
            are distributed through parameter space with a density proportional to
            the PDF. 
            
            In general, ``samples`` is either a dictionary or an array 
            depending upon the format of :class:`lsqfit.vegas_fit` 
            parameter ``param``. For example, if ::

                param = gv.gvar(dict(s='1.5(1)', v=['3.2(8)', '1.1(4)']))

            then ``samples['s'][i]`` is a sample for parameter ``p['s']``
            where index ``i=0,1...nbatch(approx)`` labels the sample. The 
            corresponding sample for ``p['v'][d]``, where ``d=0`` or ``1``, 
            is ``samples['v'][d, i]`` provided ``mode='rbatch'``, which 
            is the default. (Otherwise it is ``p['v'][i, d]``, for 
            ``mode='lbatch'``.) The corresponding weight for this sample
            is ``wgts[i]``.

            When ``param`` is an array, ``samples`` is an array with the same 
            shape plus an extra sample index which is either on the right 
            (``mode='rbatch'``, default) or left (``mode='lbatch'``).        
        """
        return self.integrator.sample(nbatch=nbatch, mode=mode)

    def chiv(self, p, jac=None):
        r""" :math:`\chi^2(p)` 
        
        Note ``chi2(p) == numpy.sum(chiv(p)**2, axis=1)``. 
        """
        # lbatchintegrand
        # jac ignored because useless (esp if have correlations)
        return vegas_fit._chiv(p, **self.chivkargs)
    
    @staticmethod
    def _chiv(p, jac, inv_wgts, nw, flatfcn_eval, _flat_y_prior_mean, minchi2):
        """ chi2(p) = numpy.sum(chiv(p)**2, axis=1) """
        # minchi2 not used here
        if p.shape is None:
            p = p.lbatch_buf
        elif p.ndim != 2:
            p = p.reshape(p.shape[0], -1)
        chiv = numpy.zeros((p.shape[0], nw), dtype=float)
        ffp = flatfcn_eval(p, jac)
        if ffp.shape[1] == _flat_y_prior_mean.shape[0]:
            delta = ffp - _flat_y_prior_mean[None, :] # lbatch index
        else:
            delta = numpy.concatenate((ffp, p), axis=1) - _flat_y_prior_mean[None, :] # lbatch index
        iw, wgt = inv_wgts[0]
        if len(iw) > 0:
            chiv[:, iw] = wgt[None, :] * delta[:, iw]
        for iw, wgt in inv_wgts[1:]:
            chiv[:, iw] = delta[:, iw].dot(wgt.T)
        return chiv

    def pdf(self, p, jac=None):
        r""" Probability density function for fit parameters ``p``.
        
        When the |vegas_fit| object is derived from a |nonlinear_fit|
        object (``vfit = vegas_fit(fit=fit)``), the PDF is 
        :math:`\exp(-(\chi^2(p) - \chi^2_\mathrm{min})/2)`. Otherwise
        it is just :math:`\exp(-\chi^2(p)/2)`. In either case it 
        is *not* normalized. Divide by ``vfit.pdfnorm`` to normalize 
        it.
        """
        # jac ignored
        # an lbatchintegrand
        return vegas_fit._pdf(p, **self.chivkargs)
    
    @staticmethod
    def _pdf(p, **kargs):
        # an lbatchintegrand 
        minchi2 = kargs['minchi2']
        return numpy.exp(- 0.5 * numpy.sum(vegas_fit._chiv(p, **kargs) ** 2, axis=1) + minchi2/2)
        
def fake_fitargs(fcn, prior=None, p0=None, x=False, fsig=0.001, correlate=True, noise=True):
    r""" Creates fake fit arguments corresponding to ``fcn(x,p)`` for testing fitters.

    Creates simulated fit data ``(x,y)`` (or ``y`` if ``x is False``) where 
    ``y`` is Gaussian with mean ``f(x, p0)`` (or ``fcn(p0)`` if ``x is False``),
    where ``p0`` are the fit parameters and ``x`` the independent variables.
    The ``y`` standard deviations are ``fsig * abs(fcn(x, p))``, where ``fsig=0.001`` 
    by default. The data are correlated if ``correlate=True`` (default). 
    When ``noise=True``, the mean values are replaced by random values 
    drawn at random from the distribution with ``noise=False``. Finally
    if ``p0`` is not specified, it is set to ``p0 = gvar.sample(prior)``.

    The function returns a dictionary of fit argments for 
    :class:``lsqfit.nonlinear_fit`` corresponding to 
    arguments ``data``, ``fcn``, and ``prior`` and/or ``p0``.
    Typical usage is::

        fitargs = fake_fitargs(fcn, prior, x)
        fit = lsqfit.nonlinear_fit(**fitargs)

    With default settings, the output fit parameters ``fit.p`` should 
    agree with input values ``p0`` to within errors. The fit should 
    have a ``fit.chi2 / fit.dof`` close to 1. 

    Args:
        fcn (callable): Fit function is ``fcn(x,p)`` 
            or ``fcn(p)`` if ``x is False``. The mean
            for the simulated fit data ``y`` equals 
            ``fcn(c,p0)`` before noise is added (if ``noise=True``).
        prior: Dictionary or array of |GVar|\s representing 
            the priors for the fit parameters.
        p0: Dictionary or array of numbers representing fit parameters.
        x: Independent fit variable or ``False`` if there is none. 
            Default is ``False``.
        fsig (float): The standard deviations of the simulated 
            data ``y`` equal ``fsig * abs(f(x, p0))``. Default
            is ``fsig = 0.001``.
        correlate (bool): Uncertainties in different simulated 
            data points are correlated if ``correlate=True``; 
            they are uncorrelated otherwise.
        noise (bool): If ``noise=True``, Gaussian noise is added 
            to the means of the simulated fit data ``y`` (default).
            Ignored otherwise.

    Returns:
        Fit arguments dictionary::

            dict(data=(x,y), fcn=fcn, prior=prior, p0=p0)

        The ``prior`` entry is dropped if no prior is specified,
        and ``data=y`` if ``x=False`` is specified.  
    """
    if p0 is None:
        if prior is None:
            raise ValueError('at least one of p0 or prior must be specified')
        else:
            p0 = gvar.sample(prior)
    else:
        p0 = gvar.asbufferdict(p0) if hasattr(p0, 'keys') else numpy.asarray(p0)
    ymean = fcn(p0) if x is False else fcn(x, p0) 
    ymean = gvar.asbufferdict(ymean) if hasattr(ymean, 'keys') else numpy.asarray(ymean)
    ysig = (fsig if fsig is not None else 0.001) * numpy.fabs(ymean.flat[:])
    ysig[ysig<=0] = min(ysig[ysig>0])
    if ymean.shape is None:
        ysig = gvar.BufferDict(ymean, buf=ysig)
    else:
        ysig.shape = ymean.shape
    y = gvar.gvar(0 * ymean, ysig)
    if correlate == True:
        ny = ymean.size
        corr = gvar.RNG.uniform(0.1, 0.9, (ny, ny))
        corr = corr @ corr.T 
        d = (1 / numpy.diag(corr)) ** 0.5 
        y.flat[:] = gvar.correlate(y.flat[:], d[:, None] * d[None, :] * corr) 
    if noise:
        y += ymean + gvar.sample(y)
    else:
        y += ymean
    data = y if x is None else (x, y) 
    if prior is None:
        return dict(data=data, fcn=fcn, p0=p0) 
    else:
        return dict(data=data, fcn=fcn, p0=p0, prior=prior) 