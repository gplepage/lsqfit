""" part of lsqfit module: extra functions  """

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-18 G. Peter Lepage.
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
import pickle
import time
import types
import warnings

import numpy

import lsqfit
import gvar

try:
    from ._gsl import gsl_multiminex as _multiminex
    from ._gsl import gammaQ as _gammaQ
except ImportError:
    from ._scipy import scipy_multiminex as _multiminex
    from ._scipy import gammaQ as _gammaQ

def empbayes_fit(z0, fitargs, **minargs):
    """ Return fit and ``z`` corresponding to the fit
    ``lsqfit.nonlinear_fit(**fitargs(z))`` that maximizes ``logGBF``.

    This function maximizes the logarithm of the Bayes Factor from
    fit  ``lsqfit.nonlinear_fit(**fitargs(z))`` by varying ``z``,
    starting at ``z0``. The fit is redone for each value of ``z``
    that is tried, in order to determine ``logGBF``.

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
        print fit.format(True)

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
        minargs (dict): Optional argument dictionary, passed on to
            :class:`lsqfit.gsl_multiminex` (or
            :class:`lsqfit.scipy_multiminex`), which finds the minimum.

    Returns:
        A tuple containing the best fit (object of type
        :class:`lsqfit.nonlinear_fit`) and the
        optimal value for parameter ``z``.
    """
    save = dict(lastz=None, lastp0=None)
    if hasattr(z0, 'keys'):
        # z is a dictionary
        if not isinstance(z0, gvar.BufferDict):
            z0 = gvar.BufferDict(z0)
        z0buf = z0.buf
        def convert(zbuf):
            return gvar.BufferDict(z0, buf=zbuf)
    elif numpy.shape(z0) == ():
        # z is a number
        z0buf = numpy.array([z0])
        def convert(zbuf):
            return zbuf[0]
    else:
        # z is an array
        z0 = numpy.asarray(z0)
        z0buf = z0
        def convert(zbuf):
            return zbuf
    def minfcn(zbuf, save=save, convert=convert):
        z = convert(zbuf)
        args = fitargs(z)
        if not hasattr(args, 'keys'):
            args, plausibility = args
        else:
            plausibility = 0.0
        if save['lastp0'] is not None:
            args['p0'] = save['lastp0']
        fit = lsqfit.nonlinear_fit(**args)
        if numpy.isnan(fit.logGBF):
            raise ValueError
        else:
            save['lastz'] = z
            save['lastp0'] = fit.pmean
        return -fit.logGBF - plausibility
    try:
        z = convert(_multiminex(z0buf, minfcn, **minargs).x)
    except ValueError:
        print('*** empbayes_fit warning: null logGBF')
        z = save['lastz']
    args = fitargs(z)
    if not hasattr(args, 'keys'):
        args, plausibility = args
    if save['lastp0'] is not None:
        args['p0'] = save['lastp0']
    return lsqfit.nonlinear_fit(**args), z


class GVarWAvg(gvar.GVar):
    """ Result from weighted average :func:`lsqfit.wavg`.

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

    .. attribute:: svdcorrection

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
            self.svdcorrection = gvar.gvar(0,0)
            self.time = 0
            self.fit = None
        else:
            self.chi2 = fit.chi2
            self.dof = fit.dof
            self.Q = fit.Q
            self.time = fit.time
            self.svdcorrection = fit.svdcorrection
            self.fit = fit

class ArrayWAvg(numpy.ndarray):
    """ Result from weighted average :func:`lsqfit.wavg`.

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

    .. attribute:: svdcorrection

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
            obj.svdcorrection = gvar.gvar(0, 0)
        else:
            obj.chi2 = fit.chi2
            obj.dof = fit.dof
            obj.Q = fit.Q
            obj.time = fit.time
            obj.svdcorrection = fit.svdcorrection
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
    """ Result from weighted average :func:`lsqfit.wavg`.

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

    .. attribute:: svdcorrection

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
            self.svdcorrection = gvar.gvar(0,0)
            self.time = 0
            self.fit = None
        else:
            self.chi2 = fit.chi2
            self.dof = fit.dof
            self.Q = fit.Q
            self.time = fit.time
            self.svdcorrection = fit.svdcorrection
            self.fit = fit

def wavg(dataseq, prior=None, fast=False, **fitterargs):
    """ Weighted average of |GVar|\s or arrays/dicts of |GVar|\s.

    The weighted average of several |GVar|\s is what one obtains from
    a  least-squares fit of the collection of |GVar|\s to the
    one-parameter fit function ::

        def f(p):
            return N * [p[0]]

    where ``N`` is the number of |GVar|\s. The average is the best-fit
    value for ``p[0]``.  |GVar|\s with smaller standard deviations carry
    more weight than those with larger standard deviations. The averages
    computed by ``wavg`` take account of correlations between the |GVar|\s.

    Typical usage is ::

        x1 = gvar.gvar(...)
        x2 = gvar.gvar(...)
        x3 = gvar.gvar(...)
        xavg = wavg([x1, x2, x3])   # weighted average of x1, x2 and x3

    where the result ``xavg`` is a |GVar| containing the weighted average.

    The individual |GVar|\s in the last example can be  replaced by
    multidimensional distributions, represented by arrays of |GVar|\s
    or dictionaries of |GVar|\s (or arrays of |GVar|\s). For example, ::

        x1 = [gvar.gvar(...), gvar.gvar(...)]
        x2 = [gvar.gvar(...), gvar.gvar(...)]
        x3 = [gvar.gvar(...), gvar.gvar(...)]
        xavg = wavg([x1, x2, x3])
            # xavg[i] is wgtd avg of x1[i], x2[i], x3[i]

    where each array ``x1``, ``x2`` ... must have the same shape.
    The result ``xavg`` in this case is an array of |GVar|\s, where
    the shape of the array is the same as that of ``x1``, etc.

    Another example is ::

        x1 = dict(a=[gvar.gvar(...), gvar.gvar(...)], b=gvar.gvar(...))
        x2 = dict(a=[gvar.gvar(...), gvar.gvar(...)], b=gvar.gvar(...))
        x3 = dict(a=[gvar.gvar(...), gvar.gvar(...)])
        xavg = wavg([x1, x2, x3])
            # xavg['a'][i] is wgtd avg of x1['a'][i], x2['a'][i], x3['a'][i]
            # xavg['b'] is gtd avg of x1['b'], x2['b']

    where different dictionaries can have (some) different keys. Here the
    result ``xavg`` is a :class:`gvar.BufferDict`` having the same keys as
    ``x1``, etc.

    Weighted averages can become costly when the number of random samples being
    averaged is large (100s or more). In such cases it might be useful to set
    parameter ``fast=True``. This causes ``wavg`` to estimate the weighted
    average by incorporating the random samples one at a time into a
    running average::

        result = dataseq[0]
        for dataseq_i in dataseq[1:]:
            result = wavg([result, dataseq_i], ...)

    This method is much faster when ``len(dataseq)`` is large, and gives the
    exact result when there are no correlations between different elements
    of list ``dataseq``. The results are approximately correct when
    ``dataseq[i]`` and ``dataseq[j]`` are correlated for ``i!=j``.

    Args:
        dataseq (list): The |GVar|\s to be averaged. ``dataseq`` is
            a one-dimensional sequence of |GVar|\s, or of arrays of |GVar|\s,
            or of dictionaries containing |GVar|\s  and/or arrays of |GVar|\s.
            All ``dataseq[i]`` must have the same shape.
        fast (bool): Setting ``fast=True`` causes ``wavg`` to compute an
            approximation to the weighted average that is much faster to
            calculate when averaging a large number of samples (100s or more).
            The default is ``fast=False``.
        fitterargs (dict): Additional arguments (e.g., ``svdcut``) for the
            :class:`lsqfit.nonlinear_fit` fitter used to do the averaging.

    Results returned by :func:`gvar.wavg` have the following extra
    attributes describing the average:

        **chi2** - ``chi**2`` for weighted average.

        **dof** - Effective number of degrees of freedom.

        **Q** - The probability that the ``chi**2`` could have been larger,
            by chance, assuming that the data are all Gaussian and consistent
            with each other. Values smaller than 0.1 or so suggest that the
            data are not Gaussian or are inconsistent with each other. Also
            called the *p-value*.

            Quality factor `Q` (or *p-value*) for fit.

        **time** - Time required to do average.

        **svdcorrection** - The *svd* corrections made to the data
            when ``svdcut`` is not ``None``.

        **fit** - Fit output from average.
    """
    if prior is not None:
        warnings.warn(
            'use of prior in lsqfit.wavg is deprecated',
            DeprecationWarning
            )
    if len(dataseq) <= 0:
        if prior is None:
            return None
        if hasattr(prior, 'keys'):
            return BufferDictWAvg(prior, None)
        if numpy.shape(prior) == ():
            return GVarWAvg(prior, None)
        else:
            return ArrayWAvg(numpy.asarray(prior), None)
    elif len(dataseq) == 1 and prior is None:
        if hasattr(dataseq[0], 'keys'):
            return BufferDictWAvg(dataseq[0], None)
        if numpy.shape(dataseq[0]) == ():
            return GVarWAvg(dataseq[0], None)
        else:
            return ArrayWAvg(numpy.asarray(dataseq[0]), None)
    if fast:
        chi2 = 0
        dof = 0
        time = 0
        ans = prior
        svdcorrection = 0.0
        for i, dataseq_i in enumerate(dataseq):
            if ans is None:
                ans = dataseq_i
            else:
                ans = wavg([ans, dataseq_i], fast=False, **fitterargs)
                chi2 += ans.chi2
                dof += ans.dof
                time += ans.time
                svdcorrection += ans.svdcorrection
        fit = ans.fit
        fit.dof = dof
        fit.Q = _gammaQ(dof / 2., chi2 / 2.)
        fit.chi2 = chi2
        fit.time = time
        fit.svdcorrection = svdcorrection
        if hasattr(ans, 'keys'):
            return BufferDictWAvg(ans, fit)
        if numpy.shape(ans) == ():
            return GVarWAvg(ans, fit)
        else:
            return ArrayWAvg(numpy.asarray(ans), fit)
    if hasattr(dataseq[0], 'keys'):
        data = {}
        keys = []
        if prior is not None:
            dataseq = [prior] + list(dataseq)
        for dataseq_i in dataseq:
            for k in dataseq_i:
                if k in data:
                    data[k].append(dataseq_i[k])
                else:
                    data[k] = [dataseq_i[k]]
                    keys.append(k)
        data = gvar.BufferDict(data, keys=keys)
        p0 = gvar.BufferDict()
        for k in data:
            p0[k] = gvar.mean(data[k][0])
            if numpy.any(p0[k] == 0):
                if numpy.shape(p0[k]) == ():
                    p0[k] = data[k][0].sdev / 10.
                else:
                    p0[k][p0[k] == 0] = gvar.sdev(data[k][0])[p0[k] == 0] / 10.
        def fcn(p):
            ans = gvar.BufferDict()
            for k in data:
                ans[k] = len(data[k]) * [p[k]]
            return ans
    else:
        p = numpy.asarray(dataseq[0])
        data = [] if prior is None else [prior]
        data += [dataseqi for dataseqi in dataseq]
        p0 = numpy.asarray(gvar.mean(data[0]))
        if numpy.any(p0 == 0):
            if numpy.shape(p0) == ():
                p0 = numpy.asarray(data[0].sdev / 10.)
            else:
                p0[p0 == 0] += gvar.sdev(data[0])[p0 == 0] / 10.
        data = numpy.array(data)
        def fcn(p):
            return len(data) * [p]
    fit = lsqfit.nonlinear_fit(data=data, fcn=fcn, p0=p0, **fitterargs)
    if p0.shape is None:
        return BufferDictWAvg(fit.p, fit)
    elif p0.shape == ():
        return GVarWAvg(fit.p.flat[0], fit)
    else:
        return ArrayWAvg(fit.p.reshape(p0.shape), fit)


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
        """ Extract fit dataset from :class:`gvar.dataset.Dataset` ``dataset``.

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
        """ Extract fit data corresponding to this model from data set ``data``.

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


class chained_lsqfit(lsqfit.nonlinear_fit):
    " Fit results from chained fit. "
    def __init__(self, p, chained_fits, fitter, prior):
        if len(chained_fits) <= 0:
            raise ValueError('no chained fits')
        self._p = p
        self.palt = p
        self.pmean = gvar.mean(p)
        self.psdev = gvar.sdev(p)
        self.chained_fits = chained_fits

        # extract fcn, fcn values, and data from fits and fitter
        # (for format(...))
        self.fcn = fitter.buildfitfcn()
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
        self.svdcorrection = 0
        self.svdn = 0
        self.dof = 0
        self.chi2 = 0
        self.nit = 0
        self.time = 0
        self.svdcut = 0
        self.tol = [0., 0., 0.]
        self.error = []
        self.logGBF = 0.
        self.fitter = 'chained fit'
        self.description = ''
        for k in self.chained_fits:
            self.svdcorrection += self.chained_fits[k].svdcorrection
            if k[:5] == 'wavg(' and k[-1] == ')':
                continue
            self.svdn += self.chained_fits[k].svdn
            self.dof += self.chained_fits[k].dof
            self.chi2 += self.chained_fits[k].chi2
            self.nit += self.chained_fits[k].nit
            self.time += self.chained_fits[k].time
            svdcut = self.chained_fits[k].svdcut
            if svdcut is not None and  abs(svdcut) > abs(self.svdcut):
                self.svdcut = svdcut
            tol = self.chained_fits[k].tol
            for i in range(3):
                self.tol[i] = max(self.tol[i], tol[i])
            error = self.chained_fits[k].error
            if error is not None:
                self.error.append(error)
            logGBF = self.chained_fits[k].logGBF
            if logGBF is not None:
                self.logGBF += logGBF
        if len(self.error) == 0:
            self.error = None
        self.tol = tuple(self.tol)
        self.Q = lsqfit.gammaQ(self.dof/2., self.chi2/2.)
        if self.logGBF == 0:
            self.logGBF = None

        # others
        self.stopping_criterion = None
        self.cov = None
        self.fitter_results = None
        self.p0 = None
        self.nblocks = None

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

    def formatall(self, *args, **kargs):
        " Add-on method for fits returned by chained_lsqfit. "
        ans = ''
        for x in self.chained_fits:
            ans += 10 * '=' + ' ' + str(x) + '\n'
            ans += self.chained_fits[x].format(*args, **kargs)
            ans += '\n'
        return ans[:-1]

class MultiFitter(object):
    """ Nonlinear least-squares fitter for a collection of models.

    Fits collections of data that are modeled by collections of models.
    Fits can be simultaneous (:meth:`lsqfit.MultiFitter.lsqfit`) or chained
    (:meth:`lsqfit.MultiFitterl.chained_lsqfit`).

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
            variables in the fit and those that are not.
        fitname (callable or ``None``): Individual fits in a chained fit are
            assigned default names, constructed from the datatags of
            the corresponding models, for access and reporting. These names
            get unwieldy when lots of models are involved. When ``fitname``
            is not ``None`` (default), each default name ``dname`` is
            replaced by ``fitname(dname)`` which should return a string.
        wavg_kargs (float): Keyword arguments for :meth:`lsqfit.wavg` when
            used to combine results from parallel sub-fits in a chained fit.
        fitterargs (dict): Additional arguments for the
            :class:`lsqfit.nonlinear_fit` object used to do the fits.
            These can be collected in a dictionary (e.g.,
            ``fitterargs=dict(tol=1e-6, maxit=500))``) or listed as
            separate arguments (e.g., ``tol=1e-6, maxit=500``).
    """

    def __init__(
        self, models, mopt=None, ratio=False, fast=True,
        wavg_kargs=dict(svdcut=1e-12), fitname=None, fitterargs={},
        **more_fitterargs
        ):
        super(MultiFitter, self).__init__()
        models = [models] if isinstance(models, MultiFitterModel) else models
        self.models = models
        self.fit = None         # last fit
        self.ratio = ratio
        self.mopt = mopt
        self.fast = fast
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
            'mopt', 'fast', 'ratio', 'wavg_kargs',
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
        def _fitfcn(p, flatmodels=self.flatmodels):
            ans = gvar.BufferDict()
            for m in flatmodels:
                ans[m.datatag] = (
                    m.fitfcn(p) if m.ncg <= 1 else
                    MultiFitter.coarse_grain(m.fitfcn(p), m.ncg)
                    )
            return ans
        return _fitfcn

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

    def lsqfit(self, data=None, pdata=None, prior=None, p0=None, **kargs):
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
            kargs: Arguments that (temporarily) override parameters specified
                when the :class:`MultiFitter` was created. Can also include
                additional arguments to be passed through to the :mod:`lsqfit`
                fitter.
        """
        # gather parameters
        if prior is None:
            raise ValueError('no prior')
        kargs, oldargs = self.set(**kargs)

        # save parameters for bootstrap (in case needed)
        fitter_args_kargs = (
            self.chained_lsqfit,
            dict(data=data, prior=prior, pdata=pdata, models=self.models),
            dict(kargs),
            )

        # build prior, data and function
        fitprior = self.buildprior(prior=prior, mopt=self.mopt)
        fitdata = self.builddata(
            mopt=self.mopt, data=data, pdata=pdata, prior=prior
            )
        fitfcn = self.buildfitfcn()

        # fit
        self.fit = lsqfit.nonlinear_fit(
            data=fitdata, prior=fitprior, fcn=fitfcn, p0=p0,
            **self.fitterargs
            )
        if len(self.flatmodels) > 1:
            fname = self.fitname(
                '(' +
                ','.join([self.fitname(k.datatag) for k in self.flatmodels])
                + ')'
                )
        else:
            fname = self.fitname(self.flatmodels[0].datatag)
        self.fit.chained_fits = collections.OrderedDict([(fname, self.fit)])

        # add methods for printing and plotting
        def _formatall(*args, **kargs):
            " Add-on method for fits returned by chained_lsqfit. "
            ans = ''
            for x in self.fit.chained_fits:
                ans += 10 * '=' + ' ' + str(x) + '\n'
                ans += self.fit.chained_fits[x].format(*args, **kargs)
                ans += '\n'
            return ans[:-1]
        self.fit.formatall = _formatall
        def _show_plots(save=False, view='ratio'):
            MultiFitter.show_plots(
                fitdata=fitdata, fitval=fitfcn(self.fit.p),
                save=save, view=view,
                )
        self.fit.show_plots = _show_plots

        # restore default keywords
        self.set(**oldargs)

        # add bootstrap method
        fitter_args_kargs[1]['p0'] = self.fit.pmean
        def _bstrap_iter(
            n=None, datalist=None, pdatalist=None, **kargs
            ):
            return MultiFitter._bootstrapped_fit_iter(
                fitter_args_kargs,
                n=n, datalist=datalist, pdatalist=pdatalist, **kargs
                )
        self.fit.bootstrapped_fit_iter = _bstrap_iter
        return self.fit

    def chained_lsqfit(
        self, data=None, pdata=None, prior=None, p0=None, **kargs
        ):
        """ Compute chained least-squares fit of models to data.

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
                a model or a tuple of models (see #2). The ``pn`` are fit
                separately: the fit output from one fit is *not* fed into the
                prior of the next (i.e., the fits are effectively in
                parallel). Results from the separate fits are averaged at the
                end to provide a single composite result for the collection of
                fits. Parallel fits are effective (and fast) when the
                different fits have few or no fit parameters in common.

            4) A dictionary that (temporarily) resets default values for
                fitter keywords. The new values, specified in the dictionary,
                apply to subsequent fits in the chain. Any number of such
                dictionaries can be included in the model chain.


        Fit results are returned in a :class:`lsqfit.MultiFitter.chained_fit`
        object ``fit``, which is very similar to a :class:`nonlinear_fit`
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
                next call to ``self.lsqfit()``).
            kargs: Arguments that override parameters specified when
                the :class:`MultiFitter` was created. Can also include
                additional arguments to be passed through to
                the :mod:`lsqfit` fitter.
        """
        if prior is None:
            raise ValueError('no prior')
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

        # execute tasks in self.tasklist
        chained_fits = collections.OrderedDict()
        all_fnames = []
        for tasktype, taskdata in self.tasklist:
            if tasktype == 'fit':
                fitter = self.__class__(models=taskdata, **kargs)
                fit = fitter.lsqfit(
                    data=data, pdata=pdata, prior=prior, p0=p0
                    )
                fname = list(fit.chained_fits.keys())[0]
                if fname in chained_fits:
                    raise ValueError('duplicate fits in chain: ' + str(fname))
                elif fname[:5] == 'wavg(' and fname[-1] == ')':
                    raise ValueError('bad fit name: ' + fname)
                else:
                    all_fnames.append(fname)
                    chained_fits[fname] = fit
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
                    nlist = all_fnames[-taskdata:]
                    plist = [chained_fits[k].p for k in nlist]
                    fit = lsqfit.wavg(plist, **self.wavg_kargs).fit
                    fname = self.fitname('wavg({})'.format(','.join(nlist)))
                    all_fnames.append(fname)
                    chained_fits[fname] = fit
            elif tasktype == 'update-kargs':
                kargs.update(taskdata)
            else:
                raise RuntimeError('unknown task: ' + tasktype)

        # build output class
        self.fit = chained_lsqfit(
            p=prior, chained_fits=chained_fits,
            fitter=self, prior=fitter_args_kargs[1]['prior'],
            )

        # add bootstrap method
        fitter_args_kargs[1]['p0'] = self.fit.pmean
        def _bstrap_iter(
            n=None, datalist=None, pdatalist=None, **kargs
            ):
            return MultiFitter._bootstrapped_fit_iter(
                fitter_args_kargs,
                n=n, datalist=datalist, pdatalist=pdatalist, **kargs
                )
        self.fit.bootstrapped_fit_iter = _bstrap_iter

        # restore default keywords
        self.set(**oldargs)
        return self.fit

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
                for sm in m:
                    if isinstance(sm, MultiFitterModel):
                        tasklist += [('fit', [sm])]
                    elif isinstance(sm, tuple):
                        tasklist += [('fit', list(sm))]
                    else:
                        raise ValueError(
                            'type {} not allowed in sublists '.format(
                                str(type(sm))
                                )
                            )
                tasklist += [('wavg', len(m))]
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
                    plt.yscale('log', nonposy='clip')
                if plotview == 'loglog':
                    plt.xscale('log', nonposx='clip')
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
                    plt.errorbar(x[ii], g[ii]/gth[ii], dg[ii]/gth[ii], fmt='o')
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








