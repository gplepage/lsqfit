""" part of lsqfit module: extra functions  """

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-14 G. Peter Lepage.
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

import numpy
import gvar
import lsqfit
import time
import collections
from ._utilities import multiminex, gammaQ


def empbayes_fit(z0, fitargs, **minargs):
    """ Call ``lsqfit.nonlinear_fit(**fitargs(z))`` varying ``z``,
    starting at ``z0``, to maximize ``logGBF`` (empirical Bayes procedure).

    The fit is redone for each value of ``z`` that is tried, in order
    to determine ``logGBF``.

    :param z0: Starting point for search.
    :type z0: array
    :param fitargs: Function of array ``z`` that determines which fit
        parameters to use. The function returns these as an argument
        dictionary for :func:`lsqfit.nonlinear_fit`.
    :type fitargs: function
    :param minargs: Optional argument dictionary, passed on to
        :class:`lsqfit.multiminex`, which finds the minimum.
    :type minargs: dictionary
    :returns: A tuple containing the best fit (object of type
        :class:`lsqfit.nonlinear_fit`) and the optimal value for parameter ``z``.
    """
    if minargs == {}: # default
        minargs = dict(tol=1e-3, step=math.log(1.1), maxit=30, analyzer=None)
    save = dict(lastz=None, lastp0=None)
    def minfcn(z, save=save):
        args = fitargs(z)
        if save['lastp0'] is not None:
            args['p0'] = save['lastp0']
        fit = lsqfit.nonlinear_fit(**args)
        if numpy.isnan(fit.logGBF):
            raise ValueError
        else:
            save['lastz'] = z
            save['lastp0'] = fit.pmean
        return -fit.logGBF
    try:
        z = multiminex(numpy.array(z0), minfcn, **minargs).x
    except ValueError:
        print('*** empbayes_fit warning: null logGBF')
        z = save['lastz']
    args = fitargs(z)
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
        by chance, assuming that the data are all Gaussain and consistent
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
        by chance, assuming that the data are all Gaussain and consistent
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
        by chance, assuming that the data are all Gaussain and consistent
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
        self.Q = fit.Q
        self.chi2 = fit.chi2
        self.dof = fit.dof
        self.time = fit.time
        self.svdcorrection = fit.svdcorrection
        self.fit = fit

def wavg(dataseq, prior=None, fast=False, **kargs):
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

    If ``prior`` is not ``None``, it is added to the list of data
    used in the average. Thus ``wavg([x2, x3], prior=x1)`` is the
    same as ``wavg([x1, x2, x3])``.

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

        result = prior
        for dataseq_i in dataseq:
            result = wavg([result, dataseq_i], ...)

    This method is much faster when ``len(dataseq)`` is large, and gives the
    exact result when there are no correlations between different elements
    of list ``dataseq``. The results are approximately correct when
    ``dataseq[i]`` and ``dataseq[j]`` are correlated for ``i!=j``.

    :param dataseq: The |GVar|\s to be averaged. ``dataseq`` is a one-dimensional
        sequence of |GVar|\s, or of arrays of |GVar|\s, or of dictionaries
        containing |GVar|\s or arrays of |GVar|\s. All ``dataseq[i]`` must
        have the same shape.
    :param prior: Prior values for the averages, to be included in the weighted
        average. Default value is ``None``, in which case ``prior`` is ignored.
    :type prior: |GVar| or array/dictionary of |GVar|\s
    :param fast: Setting ``fast=True`` causes ``wavg`` to compute an
        approximation to the weighted average that is much faster to calculate
        when averaging a large number of samples (100s or more). The default is
        ``fast=False``.
    :type fast: bool
    :param kargs: Additional arguments (e.g., ``svdcut``) to the fitter
        used to do the averaging.
    :type kargs: dict

    Results returned by :func:`gvar.wavg` have the following extra
    attributes describing the average:

    .. attribute:: chi2

        ``chi**2`` for weighted average.

    .. attribute:: dof

        Effective number of degrees of freedom.

    .. attribute:: Q

        The probability that the ``chi**2`` could have been larger,
        by chance, assuming that the data are all Gaussain and consistent
        with each other. Values smaller than 0.1 or suggest that the
        data are not Gaussian or are inconsistent with each other. Also
        called the *p-value*.

        Quality factor `Q` (or *p-value*) for fit.

    .. attribute:: time

        Time required to do average.

    .. attribute:: svdcorrection

        The *svd* corrections made to the data when ``svdcut`` is not ``None``.

    .. attribute:: fit

        Fit output from average.
    """
    if len(dataseq) <= 0:
        if prior is None:
            return None
        wavg.Q = 1
        wavg.chi2 = 0
        wavg.dof = 0
        wavg.time = 0
        wavg.fit = None
        wavg.svdcorrection = None
        if hasattr(prior, 'keys'):
            return BufferDictWAvg(dataseq[0], wavg)
        if numpy.shape(prior) == ():
            return GVarWAvg(prior, wavg)
        else:
            return ArrayWAvg(numpy.asarray(prior), wavg)
    elif len(dataseq) == 1 and prior is None:
        wavg.Q = 1
        wavg.chi2 = 0
        wavg.dof = 0
        wavg.time = 0
        wavg.fit = None
        wavg.svdcorrection = None
        if hasattr(dataseq[0], 'keys'):
            return BufferDictWAvg(dataseq[0], wavg)
        if numpy.shape(dataseq[0]) == ():
            return GVarWAvg(dataseq[0], wavg)
        else:
            return ArrayWAvg(numpy.asarray(dataseq[0]), wavg)
    if fast:
        chi2 = 0
        dof = 0
        time = 0
        ans = prior
        svdcorrection = gvar.BufferDict()
        for i, dataseq_i in enumerate(dataseq):
            if ans is None:
                ans = dataseq_i
            else:
                ans = wavg([ans, dataseq_i], fast=False, **kargs)
                chi2 += wavg.chi2
                dof += wavg.dof
                time += wavg.time
                if wavg.svdcorrection is not None:
                    for k in wavg.svdcorrection:
                        svdcorrection[str(i) + ':' + k] = wavg.svdcorrection[k]
        wavg.chi2 = chi2
        wavg.dof = dof
        wavg.time = time
        wavg.Q = gammaQ(dof / 2., chi2 / 2.)
        wavg.svdcorrection = svdcorrection
        wavg.fit = None
        ans.dof = wavg.dof
        ans.Q = wavg.Q
        ans.chi2 = wavg.chi2
        ans.time = wavg.time
        ans.svdcorrection = wavg.svdcorrection
        ans.fit = wavg.fit
        return ans
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
            p0[k] = gvar.mean(data[k][0]) + gvar.sdev(data[k][0]) / 10.
        def fcn(p):
            ans = gvar.BufferDict()
            for k in data:
                ans[k] = len(data[k]) * [p[k]]
            return ans
    else:
        p = numpy.asarray(dataseq[0])
        data = [] if prior is None else [prior]
        data += [dataseqi for dataseqi in dataseq]
        p0 = numpy.asarray(gvar.mean(data[0]) + gvar.sdev(data[0]) / 10.)
        data = numpy.array(data)
        def fcn(p):
            return len(data) * [p]
    fit = lsqfit.nonlinear_fit(data=data, fcn=fcn, p0=p0, **kargs)
    # wavg.Q = fit.Q
    # wavg.chi2 = fit.chi2
    # wavg.dof = fit.dof
    # wavg.time = fit.time
    # wavg.svdcorrection = fit.svdcorrection
    # wavg.fit = fit
    if p0.shape is None:
        return BufferDictWAvg(gvar.BufferDict(p0, buf=fit.p.flat), fit)
    elif p0.shape == ():
        return GVarWAvg(fit.p.flat[0], fit)
    else:
        return ArrayWAvg(fit.p.reshape(p0.shape), fit)

# if __name__ == '__main__':
#     pass
