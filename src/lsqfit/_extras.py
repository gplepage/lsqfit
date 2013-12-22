""" part of lsqfit module: extra functions  """

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012-13 G. Peter Lepage. 
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
from ._utilities import multiminex
from gvar import gammaQ


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
    ##
    try:
        z = multiminex(numpy.array(z0), minfcn, **minargs).x
    except ValueError:
        print('*** empbayes_fit warning: null logGBF')
        z = save['lastz']
    args = fitargs(z)
    if save['lastp0'] is not None:
        args['p0'] = save['lastp0']
    return lsqfit.nonlinear_fit(**args), z
##

class WAvg(gvar.GVar):
    """ Result from weighted average :func:`lsqfit.wavg`. 

    :class:`WAvg` objects are |GVar|\s but 
    """
    def __init__(self, avg, chi2, dof, Q):
       super(WAvg, self).__init__(*avg.internaldata)
       self.chi2 = chi2
       self.dof = dof
       self.Q = Q

def wavg(xa, svdcut=None, svdnum=None, rescale=True, covfac=None):
    """ Weighted average of |GVar|\s or arrays/dicts of |GVar|\s.
        
    The weighted average of several |GVar|\s is what one obtains from
    a  least-squares fit of the collection of |GVar|\s to the one-
    parameter fit function ``def f(p): return N * [p[0]]`` where ``N``
    is the number of |GVar|\s. The average is the best-fit value for
    ``p[0]``.  |GVar|\s with smaller standard deviations carry more
    weight than those with larger standard deviations. The averages
    computed by ``wavg`` take account of correlations between the
    |GVar|\s.
        
    Typical usage is::
        
        x1 = gvar.gvar(...)
        x2 = gvar.gvar(...)
        x3 = gvar.gvar(...)
        xavg = wavg([x1, x2, x3])   # weighted average of x1, x2 and x3
    
    The individual |GVar|\s in this example can be  replaced by
    multidimensional distributions, represented by arrays of |GVar|\s
    or dictionaries of |GVar|\s or arrays of |GVar|\s. For example, ::

        x1 = [gvar.gvar(...), gvar.gvar(...)]
        x2 = [gvar.gvar(...), gvar.gvar(...)]
        x3 = [gvar.gvar(...), gvar.gvar(...)]
        xavg = wavg([x1, x2, x3])   
            # xavg[i] is wgtd avg of x1[i], x2[i], x3[i]

    and ::

        x1 = dict(a=[gvar.gvar(...), gvar.gvar(...)], b=gvar.gvar(...))
        x2 = dict(a=[gvar.gvar(...), gvar.gvar(...)], b=gvar.gvar(...))
        x3 = dict(a=[gvar.gvar(...), gvar.gvar(...)], b=gvar.gvar(...))
        xavg = wavg([x1, x2, x3])   
            # xavg['a'][i] is wgtd avg of x1['a'][i], x2['a'][i], x3['a'][i]
            # xavg['b'] is gtd avg of x1['b'], x2['b'], x3['b']       
        
    :param xa: The |GVar|\s to be averaged. ``xa`` is a one-dimensional
        sequence of |GVar|\s, or of arrays of |GVar|\s, or of dictionaries 
        containing |GVar|\s or arrays of |GVar|\s. All ``xa[i]`` must
        have the same shape.
    :param svdcut: If positive, eigenvalues of the ``xa`` covariance matrix
        that are smaller than ``svdcut`` times the maximum eigenvalue 
        are replaced by ``svdcut`` times the maximum eigenvalue. If 
        negative, eigenmodes with eigenvalues smaller than ``|svdcut|``
        times the largest eigenvalue are discarded. If zero or ``None``,
        the covariance matrix is left unchanged.
    :type svdcut: ``None`` or ``float``
    :param svdnum: If positive, at most ``svdnum`` eigenmodes of the 
        ``xa`` covariance matrix are retained; the modes with the smallest
        eigenvalues are discarded. ``svdnum`` is ignored if set to
        ``None``.
    :type svdnum: ``None`` or ``int``
    :param rescale: If ``True``, rescale covariance matrix so diagonal 
        elements all equal 1 before applying *svd* cuts. (Default is
        ``True``.)
    :type rescale: ``bool``
    :param covfac: The covariance matrix (or matrices) of ``xa`` is 
        multiplied by ``covfac`` if ``covfac`` is not ``None``.
    :type covfac: ``None`` or number
    :returns: Weighted average of the ``xa`` elements. The result has the 
        same type and shape as each element of ``xa`` (that is, either a
        |GVar| or an array of |GVar|\s.)
        
    The following function attributes are also set:    
        
    .. attribute:: wavg.chi2
        
        ``chi**2`` for weighted average.
        
    .. attribute:: wavg.dof
        
        Effective number of degrees of freedom.
        
    .. attribute:: wavg.Q
        
        The probability that the ``chi**2`` could have been larger, 
        by chance, assuming that the data are all Gaussain and consistent
        with each other. Values smaller than 0.1 or suggest that the 
        data are not Gaussian or are inconsistent with each other. Also 
        called the *p-value*.

        Quality factor `Q` (or *p-value*) for fit.

    .. attribute:: wavg.time

        Time required to do average.
        
    """
    cpu_time = time.clock()
    xa = numpy.asarray(xa)
    if xa.size == 0:
        return None
    s = None
    svdcorrection = []
    if len(xa.shape) > 1:
        ## xa is an array of arrays ##
        shape = xa[0].shape
        xaflat = [xai.flat for xai in xa]
        chi2 = 0.0
        dof = 0
        ans = []
        for xtuple in zip(*xaflat):
            ans.append(wavg(
                xtuple, 
                svdcut=svdcut, svdnum=svdnum,
                rescale=rescale, covfac=covfac
                ))
            chi2 += wavg.chi2
            dof += wavg.dof
            svdcorrection.append(wavg.svdcorrection)
        ans = numpy.array(ans)
        ans.shape = shape
    elif hasattr(xa[0], 'keys'):
        data = collections.OrderedDict()
        # collect copies of data -- different dicts can have different keys
        for xai in xa:
            for k in xai:
                if k in data:
                    data[k].append(xai[k])
                else:
                    data[k] = [xai[k]]
        ans = gvar.BufferDict()
        chi2 = 0.0
        dof = 0
        for k in data:
            ans[k] = wavg(
                data[k], 
                svdcut=svdcut, svdnum=svdnum,
                rescale=rescale, covfac=covfac
                )
            chi2 += wavg.chi2
            dof += wavg.dof
            svdcorrection.append(wavg.svdcorrection)
    else:
        cov = gvar.evalcov(xa)
        if covfac is not None:
            cov *= covfac
        ## invert cov ## 
        if numpy.all(numpy.diag(numpy.diag(cov))==cov):
            ## cov is diagonal ## 
            invcov = 1./numpy.diag(cov)
            dof = len(xa)-1
            ans = numpy.dot(invcov, xa)/sum(invcov)
            chi2 = numpy.sum((xa-ans)**2*invcov).mean
            ##
        else:
            ## cov is not diagonal ##
            if (svdcut is None or svdcut==0) and (svdnum is None or svdnum<0):
                invcov = numpy.linalg.inv(cov)
                dof = len(xa)-1
            else:
                ## apply svdcuts; compute conditioned inverse ## 
                s = gvar.SVD(cov, svdcut=svdcut, svdnum=svdnum, rescale=rescale,
                             compute_delta=True)
                invcov = numpy.sum(numpy.outer(wj, wj) for wj 
                                        in reversed(s.decomp(-1)))
                dof = len(s.val)-1
                if s.delta is not None:
                    svdcorrection = sum(s.delta)
                ##
            ##
            sum_invcov = numpy.sum(invcov, axis=1)
            ans = numpy.dot(sum_invcov, xa)/sum(sum_invcov)
            chi2 = numpy.dot((xa-ans), numpy.dot(invcov, (xa-ans))).mean
        ans = WAvg(ans, chi2, dof, gammaQ(dof/2., chi2/2.))
        ##
    wavg.chi2 = chi2 
    wavg.dof = dof
    wavg.Q = gammaQ(dof/2., chi2/2.)
    wavg.s = s
    wavg.svdcorrection = svdcorrection
    wavg.time = time.clock() - cpu_time
    return ans
##

if __name__ == '__main__':
    pass
