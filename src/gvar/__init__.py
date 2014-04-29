""" Correlated gaussian random variables.
    
Objects of type :class:`gvar.GVar` represent gaussian random variables,
which are specified by a mean and standard deviation. They are created
using :func:`gvar.gvar`: for example, ::
    
    >>> x = gvar.gvar(0,3)          # 0 +- 3
    >>> y = gvar.gvar(2,4)          # 2 +- 4
    >>> z = x+y                     # 2 +- 5
    >>> print(z)
    2 +- 5
    >>> print(z.mean)
    2.0
    >>> print(z.sdev)
    5.0
    
This module contains tools for creating and manipulating gaussian random
variables including:
    
    - ``mean(g)`` --- extract means
    
    - ``sdev(g)`` --- extract standard deviations
    
    - ``var(g)`` --- extract variances

    - ``chi2(g1, g2)`` --- ``chi**2`` of ``g1-g2``.
    
    - ``evalcov(g)`` --- compute covariance matrix

    - ``evalcorr(g)`` --- compute correlation matrix
    
    - ``fmt_values(g)`` --- create table of values for printing
    
    - ``fmt_errorbudget(g)`` --- create error-budget table for printing

    - ``fmt_chi2(f)`` --- format chi**2 information in f as string for printing
    
    - class ``BufferDict`` --- ordered dictionary with data buffer
    
    - ``raniter(g,N)`` --- iterator for random numbers
    
    - ``bootstrap_iter(g,N)`` --- bootstrap iterator
    
    - ``svd(g)`` --- SVD modification of covariance matrix
    
    - ``dataset.bin_data(data)`` --- bin random sample data
    
    - ``dataset.avg_data(data)`` --- estimate means of random sample data
    
    - ``dataset.bootstrap_iter(data,N)`` --- bootstrap random sample data
    
    - class ``dataset.Dataset`` --- class for collecting random sample data

There are also sub-modules that implement some standard numerical analysis 
tools for use with |GVar|\s:

    - ``ode`` --- integration of systems of ordinary differential equations

    - ``cspline`` --- cubic splines for 1-d data
    
"""

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
import sys

from ._gvarcore import *
gvar = GVarFactory()            # order matters for this statement

from ._svec_smat import *
from ._bufferdict import BufferDict, asbufferdict
from ._utilities import *
from ._version import version as __version__

from . import dataset
from . import ode 
from . import cspline

try:
    # use lsqfit's gammaQ if available; otherwise use one in ._utilities
    from lsqfit._utilities import gammaQ
except:
    pass

_GVAR_LIST = []

def ranseed(seed=None):
    """ Seed random number generators with tuple ``seed``.
        
    Argument ``seed`` is a :class:`tuple` of integers that is used to seed
    the random number generators used by :mod:`numpy` and  
    :mod:`random` (and therefore by :mod:`gvar`). Reusing 
    the same ``seed`` results in the same set of random numbers.

    ``ranseed`` generates its own seed when called without an argument
    or with ``seed=None``. This seed is stored in ``ranseed.seed`` and 
    also returned by the function. The seed can be used to regenerate
    the same set of random numbers at a later time.
        
    :param seed: A tuple of integers. Generates a random tuple if ``None``.
    :type seed: tuple or None
    :returns: The seed.
    """
    if seed is None:
        seed = numpy.random.randint(1, sys.maxint, size=3)
    seed = tuple(seed)
    numpy.random.seed(seed)
    ranseed.seed = seed
    return seed
    
def switch_gvar(cov=None):
    """ Switch :func:`gvar.gvar` to new :class:`gvar.GVarFactory`.
        
    :returns: New :func:`gvar.gvar`.
    """
    global gvar
    _GVAR_LIST.append(gvar)
    gvar = GVarFactory(cov)
    return gvar
##
    
def restore_gvar():
    """ Restore previous :func:`gvar.gvar`.
        
    :returns: Previous :func:`gvar.gvar`.
    """
    global gvar
    try:
        gvar = _GVAR_LIST.pop()
    except IndexError:
        raise RuntimeError("no previous gvar")
    return gvar
##
    
def gvar_factory(cov=None):
    """ Return new function for creating |GVar|\s (to replace 
    :func:`gvar.gvar`). 
        
    If ``cov`` is specified, it is used as the covariance matrix
    for new |GVar|\s created by the function returned by 
    ``gvar_factory(cov)``. Otherwise a new covariance matrix is created
    internally.
    """
    return GVarFactory(cov)
##
            
def asgvar(x):
    """ Return x if it is type |GVar|; otherwise return 'gvar.gvar(x)`."""
    if isinstance(x,GVar):
        return x
    else:
        return gvar(x)
##

def chi2(g1, g2=None, svdcut=1e-15, svdnum=None, nocorr=False, fmt=False):
    """ Compute chi**2 of ``g1-g2``. 

    ``chi**2`` is a measure of whether the multi-dimensional 
    Gaussian distributions ``g1`` and ``g2`` (dictionaries or arrays) 
    agree with each other --- that is, do their means agree 
    within errors for corresponding elements. The probability is high 
    if ``chi2(g1,g2)/chi2.dof`` is of order 1 or smaller.

    Usually ``g1`` and ``g2`` are dictionaries with the same keys,
    where ``g1[k]`` and ``g2[k]`` are |GVar|\s or arrays of 
    |GVar|\s having the same shape. Alternatively ``g1`` and ``g2``
    can be |GVar|\s, or arrays of |GVar|\s having the same shape. 
    
    One of ``g1`` or ``g2`` can contain numbers instead of |GVar|\s,
    in which case ``chi**2`` is a measure of the likelihood that 
    the numbers came from the distribution specified by the other 
    argument. 

    One or the other of ``g1`` or ``g2`` can be missing keys, or missing
    elements from arrays. Only the parts of ``g1`` and ``g2`` that 
    overlap are used. Also setting ``g2=None`` is equivalent to replacing its 
    elements by zeros.

    ``chi**2`` is computed from the inverse of the covariance matrix
    of ``g1-g2``. The matrix inversion can be sensitive to roundoff 
    errors. In such cases, *SVD* cuts can be applied by setting
    parameters ``svdcut`` and ``svdnum``. See the documentation 
    for :class:`gvar.SVD` for information about these parameters.

    The return value is the ``chi**2``. Extra data is stored in 
    ``chi2`` itself:

    .. attribute:: chi2.dof

        Number of degrees of freedom (that is, the number of variables
        compared).

    .. attribute:: chi2.Q

        The probability that the ``chi**2`` could have been larger, 
        by chance, even if ``g1`` and ``g2`` agree. 
        Values smaller than 0.1 or so suggest that they do not
        agree. Also called the *p-value*.

    If argument ``fmt==True``, then a string is returned containing the
    ``chi**2`` per degree of freedom, the number of degrees of freedom, and
    ``Q``.
    """
    # leaving nocorr (turn off correlations) undocumented because I
    #   suspect I will remove it
    if g2 is None:
        diff = BufferDict(g1).buf if hasattr(g1, 'keys') else numpy.asarray(g1).flatten()
    elif hasattr(g1, 'keys') and hasattr(g2, 'keys'):
        # g1 and g2 are dictionaries
        g1 = BufferDict(g1)
        g2 = BufferDict(g2)
        diff = BufferDict()
        keys = set(g1.keys())
        keys = keys.intersection(g2.keys())
        for k in keys:
            g1k = g1[k]
            g2k = g2[k]
            shape = tuple(
                [min(s1,s2) for s1, s2 in zip(numpy.shape(g1k), numpy.shape(g2k))]
                )
            diff[k] = numpy.zeros(shape, object)
            if len(shape) == 0:
                diff[k] = g1k - g2k
            else:
                for i in numpy.ndindex(shape):
                    diff[k][i] = g1k[i] - g2k[i]
        diff = diff.buf
    elif not hasattr(g1, 'keys') and not hasattr(g2, 'keys'):
        # g1 and g2 are arrays or scalars
        g1 = numpy.asarray(g1)
        g2 = numpy.asarray(g2)
        shape = tuple(
            [min(s1,s2) for s1, s2 in zip(numpy.shape(g1), numpy.shape(g2))]
            )
        diff = numpy.zeros(shape, object)
        if len(shape) == 0:
            diff = numpy.array(g1 - g2)
        else:
            for i in numpy.ndindex(shape):
                diff[i] = g1[i] - g2[i]
        diff = diff.flatten()
    else:
        # g1 and g2 are something else
        raise ValueError(
            'cannot compute chi**2 for types ' + str(type(g1)) + ' ' +
            str(type(g2))
            )
    chi2.dof = diff.size
    if chi2.dof == 0:
        chi2.Q = 0
        return 0.0    
    if nocorr:
        # ignore correlations
        ans = numpy.sum(mean(diff) ** 2 / var(diff))
        chi2.dof = len(diff)
        chi2.s = None
    else:
        s = SVD(evalcov(diff), svdcut=svdcut, svdnum=svdnum, rescale=True)
        ans = numpy.sum(numpy.dot(s.decomp(-1), mean(diff))**2)
        chi2.s = s
        chi2.dof = len(s.val)
    chi2.Q = gammaQ(chi2.dof/2., ans/2.)
    chi2.chi2 = ans
    return ans if fmt == False else fmt_chi2(chi2)

def fmt_chi2(f):
    """ Return string containing ``chi**2/dof``, ``dof`` and ``Q`` from ``f``.

    Assumes ``f`` has attributes ``chi2``, ``dof`` and ``Q``. The 
    logarithm of the Bayes factor will also be printed if ``f`` has
    attribute ``logGBF``.
    """
    if hasattr(f, 'logGBF'):
        fmt = "chi2/dof = %.2g [%d]    Q = %.2g    log(GBF) = %.5g"
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt % (chi2_dof, f.dof, f.Q, f.logGBF)
    else:
        fmt = "chi2/dof = %.2g [%d]    Q = %.2g"
        chi2_dof = f.chi2 / f.dof if f.dof != 0 else 0
        return fmt % (chi2_dof, f.dof, f.Q)

def svd(g, svdcut=None, svdnum=None, rescale=True, compute_inv=False):
    """ Apply svd cuts to collection of |GVar|\s in ``g``. 
        
    ``g`` is an array of |GVar|\s or a dictionary containing |GVar|\s
    and/or arrays of |GVar|\s. ``svd(g,...)`` returns a copy of ``g`` whose
    |GVar|\s have been modified so that their covariance matrix is less
    singular than for the original ``g`` (the |GVar| means are unchanged).
    This is done using an *svd* algorithm which is controlled by three
    parameters: ``svdcut``, ``svdnum`` and ``rescale`` (see
    :class:`gvar.SVD` for more details). *svd* cuts are not applied when
    the covariance matrix is diagonal (that is, when there are no
    correlations between different elements of ``g``).
        
    The input parameters are :
        
    :param g: An array of |GVar|\s or a dicitionary whose values are 
        |GVar|\s and/or arrays of |GVar|\s.
    :param svdcut: If positive, replace eigenvalues of the covariance 
        matrix with ``svdcut*(max eigenvalue)``; if negative, discard
        eigenmodes with eigenvalues smaller than ``svdcut`` times the
        maximum eigenvalue. Default is ``None``.
    :type svdcut: ``None`` or number ``(|svdcut|<=1)``.
    :param svdnum: If positive, keep only the modes with the largest 
        ``svdnum`` eigenvalues; ignore if set to ``None``. Default is
        ``None``.
    :type svdnum: ``None`` or int
    :param rescale: Rescale the input matrix to make its diagonal elements 
        equal to 1.0 before applying *svd* cuts. (Default is ``True``.)
    :param compute_inv: Compute representation of inverse of covariance 
        matrix if ``True``; the result is stored in ``svd.inv_wgt`` (see
        below). Default value is ``False``.
    :returns: A copy of ``g`` with the same means but with a covariance
        matrix modified by *svd* cuts.
       
    Data from the *svd* analysis of ``g``'s covariance matrix is stored in
    ``svd`` itself:
            
    .. attribute:: svd.val
        
        Eigenvalues of the covariance matrix after *svd* cuts (and after
        rescaling if ``rescale=True``); the eigenvalues are ordered, with
        the smallest first.
        
    .. attribute:: svd.vec
     
        Eigenvectors of the covariance matrix after *svd* cuts (and after
        rescaling if ``rescale=True``), where ``svd.vec[i]`` is the vector
        corresponding to ``svd.val[i]``.
          
    .. attribute:: svd.eigen_range
        
        Ratio of the smallest to largest eigenvalue before *svd* cuts are
        applied (but after rescaling if ``rescale=True``).
        
    .. attribute:: svd.D    
        
        Diagonal of matrix used to rescale the covariance matrix before
        applying *svd* cuts (cuts are applied to ``D*cov*D``) if
        ``rescale=True``; ``svd.D`` is ``None`` if ``rescale=False``.
          
    .. attribute:: svd.logdet
        
        Logarithm of the determinant of the covariance matrix after *svd*
        cuts are applied.
          
    .. attribute:: svd.correction
        
        Vector of the *svd* corrections to ``g.flat``;
        
    .. attribute:: svd.inv_wgt
        
        The sum of the outer product of vectors ``inv_wgt[i]`` with
        themselves equals the inverse of the covariance matrix after *svd*
        cuts. Only computed if ``compute_inv=True``. The order of the
        vectors is reversed relative to ``svd.val`` and ``svd.vec``
    """
    ## replace g by a copy of g ##
    if hasattr(g,'keys'):
        g = BufferDict(g)
    else:
        g = numpy.array(g)
    ##
    cov = evalcov(g.flat)
    if numpy.all(cov == numpy.diag(numpy.diag(cov))):
        ## covariance matrix diagonal => don't change it ##
        cov = numpy.diag(cov)
        if compute_inv:
            svd.inv_wgt = cov**(-0.5)
        svd.logdet = numpy.sum(numpy.log(covi) for covi in cov)
        svd.correction = None
        svd.val = sorted(cov)
        svd.vec = numpy.eye(len(cov))
        svd.eigen_range = min(cov)/max(cov)
        svd.D = None
        ##
        return g
    s = SVD(cov, svdcut=svdcut, svdnum=svdnum, rescale=rescale,
            compute_delta=True)
    svd.logdet = (0 if s.D is None else 
                  -2*numpy.sum(numpy.log(di) for di in s.D))
    svd.logdet += numpy.sum(numpy.log(vali) for vali in s.val)
    svd.correction = s.delta
    svd.val = s.val
    svd.vec = s.vec
    svd.eigen_range = s.kappa
    svd.D = s.D
    if compute_inv:
        svd.inv_wgt = s.decomp(-1)[::-1]    # reverse order for roundoff
    if s.delta is not None:
        g.flat += s.delta
    return g
##
        
## legacy code support ##
fmt_partialsdev = fmt_errorbudget 
##
