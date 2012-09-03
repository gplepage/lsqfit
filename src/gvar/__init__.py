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
    
    - ``evalcov(g)`` --- compute covariance matrix
    
    - ``fmt_values(g)`` --- create table of values for printing
    
    - ``fmt_errorbudget(g)`` --- create error-budget table for printing
    
    - class ``BufferDict`` --- ordered dictionary with data buffer
    
    - ``raniter(g,N)`` --- iterator for random numbers
    
    - ``bootstrap_iter(g,N)`` --- bootstrap iterator
    
    - ``svd(g)`` --- SVD modification of covariance matrix
    
    - ``dataset.bin_data(data)`` --- bin random sample data
    
    - ``dataset.avg_data(data)`` --- estimate means of random sample data
    
    - ``dataset.bootstrap_iter(data,N)`` --- bootstrap random sample data
    
    - class ``dataset.Dataset`` --- class for collecting random sample data
    
"""

# Created by G. Peter Lepage (Cornell University) on 2012-05-31.
# Copyright (c) 2012 G. Peter Lepage. 
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

from ._gvarcore import *
gvar = GVarFactory()            # order matters for this statement

from ._svec_smat import *
from ._bufferdict import BufferDict
from ._utilities import *

from . import dataset

_GDEV_LIST = []
    
def switch_gvar(cov=None):
    """ Switch :func:`gvar.gvar` to new :class:`gvar.GVarFactory`.
        
    :returns: New :func:`gvar.gvar`.
    """
    global gvar
    _GDEV_LIST.append(gvar)
    gvar = GVarFactory(cov)
    return gvar
##
    
def restore_gvar():
    """ Restore previous :func:`gvar.gvar`.
        
    :returns: Previous :func:`gvar.gvar`.
    """
    global gvar
    try:
        gvar = _GDEV_LIST.pop()
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
