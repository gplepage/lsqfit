""" Correlated gaussian random variables.
    
Objects of type :class:`gvar.GVar` represent gaussian random variables, which 
are specified by a mean and standard deviation. They are created using
:func:`gvar.gvar`: for example, ::
    
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
    
    - ``bin_data(data)`` --- bin random sample data
    
    - ``avg_data(data)`` --- estimate means of random sample data
    
    - class ``Dataset`` --- class for collecting random sample data
    
    - class ``BufferDict`` --- ordered dictionary with data buffer
    
    - ``raniter(g,N)`` --- iterator for random numbers
    
    - ``bootstrap_iter(g,N)`` --- bootstrap iterator
    
    - ``svd(mat)`` --- SVD analysis (esp. for covariance matrices)
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

from ._gvar import *
from ._svec_smat import *
from ._bufferdict import BufferDict
from ._utilities import *

from . import dataset

def svd(g, svdcut=None, svdnum=None, rescale=True, compute_inv=False):
    """ Apply svd cuts to collection of |GVar|\s in ``g``. 
        
    ``g`` is an array of |GVar|\s or a dictionary containing |GVar|\s 
    and/or arrays of |GVar|\s. ``svd(g,...)`` creates a copy of ``g`` 
    whose |GVar|\s have been modified so that their covariance matrix is
    less singular than for the original ``g`` (the |GVar| means are unchanged).
    This is done using an *svd* algorithm which is controlled by three 
    parameters: ... finish later
                
    When argument ``compute_inv=True``, ``svd`` 
    """
    if hasattr(g,'keys'):
        g = BufferDict(g)
    else:
        g = numpy.array(g)
    cov = evalcov(g.flat)
    if numpy.all(cov == numpy.diag(numpy.diag(cov))):
        ## covariance matrix diagonal => don't change ##
        if compute_inv:
            cov = numpy.diag(cov)
            svd.wgt = cov**(-0.5)
            svd.logdet = numpy.sum(numpy.log(covi) for covi in cov)
            svd.svdcorrection = None
            svd.val = cov
        ##
        return g
    s = SVD(cov, svdcut=svdcut, svdnum=svdnum, rescale=rescale,
            compute_delta=compute_inv)
    if compute_inv:
        svd.logdet = (0 if s.D is None else 
                      -2*numpy.sum(numpy.log(di) for di in s.D))
        svd.wgt = s.decomp(-1)[::-1]    # reverse order for roundoff
        svd.svdcorrection = s.delta
        svd.logdet += numpy.sum(numpy.log(vali) for vali in s.val)
        svd.val = s.val
        svd.vec = s.vec
    if s.delta is not None:
        g.flat += s.delta
    return g
##
        
