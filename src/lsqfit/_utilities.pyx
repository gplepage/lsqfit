# cython: language_level=3str
# Copyright (c) 2011-24 G. Peter Lepage.
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

cimport gvar

import gvar
import numpy


cdef gvar.GVar[::1] dot(double[:,:] w, x):
    r""" Compute dot product of matrix ``w`` with vector ``x``.

    This is a substitute for ``numpy.dot`` that is optimized for the
    case where ``w`` is a 2-dimensional array of ``float``\s, and ``x`` is
    a 1=dimensional array of ``gvar.GVar``\s. Other cases are handed off
    to ``numpy.dot``.
    """
    cdef gvar.GVar g
    cdef gvar.GVar gans, gx
    cdef Py_ssize_t i, nans
    cdef gvar.GVar[::1] ans
    nans = w.shape[0]
    ans = numpy.zeros(nans, object)
    for i in range(nans):
        ans[i] = gvar.wsum_gvar(w[i], x)
    return ans


def _build_chiv_chivw(yp_pdf, fcn, prior):
    """ Build ``chiv`` where ``chi**2=sum(chiv(p)**2)``.

    Also builds ``chivw``.
    """
    cv = chiv(fd=yp_pdf, fcn=fcn, noprior=prior is None)
    # cv = functools.partial(chiv, fd=yp_pdf, fcn=fcn, noprior=prior is None)
    cvw = chivw(fd=yp_pdf, fcn=fcn, noprior=prior is None)
    # cvw = functools.partial(chivw, fd=yp_pdf, fcn=fcn, noprior=prior is None)
    return cv, cvw

cdef class chiv(object):
    """ chi**2 = sum(chiv(p)**2) """
    cdef object mean
    cdef Py_ssize_t nw
    cdef object inv_wgts
    cdef object fcn 
    cdef bint noprior

    def __init__(self, fd, fcn, noprior):
        self.mean = fd.mean 
        self.nw = fd.nchiv 
        self.inv_wgts = fd.i_invwgts 
        self.fcn = fcn 
        self.noprior = noprior 

    def __call__(self, p, mixed=False): 
        # mixed=True indicates that delta might contain 
        # a mix of floats and GVars. This is used only 
        # by the varpro algorithm.     
        cdef Py_ssize_t i1, i2
        cdef Py_ssize_t[:] iw
        cdef double[:] wgts
        cdef double[:,:] wgt
        cdef bint all_gvar = True
        if self.noprior:
            delta = self.fcn(p) - self.mean
        else:
            delta = numpy.concatenate((self.fcn(p), p)) - self.mean
        if isinstance(delta[0], gvar.GVar) or mixed:
            ans = numpy.zeros(self.nw, object)
            all_gvar = False if mixed else True
        else:
            ans = numpy.zeros(self.nw, float)
            # delta = numpy.asarray(delta, dtype=float)
            all_gvar = False
        iw, wgts = self.inv_wgts[0]
        i1 = 0
        i2 = len(iw)
        if i2 > 0:
            ans[i1:i2] = numpy.multiply(wgts, delta[iw])
        for iw, wgt in self.inv_wgts[1:]:
            i1 = i2
            i2 += len(wgt)
            ans[i1:i2] = dot(wgt, delta[iw]) if all_gvar else numpy.dot(wgt, delta[iw])
        return ans

cdef class chivw(object):
    cdef object mean
    cdef Py_ssize_t niw
    cdef object inv_wgts
    cdef object fcn 
    cdef bint noprior

    def __init__(self, fd, fcn, noprior):
        self.mean = fd.mean 
        self.niw = fd.mean.size 
        self.inv_wgts = fd.i_invwgts 
        self.fcn = fcn 
        self.noprior = noprior 

    def __call__(self, p, mixed=False):        
        # mixed=True indicates that delta might contain 
        # a mix of floats and GVars. This is used only 
        # by the varpro algorithm      
        cdef Py_ssize_t[:] iw
        cdef double[:] wgts
        cdef double[:] wj
        cdef double[:,:] wgt
        cdef double[:,:] wght2
        cdef bint all_gvar
        if self.noprior:
            delta = self.fcn(p) - self.mean
        else:
            delta = numpy.concatenate((self.fcn(p), p)) - self.mean
        if isinstance(delta[0], gvar.GVar) or mixed:
            ans = numpy.zeros(self.niw, object)
            all_gvar = False if mixed else True
        else:
            ans = numpy.zeros(self.niw, float)
            # delta = numpy.asarray(delta, dtype=float)
            all_gvar = False
        iw, wgts = self.inv_wgts[0]
        if len(iw) > 0:
            ans[iw] = numpy.multiply(numpy.power(wgts, 2), delta[iw])
        for iw, wgt in self.inv_wgts[1:]:
            wgt2 = numpy.zeros((wgt.shape[1], wgt.shape[1]), float)
            for wj in wgt:
                wgt2 += numpy.outer(wj, wj)
            ans[iw] = dot(wgt2, delta[iw]) if all_gvar else numpy.dot(wgt2, delta[iw]) 
        return ans
