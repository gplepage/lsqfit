# Copyright (c) 2011-17 G. Peter Lepage.
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

cimport numpy
cimport cython
cimport gvar

import gvar
import numpy

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t


def dot(numpy.ndarray[numpy.float_t, ndim=2] w not None, x):
    """ Compute dot product of matrix ``w`` with vector ``x``.

    This is a substitute for ``numpy.dot`` that is highly optimized for the
    case where ``w`` is a 2-dimensional array of ``float``\s, and ``x`` is
    a 1=dimensional array of ``gvar.GVar``\s. Other cases are handed off
    to ``numpy.dot``.
    """
    cdef gvar.GVar g
    cdef gvar.GVar gans, gx
    cdef Py_ssize_t i, nx, nans
    cdef numpy.ndarray[object, ndim=1] ans
    if not isinstance(x[0], gvar.GVar):
        return numpy.dot(w, x)
    nx = len(x)
    nans = w.shape[0]
    assert nx==w.shape[1], str(nx)+'!='+str(w.shape[1])
    ans = numpy.zeros(nans, object)
    for i in range(nans):
        ans[i] = gvar.wsum_gvar(w[i], x)
    return ans
#
#
def _build_chiv_chivw(fdata, fcn, prior):
    """ Build ``chiv`` where ``chi**2=sum(chiv(p)**2)``.

    Also builds ``chivw``.
    """
    nw = sum(len(wgts) for iw, wgts in fdata.inv_wgts)
    niw = sum(len(iw) for iw, wgts in fdata.inv_wgts)
    if prior is not None:
        def chiv(p, fd=fdata):
            cdef Py_ssize_t i1, i2
            cdef numpy.ndarray[INTP_TYPE, ndim=1] iw
            cdef numpy.ndarray[numpy.float_t, ndim=1] wgts
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt
            cdef numpy.ndarray ans, delta
            delta = numpy.concatenate((fcn(p), p)) - fd.mean
            if delta.dtype == object:
                ans = numpy.zeros(nw, object)
            else:
                ans = numpy.zeros(nw, numpy.float_)
            iw, wgts = fd.inv_wgts[0]
            i1 = 0
            i2 = len(iw)
            if i2 > 0:
                ans[i1:i2] = wgts * delta[iw]
            for iw, wgt in fd.inv_wgts[1:]:
                i1 = i2
                i2 += len(wgt)
                ans[i1:i2] = dot(wgt, delta[iw])
            return ans
        def chivw(p, fd=fdata):
            cdef numpy.ndarray[INTP_TYPE, ndim=1] iw
            cdef numpy.ndarray[numpy.float_t, ndim=1] wgts, wj
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt2
            cdef numpy.ndarray ans, delta
            delta = numpy.concatenate((fcn(p), p)) - fd.mean
            if delta.dtype == object:
                ans = numpy.zeros(niw, object)
            else:
                ans = numpy.zeros(niw, numpy.float_)
            iw, wgts = fd.inv_wgts[0]
            if len(iw) > 0:
                ans[iw] = wgts ** 2 * delta[iw]
            for iw, wgt in fd.inv_wgts[1:]:
                wgt2 = numpy.zeros((wgt.shape[1], wgt.shape[1]), numpy.float_)
                for wj in wgt:
                    wgt2 += numpy.outer(wj, wj)
                ans[iw] = dot(wgt2, delta[iw])
            return ans
        chiv.nf = nw
    else:
        def chiv(p, fd=fdata):
            cdef Py_ssize_t i1, i2
            cdef numpy.ndarray[INTP_TYPE, ndim=1] iw
            cdef numpy.ndarray[numpy.float_t, ndim=1] wgts
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt
            cdef numpy.ndarray ans, delta
            delta = fcn(p) - fd.mean
            if delta.dtype == object:
                ans = numpy.zeros(nw, object)
            else:
                ans = numpy.zeros(nw, numpy.float_)
            iw, wgts = fd.inv_wgts[0]
            i1 = 0
            i2 = len(iw)
            if i2 > 0:
                ans[i1:i2] = wgts * delta[iw]
            for iw, wgt in fd.inv_wgts[1:]:
                i1 = i2
                i2 += len(wgt)
                ans[i1:i2] = dot(wgt, delta[iw])
            return ans
        def chivw(p, fd=fdata):
            cdef numpy.ndarray[INTP_TYPE, ndim=1] iw
            cdef numpy.ndarray[numpy.float_t, ndim=1] wgts, wj
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt
            cdef numpy.ndarray[numpy.float_t, ndim=2] wgt2
            cdef numpy.ndarray ans, delta
            delta = fcn(p) - fd.mean
            if delta.dtype == object:
                ans = numpy.zeros(niw, object)
            else:
                ans = numpy.zeros(niw, numpy.float_)
            iw, wgts = fd.inv_wgts[0]
            if len(iw) > 0:
                ans[iw] = wgts ** 2 * delta[iw]
            for iw, wgt in fd.inv_wgts[1:]:
                wgt2 = numpy.zeros((wgt.shape[1], wgt.shape[1]), numpy.float_)
                for wj in wgt:
                    wgt2 += numpy.outer(wj, wj)
                ans[iw] = dot(wgt2, delta[iw])
            return ans
        chiv.nf = nw
    return chiv, chivw
