# Copyright (c) 2011 G. Peter Lepage.
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

cdef struct svec_element:
    double v
    unsigned int i

cdef class svec:
    cdef svec_element * v
    cdef readonly int size
    cpdef numpy.ndarray[numpy.double_t,ndim=1] toarray(svec,unsigned int msize=?)
    cpdef numpy.ndarray[numpy.int_t,ndim=1] indices(svec)
    cpdef _assign(self,numpy.ndarray[numpy.double_t,ndim=1],
                     numpy.ndarray[numpy.int_t,ndim=1])
    cpdef double dot(svec,svec)
    cpdef svec clone(svec)
    cpdef svec add(svec,svec,double a=*,double b=*)
    cpdef svec mul(svec self,double a)
   
   
cdef class smat:
    cdef object rowlist
    cpdef numpy.ndarray[numpy.int_t,ndim=1] append_diag(self,
                                        numpy.ndarray[numpy.double_t,ndim=1])
    cpdef numpy.ndarray[numpy.int_t,ndim=1] append_diag_m(self,
                                        numpy.ndarray[numpy.double_t,ndim=2])
    cpdef svec dot(self,svec)
    cpdef double expval(self,svec)
    cpdef numpy.ndarray[numpy.double_t,ndim=2] toarray(self)

cdef class GVar:
    cdef double v   
    cdef svec d      
    cdef readonly smat cov  
    cpdef GVar clone(self)
