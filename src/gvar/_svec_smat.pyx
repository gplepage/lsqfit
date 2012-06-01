# Created by Peter Lepage (Cornell University) on 2012-05-31.
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

cimport numpy
cimport cython

cdef extern from "stdlib.h":
    void* malloc(int size)
    void* realloc(void* p,int size)
    void free(void* p)
    int sizeof(double)

cdef extern from "string.h":
    void* memset(void* mem,int val,int bytes)


cdef class svec:
    """ sparse vector --- for GVar derivatives (only)"""
    # cdef svec_element * v
    # cdef readonly usigned int size ## number of elements in v
    
    def __cinit__(self,unsigned int size,*arg,**karg):
        self.v = <svec_element *> malloc(size*sizeof(self.v[0]))
        memset(self.v,0,size*sizeof(self.v[0]))
        self.size = size
    ##
    def __dealloc__(self):
        free(<void *> self.v)
    ##
    def __len__(self):
        """ """
        cdef unsigned int i
        if self.size==0:
            return 0
        else:
            return max([self.v[i].i for i in range(self.size)])
    ##
    cpdef svec clone(self):
        cdef svec ans
        cdef unsigned int i
        ans = svec(self.size)
        for i in range(self.size):
            ans.v[i].v = self.v[i].v
            ans.v[i].i = self.v[i].i
        return ans
    ##
    cpdef numpy.ndarray[numpy.int_t,ndim=1] indices(self):
        cdef unsigned int i
        cdef numpy.ndarray [numpy.int_t,ndim=1] ans
        ans = numpy.zeros(self.size,int)
        n = 0
        for i in range(self.size):
            ans[i] = self.v[i].i
        return ans
    ##
    cpdef numpy.ndarray[numpy.double_t,ndim=1] toarray(self,unsigned int msize=0):
        """ Create numpy.array version of self, padded with zeros to length
        msize if msize is not None and larger than the actual size.
        """
        cdef unsigned int i,nsize
        cdef numpy.ndarray[numpy.double_t,ndim=1] ans
        if self.size==0:
            return numpy.zeros(msize,float)
        nsize = max(self.v[self.size-1].i+1,msize)
        ans = numpy.zeros(nsize,float)
        for i in range(self.size):
            ans[self.v[i].i] = self.v[i].v
        return ans
    ##
    cpdef _assign(self,numpy.ndarray[numpy.double_t,ndim=1] v,
                     numpy.ndarray[numpy.int_t,ndim=1] idx):
        """ Assign v and idx to self.v[i].v and self.v[i].i.
            
        Assumes that len(v)==len(idx)==self.size and idx sorted 
        """
        for i in range(self.size): 
            self.v[i].v = v[i]
            self.v[i].i = idx[i]
    ##
    def assign(self,v,idx=None):
        """ assign v and idx to self.v[i].v and self.v[i].i """
        cdef unsigned int nv
        nv = len(v)
        assert nv==len(idx) and nv==self.size,"v,idx length mismatch"
        if nv>0:
            idx,v = zip(*sorted(zip(idx,v)))
            idx = list(idx)
            v = list(v)
            for i in range(self.size): 
                self.v[i].v = v[i]
                self.v[i].i = idx[i]
    ##
    cpdef double dot(svec self,svec v):
        """ Compute dot product of self and v: <self|v> """
        cdef svec va,vb
        cdef unsigned int ia,ib
        cdef double ans
        va = self
        vb = v
        ia = 0
        ib = 0
        ans = 0.0
        if va.size==0 or vb.size==0:
            return 0.0
        if va.v[va.size-1].i<vb.v[0].i or vb.v[vb.size-1].i<va.v[0].i:
            return ans 
        while ia<va.size and ib<vb.size:
            if va.v[ia].i==vb.v[ib].i:
                ans += va.v[ia].v*vb.v[ib].v
                ia += 1
                ib += 1
            elif va.v[ia].i<vb.v[ib].i:
                ia += 1
            else:
                ib += 1
        return ans
    ##
    cpdef svec add(svec self,svec v,double a=1.,double b=1.):
        """ Compute a*self + b*v. """
        cdef svec va,vb
        cdef unsigned int ia,ib
        va = self
        vb = v
        ans = svec(va.size+vb.size)     # could be too big
        ia = 0
        ib = 0
        ians = 0
        while ia<va.size or ib<vb.size:
            if va.v[ia].i==vb.v[ib].i:
                ans.v[ians].i = va.v[ia].i
                ans.v[ians].v = a*va.v[ia].v+b*vb.v[ib].v
                ians += 1
                ia += 1
                ib += 1
                if ia>=va.size:
                    while ib<vb.size:
                        ans.v[ians].i = vb.v[ib].i
                        ans.v[ians].v = b*vb.v[ib].v
                        ib += 1
                        ians += 1
                    break
                elif ib>=vb.size:
                    while ia<va.size:
                        ans.v[ians].i = va.v[ia].i
                        ans.v[ians].v = a*va.v[ia].v
                        ia += 1
                        ians += 1
                    break
            elif va.v[ia].i<vb.v[ib].i:
                ans.v[ians].i = va.v[ia].i
                ans.v[ians].v = a*va.v[ia].v
                ians += 1
                ia += 1
                if ia>=va.size: 
                    while ib<vb.size:
                        ans.v[ians].i = vb.v[ib].i
                        ans.v[ians].v = b*vb.v[ib].v
                        ib += 1
                        ians += 1
                    break
            else:
                ans.v[ians].i = vb.v[ib].i
                ans.v[ians].v = b*vb.v[ib].v
                ians += 1
                ib += 1
                if ib>=vb.size:
                    while ia<va.size:
                        ans.v[ians].i = va.v[ia].i
                        ans.v[ians].v = a*va.v[ia].v
                        ia += 1
                        ians += 1
                    break
        ans.size = ians
        ans.v = <svec_element *> realloc(<void*> ans.v,
                                         ans.size*sizeof(self.v[0]))
        return ans
    ##
    cpdef svec mul(svec self,double a):
        """ Compute a*self. """
        cdef unsigned int i
        ans = svec(self.size)
        for i in range(self.size):
            ans.v[i].i = self.v[i].i
            ans.v[i].v = a*self.v[i].v
        return ans
    ##
    
    
cdef class smat:
    """ sym. sparse matrix --- for GVar covariance matrices (only) """
    # cdef object rowlist
    
    def __init__(self):
        self.rowlist = []
    ##
    def __len__(self):
        """ Dimension of matrix. """
        return len(self.rowlist)
    ##
    cpdef numpy.ndarray[numpy.int_t,ndim=1] append_diag(self,
                                    numpy.ndarray[numpy.double_t,ndim=1] d):
        """ Add d[i] along diagonal. """
        cdef unsigned int i,nr
        cdef numpy.ndarray[numpy.double_t,ndim=1] v
        cdef numpy.ndarray[numpy.int_t,ndim=1] idx,vrange
        idx = numpy.zeros(1,int)
        nr = len(self.rowlist)
        v = numpy.zeros(1,float)
        vrange = numpy.arange(nr,nr+d.shape[0],dtype=int)
        for i in range(d.shape[0]):
            v[0] = d[i]
            idx[0] = len(self.rowlist)
            self.rowlist.append(svec(1))
            self.rowlist[-1]._assign(v,idx)
        return vrange
    ##
    cpdef numpy.ndarray[numpy.int_t,ndim=1] append_diag_m(self,
                                    numpy.ndarray[numpy.double_t,ndim=2] m):
        """ Add matrix m on diagonal. """
        cdef unsigned int i,j,nr,nm
        cdef numpy.ndarray[numpy.double_t,ndim=1] v
        cdef numpy.ndarray[numpy.int_t,ndim=1] idx,vrange
        assert m.shape[0]==m.shape[1],"m must be square matrix"
        nm = m.shape[0]
        idx = numpy.zeros(nm,int)
        v = numpy.zeros(nm,float)
        nr = len(self.rowlist)
        vrange = numpy.arange(nr,nr+nm,dtype=int)
        for i in range(nm):
            idx[i] = nr+i
        for i in range(nm):
            for j in range(nm):
                v[j] = m[i,j]
            self.rowlist.append(svec(nm))
            self.rowlist[-1]._assign(v,idx)
        return vrange
    ##
    cpdef double expval(self,svec vv):
        """ Compute expectation value <vv|self|vv>. """
        cdef unsigned int i
        cdef svec row
        cdef double ans
        ans = 0.0
        for i in range(vv.size):
            row = self.rowlist[vv.v[i].i]
            ans += row.dot(vv)*vv.v[i].v
        return ans
    ##
    cpdef svec dot(self,svec vv):
        """ Compute dot product self|vv>. """
        cdef numpy.ndarray[numpy.double_t,ndim=1] v
        cdef numpy.ndarray[numpy.int_t,ndim=1] idx
        cdef unsigned int nr
        nr = len(self.rowlist)
        v = numpy.zeros(nr,float)
        idx = numpy.zeros(nr,int)
        size = 0
        for i,row in enumerate(self.rowlist):
            rowv = row.dot(vv)
            if rowv!=0.0:
                idx[size] = i
                v[size] = rowv
                size += 1
        ans = svec(size)
        ans._assign(v[:size],idx[:size])
        return ans
    ##   
    cpdef numpy.ndarray[numpy.double_t,ndim=2] toarray(self):
        """ Create numpy ndim=2 array version of self. """
        cdef numpy.ndarray[numpy.double_t,ndim=2] ans
        cdef unsigned int nr = len(self.rowlist)
        ans = numpy.zeros((nr,nr),float)
        for i in range(nr):
            row = self.rowlist[i].toarray()
            ans[i][:len(row)] = row
        return ans
    ##



