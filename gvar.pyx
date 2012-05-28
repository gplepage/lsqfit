# c#ython: profile=True
# remove extra # above for profiling
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

# Created by Peter Lepage (Cornell University) on 2011-08-17.
# Copyright (c) 2011-2012 G. Peter Lepage.
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

import sys
import collections
import fileinput
PY3 = True if sys.version>'3' else False    

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if b >= a else b

cimport numpy
cimport cython

cdef extern from "stdlib.h":
    void* malloc(int size)
    void* realloc(void* p,int size)
    void free(void* p)
    int sizeof(double)

cdef extern from "string.h":
    void* memset(void* mem,int val,int bytes)

cdef extern from "math.h":
    double c_pow "pow" (double x,double y)
    double c_sin "sin" (double x)
    double c_cos "cos" (double x)
    double c_tan "tan" (double x)
    double c_sinh "sinh" (double x)
    double c_cosh "cosh" (double x)
    double c_tanh "tanh" (double x)
    double c_log "log" (double x)
    double c_exp "exp" (double x)
    double c_sqrt "sqrt" (double x)
    double c_asin "asin" (double x)
    double c_acos "acos" (double x)
    double c_atan "atan" (double x)


import numpy
from numpy import sin,cos,tan,exp,log,sqrt
from numpy import sinh,cosh,tanh,arcsin,arccos,arctan
from numpy import arcsinh,arccosh,arctanh
from re import compile as _compile
import collections
import copy

try:
    import powerseries
    _ARRAY_TYPES = [numpy.ndarray,powerseries.PowerSeries]
except:
    _ARRAY_TYPES = [numpy.ndarray]
   
## svec and smat --- sparse vectors and matrices ##
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
##

## GVar ## 
cdef class GVar:
    # cdef double v     -- value or mean
    # cdef svec d       -- vector of derivatives
    # cdef readonly smat cov    -- covariance matrix
    
    def __cinit__(self,double v,svec d,smat cov):
        self.v = v
        self.d = d
        self.cov = cov
    ##
    cpdef GVar clone(self):
        return GVar(self.v,self.d,self.cov)
    ##
    def __repr__(self):
        # return "construct_gvar(%s,%s,%s)" % (repr(self.mean),repr(self.der),repr(self.cov))
        return str(self)
    ##
    def __str__(self):
        """ Convert to string with format: mean +- std-dev. """
        return "%g +- %g"%(self.mean,self.sdev)
    ##
    def __hash__(self):
        return id(self)
    ##
    def __richcmp__(xx,yy,op):
        """ only == and != defined """
        if (op not in [2,3]) or not (isinstance(xx,GVar) 
                                    and isinstance(yy,GVar)):
            raise TypeError("unorderable types")
        if ((xx.cov is yy.cov) and (xx.mean==yy.mean) 
                and numpy.all(xx.der==yy.der)):
            return True if op==2 else False
        else:
            return True if op==3 else False
    ##
    def __call__(self):
        """ Generate random number from ``self``'s distribution."""
        return numpy.random.normal(self.mean,self.sdev)
    ##
    def __neg__(self):
        return GVar(-self.v,self.d.mul(-1.),self.cov)
    ##
    def __pos__(self):
        return self
    ##
    def __add__(xx,yy):
        cdef GVar x,y
        cdef unsigned int i,nx,di,ny
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                return GVar(x.v+y.v,x.d.add(y.d),x.cov)
            else:
                x = xx
                return GVar(x.v+yy,x.d,x.cov)
        elif isinstance(yy,GVar):
            y = yy
            return GVar(y.v+xx,y.d,y.cov)
        else: 
            return NotImplemented
    ##
    def __sub__(xx,yy):
        cdef GVar x,y
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                return GVar(x.v-y.v,x.d.add(y.d,1.,-1.),x.cov)
            else:
                x = xx
                return GVar(x.v-yy,x.d,x.cov)
        elif isinstance(yy,GVar):
            y = yy
            return GVar(xx-y.v,y.d.mul(-1.),y.cov)
        else: 
            return NotImplemented
    ##
    def __mul__(xx,yy):
        cdef GVar x,y
        
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                return GVar(x.v*y.v,x.d.add(y.d,y.v,x.v),x.cov)
            else:
                x = xx
                return GVar(x.v*yy,x.d.mul(yy),x.cov)
        elif isinstance(yy,GVar):
            y = yy
            return GVar(xx*y.v,y.d.mul(xx),y.cov)
        else: 
            return NotImplemented
    ##
    # truediv and div are the same --- 1st is for python3, 2nd for python2
    def __truediv__(xx,yy):
        cdef GVar x,y
        cdef double xd,yd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                return GVar(x.v/y.v,x.d.add(y.d,1./y.v,-x.v/y.v**2),x.cov)
            else:
                x = xx
                yd=yy
                return GVar(x.v/yd,x.d.mul(1./yd),x.cov)
        elif isinstance(yy,GVar):
            y = yy
            xd=xx
            return GVar(xd/y.v,y.d.mul(-xd/y.v**2),y.cov)
        else: 
            return NotImplemented
    ##
    def __div__(xx,yy):
        cdef GVar x,y
        cdef double xd,yd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                return GVar(x.v/y.v,x.d.add(y.d,1./y.v,-x.v/y.v**2),x.cov)
            else:
                x = xx
                yd=yy
                return GVar(x.v/yd,x.d.mul(1./yd),x.cov)
        elif isinstance(yy,GVar):
            y = yy
            xd=xx
            return GVar(xd/y.v,y.d.mul(-xd/y.v**2),y.cov)
        else: 
            return NotImplemented
    ##
    def __pow__(xx,yy,zz):
        cdef GVar x,y
        cdef double ans,f1,f2,yd,xd
        if type(yy) in _ARRAY_TYPES:
            return NotImplemented   # let ndarray handle it
        elif isinstance(xx,GVar):
            if isinstance(yy,GVar):
                x = xx
                y = yy
                assert x.cov is y.cov,"Incompatible GVars."
                ans = c_pow(x.v,y.v)
                f1 = c_pow(x.v,y.v-1)*y.v
                f2 = ans*c_log(x.v)
                return GVar(ans,x.d.add(y.d,f1,f2),x.cov)
            else:
                x = xx
                yd= yy
                ans = c_pow(x.v,yd)
                f1 = c_pow(x.v,yd-1)*yy
                return GVar(ans,x.d.mul(f1),x.cov)
        elif isinstance(yy,GVar):
            y = yy
            xd= xx
            ans = c_pow(xd,y.v)
            f1 = ans*c_log(xd)
            return GVar(ans,y.d.mul(f1),y.cov)
        else: 
            return NotImplemented
    ##
    def sin(self):
        return GVar(c_sin(self.v),self.d.mul(c_cos(self.v)),self.cov)
    ##
    def cos(self):
        return GVar(c_cos(self.v),self.d.mul(-c_sin(self.v)),self.cov)
    ##
    def tan(self):
        cdef double ans = c_tan(self.v)
        return GVar(ans,self.d.mul(1+ans*ans),self.cov)
    ##
    def arcsin(self):
        return GVar(c_asin(self.v),self.d.mul(1./(1.-self.v**2)**0.5),self.cov)
    ##
    def asin(self): 
        return self.arcsin()
    ##
    def arccos(self):
        return GVar(c_acos(self.v),self.d.mul(-1./(1.-self.v**2)**0.5),self.cov)
    ##
    def acos(self): 
        return self.arccos()
    ##
    def arctan(self):
        return GVar(c_atan(self.v),self.d.mul(1./(1.+self.v**2)),self.cov)
    ##
    def atan(self):
        return self.arctan()
    ##
    def sinh(self):
        return GVar(c_sinh(self.v),self.d.mul(c_cosh(self.v)),self.cov)
    ##
    def cosh(self):
        return GVar(c_cosh(self.v),self.d.mul(c_sinh(self.v)),self.cov)
    ##
    def tanh(self):
        return GVar(c_tanh(self.v),self.d.mul(1./(c_cosh(self.v)**2)),self.cov)
    ##
    def arcsinh(self):
        return log(self+sqrt(self*self+1.))
    ##
    def asinh(self):
        return self.arcsinh()
    ##
    def arccosh(self):
        return log(self+sqrt(self*self-1.))
    ##
    def acosh(self):
        return self.arccosh()
    ##
    def arctanh(self):
        return log((1.+self)/(1.-self))/2.
    ##
    def atanh(self):
        return self.arctanh()
    ##
    def exp(self):
        cdef double ans
        ans = c_exp(self.v)
        return GVar(ans,self.d.mul(ans),self.cov)
    ##
    def log(self):
        return GVar(c_log(self.v),self.d.mul(1./self.v),self.cov)
    ##
    def sqrt(self):
        cdef double ans = c_sqrt(self.v)
        return GVar(ans,self.d.mul(0.5/ans),self.cov)
    ##
    def fmt(self,d=None,sep=''):
        """ Convert to string with format: ``mean(sdev)``. 
            
        Leading zeros in the standard deviation are omitted: for example,
        ``25.67 +- 0.02`` becomes ``25.67(2)``. Parameter ``d`` specifies
        how many digits follow the decimal point in the mean. Parameter
        ``sep`` is a string that is inserted between the ``mean`` and the
        ``(sdev)``. If ``d`` is ``None``, it is set automatically
        to the larger of ``int(1-log(self.sdev)/log(10))`` or ``0``; this
        will display the smallest number of digits needed to expose the
        error.
        """
        dv = self.sdev
        v = self.mean
        if d is None:
            d = int(1-log(dv)/log(10.))
            if d<0:
                d = 0
        if d is None or d<0 or d!=int(d):
            return self.__str__()
        fac = 10.**d
        v = round(v,d)
        if dv<1.0 and d>0:
            ft =  '%.'+str(d)+'f%s(%d)'
            dv = round(dv*fac,0)
            return ft % (v,sep,int(dv))
        else:
            dv = round(dv,d)
            ft = '%.'+str(d)+'f%s(%.'+str(d)+'f)'
            return ft % (v,sep,dv)
    ##
    def partialvar(self,*args):
        """ Compute partial variance due to |GVar|\s in ``args``.
            
        This method computes the part of ``self.var`` due to the |GVar|\s
        in ``args``. If ``args[i]`` is derived from other |GVar|\s, the
        variance coming from these is included in the result.
            
        :param args[i]: Variables contributing to the partial variance.
        :type args[i]: |GVar| or array/dictionary of |GVar|\s
        :returns: Partial variance due to all of ``args``.
        """
        cdef GVar ai
        cdef svec md
        cdef numpy.ndarray[numpy.int_t,ndim=1] dmask
        cdef numpy.ndarray[numpy.int_t,ndim=1] md_idx
        cdef numpy.ndarray[numpy.double_t,ndim=1] md_v
        cdef unsigned int i,j,md_size
        cdef unsigned dstart,dstop
        if self.d.size<=0:
            return 0.0
        dstart = self.d.v[0].i
        dstop = self.d.v[self.d.size-1].i+1
        ## create a mask = 1 if args[i].der component!=0; 0 otherwise ## 
        dmask = numpy.zeros(dstop-dstart,int)
        for a in args:
            if hasattr(a,'keys'):
                if not hasattr(a,'flat'):
                    a = BufferDict(a)
            else:
                a = numpy.asarray(a)
            for ai in a.flat:
                assert ai.cov is self.cov,"Incompatible |GVar|\s."
                for i in range(ai.d.size):
                    j = ai.d.v[i].i
                    if j<dstart:
                        continue
                    elif j>=dstop:
                        break
                    else:
                        dmask[j-dstart] |= 1
        ##
        ## create masked derivative vector for self ##
        md_size = 0
        md_idx = numpy.zeros(dstop-dstart,int)
        md_v = numpy.zeros(dstop-dstart,float)
        for i in range(self.d.size):
            if dmask[self.d.v[i].i-dstart]==0:
                continue
            else:
                md_idx[md_size] = self.d.v[i].i
                md_v[md_size] = self.d.v[i].v
                md_size += 1
        md = svec(md_size)
        md._assign(md_v[:md_size],md_idx[:md_size])
        ##
        return md.dot(self.cov.dot(md))
    ##
    def partialsdev(self,*args): 
        """ Compute partial standard deviation due to |GVar|\s in ``args``.
            
        This method computes the part of ``self.sdev`` due to the |GVar|\s
        in ``args``. If ``args[i]`` is derived from other |GVar|\s, the
        standard deviation coming from these is included in the result.
            
        :param args[i]: Variables contributing to the partial standard
            deviation.
        :type args[i]: |GVar| or array/dictionary of |GVar|\s
        :returns: Partial standard deviation due to ``args``.
        """
        ans = self.partialvar(*args)
        return ans**0.5 if ans>0 else -(-ans)**0.5
    ##
    property val:
        """ Mean value. """ 
        def __get__(self):
            return self.v
        ##
    property der:
        """ Array of derivatives with respect to  underlying (original)
        |GVar|\s. 
        """
        def __get__(self):
            return self.d.toarray(len(self.cov))
        ##
    property mean:
        """ Mean value. """
        def __get__(self):
            return self.v
        ##
    property sdev:
        """ Standard deviation. """
        def __get__(self):
            cdef double var = self.cov.expval(self.d) # self.d.dot(self.cov.dot(self.d))
            if var>=0:
                return c_sqrt(var)
            else:
                return -c_sqrt(-var)
        ##
    property var:
        """ Variance. """
        # @cython.boundscheck(False)
        def __get__(self):
            return self.cov.expval(self.d)  # self.d.dot(self.cov.dot(self.d))
        ##
    def dotder(self,numpy.ndarray[numpy.double_t,ndim=1] v not None):
        """ Return the dot product of ``self.der`` and ``v``. """
        cdef double ans = 0
        cdef unsigned int i
        for i in range(self.d.size):
            ans += v[self.d.v[i].i]*self.d.v[i].v
        return ans
    ## 
##

## BufferDict ##
BufferDictData = collections.namedtuple('BufferDictData',['slice','shape'])
""" Data type for BufferDict._data[k]. Note shape==None implies a scalar. """
    
class BufferDict(collections.MutableMapping):
    """ Dictionary whose data is packed into a 1-d buffer (numpy.array).
        
    A |BufferDict| object is a dictionary-like object whose values must either
    be scalars or arrays (like :mod:`numpy` arrays, with arbitrary shapes). 
    The scalars and arrays are assembled into different parts of a single 
    one-dimensional buffer. The various scalars and arrays are retrieved 
    using keys, as in a dictionary: *e.g.*,
        
        >>> a = BufferDict()
        >>> a['scalar'] = 0.0
        >>> a['vector'] = [1.,2.]
        >>> a['tensor'] = [[3.,4.],[5.,6.]]
        >>> print(a.flatten())              # print a's buffer
        [ 0.  1.  2.  3.  4.  5.  6.]
        >>> for k in a:                     # iterate over keys in a
        ...     print(k,a[k])
        scalar 0.0
        vector [ 1.  2.]
        tensor [[ 3.  4.]
         [ 5.  6.]]
        >>> a['vector'] = a['vector']*10    # change the 'vector' part of a
        >>> print(a.flatten())
        [  0.  10.  20.   3.   4.   5.   6.]
        
    The first four lines here could have been collapsed to one statement::
        
        a = BufferDict(scalar=0.0,vector=[1.,2.],tensor=[[3.,4.],[5.,6.]])
        
    or ::
        
        a = BufferDict([('scalar',0.0),('vector',[1.,2.]),
                        ('tensor',[[3.,4.],[5.,6.]])])
        
    where in the second case the order of the keys is preserved in ``a`` (that 
    is, ``BufferDict`` is an ordered dictionary).
        
    The keys and associated shapes in a |BufferDict| can be transferred to a
    different buffer, creating a new |BufferDict|: *e.g.*, using ``a`` from
    above,
        
        >>> buf = numpy.array([0.,10.,20.,30.,40.,50.,60.])
        >>> b = BufferDict(a,buf=buf)       # clone a but with new buffer
        >>> print(b['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        >>> b['scalar'] += 1
        >>> print(buf)
        [  1.  10.  20.  30.  40.  50.  60.]
        
    Note how ``b`` references ``buf`` and can modify it. One can also replace
    the buffer in the original |BufferDict| using, for example, 
    ``a.flat = buf``:
        
        >>> a.flat = buf
        >>> print(a['tensor'])
        [[ 30.  40.]
         [ 50.  60.]]
        
    ``a.flat`` is an iterator for the ``numpy`` array used for ``a``'s buffer.
    It can be used to access and change the buffer directly. ``a.flatten()``
    is a copy of the buffer.
         
    A |BufferDict| functions like a dictionary except: a) items cannot be
    deleted once inserted; b) all values must be either scalars or
    (:mod:`numpy`) arrays of scalars, where the scalars can be any noniterable
    type that works with :mod:`numpy` arrays; and c) any new value assigned to a
    key must have the same size and shape as the original value.
        
    Note that |BufferDict|\s can be pickled and unpickled even when they 
    store |GVar|\s (which themselves cannot be pickled separately).
    """
    def __init__(self,*args,**kargs):
        super(BufferDict, self).__init__()
        self.shape = None
        if len(args)==0:
            ## kargs are dictionary entries ##
            self._buf = numpy.array([],int)
            self._keys = []
            self._data = {}
            for k in sorted(kargs):
                self[k] = kargs[k]
            ##
        else:
            if len(args)==2 and len(kargs)==0:
                bd,buf = args
            elif len(args)==1 and len(kargs)==0:
                bd = args[0]
                buf = None
            elif len(args)==1 and 'buf' in kargs and len(kargs)==1:
                bd = args[0]
                buf = kargs['buf']
            else:
                raise ValueError("Bad arguments for BufferDict.")
            if isinstance(bd,BufferDict):
                ## make copy of BufferDict bd, possibly with new buffer ##
                self._keys = copy.copy(bd._keys)
                self._data = copy.copy(bd._data)
                self._buf = numpy.array(bd._buf) if buf is None else numpy.asarray(buf)
                if bd.size!=self.size:
                    raise ValueError("buf is wrong size: "+str(self.size)+" not "
                                        +str(bd.size))
                if self._buf.ndim!=1:
                    raise ValueError("buf must be 1-d: "+str(self._buf.shape))
                ##
            elif buf is None:
                self._buf = numpy.array([],int)
                self._keys = []
                self._data = {}
                ## add initial data ## 
                if hasattr(bd,"keys"):
                    ## bd a dictionary ##
                    for k in sorted(bd):
                        self[k] = bd[k]
                    ##
                else:
                    ## bd an array of tuples ##
                    if not all([(isinstance(bdi,tuple) and len(bdi)==2) for bdi in bd]):
                        raise ValueError("BufferDict argument must be dict or list of 2-tuples.")
                    for ki,vi in bd:
                        self[ki] = vi
                    ##
                ##
            else:
                raise ValueError("bd must be a BufferDict in BufferDict(bd,buf): "
                                    +str(bd.__class__))
    ##
    def __getstate__(self):
        """ Capture state for pickling when elements are GVars. """
        if len(self._buf)<1:
            return self.__dict__.copy()
        odict = self.__dict__.copy()
        if isinstance(self._buf[0],GVar):
            buf = odict['_buf']
            del odict['_buf']
            odict['_buf.mean'] = mean(buf)
            odict['_buf.cov'] = evalcov(buf)
        data = odict['_data']
        del odict['_data']
        odict['_data.tuple'] = {}
        for k in data:
            odict['_data.tuple'][k] = (data[k].slice,data[k].shape)
        return odict
    ##
    def __setstate__(self,odict):
        """ Restore state when unpickling when elements are GVars. """
        if '_buf.mean' in odict:
            buf = gvar(odict['_buf.mean'],odict['_buf.cov'])
            del odict['_buf.mean']
            del odict['_buf.cov']
            odict['_buf'] = buf
        if '_data.tuple' in odict:
            data = odict['_data.tuple']
            del odict['_data.tuple']
            odict['_data'] = {}
            for k in data:
                odict['_data'][k] = BufferDictData(slice=data[k][0],
                                                    shape=data[k][1])
        self.__dict__.update(odict)
    ##
    def add(self,k,v=None):
        """ Augment buffer with data ``v``, indexed by key ``k``.
            
        ``v`` is either a scalar or a :mod:`numpy` array (or a list or
        other data type that can be changed into a numpy.array).
        If ``v`` is a :mod:`numpy` array, it can have any shape.
            
        Same as ``self[k] = v`` except: 1) when ``v is None``, in which case k 
        is assumed to be a dictionary and each entry in it is added; and 
        2) when ``k`` is already used in ``self``, in which case a 
        ``ValueError`` is raised.
        """
        if v is None:
            if hasattr(k,'keys'):
                for kk in k:
                    self[kk] = k[kk]
            else:
                for ki,vi in k:
                    self[ki] = vi
        else:
            if k in self:
                raise ValueError("Key %s already used."%k)
            else:
                self[k] = v
    ##
    def __getitem__(self,k):
        """ Return piece of buffer corresponding to key ``k``. """
        if k not in self._data:
            raise KeyError("undefined key: %s" % k)
        if isinstance(self._buf,list):
            self._buf = numpy.array(self._buf)
        d = self._data[k]
        ans = self._buf[d.slice]
        return ans if d.shape is None else ans.reshape(d.shape)
    ##
    def __setitem__(self,k,v):
        """ Set piece of buffer corresponding to ``k`` to value ``v``. 
            
        The shape of ``v`` must equal that of ``self[k]``. If key ``k`` 
        is not in ``self``, use ``self.add(k,v)`` to add it.
        """
        if k not in self:
            # if not isinstance(self._buf,list):
            #     self._buf = list(self._buf)
            v = numpy.asarray(v)
            if v.shape==():
                ## add single piece of data ##
                self._data[k] = BufferDictData(slice=len(self._buf),shape=None)
                self._buf = numpy.append(self._buf,v)
                ##
            else:
                ## add array ##
                n = numpy.size(v)
                i = len(self._buf)
                self._data[k] = BufferDictData(slice=slice(i,i+n),shape=tuple(v.shape))
                self._buf = numpy.append(self._buf,v)
                ##
            self._keys.append(k)
        else:
            # if isinstance(self._buf,list):
            #     self._buf = numpy.array(self._buf)
            d = self._data[k]
            if d.shape is None:
                try:
                    self._buf[d.slice] = v
                except ValueError:
                    print("*** Not a scalar?",numpy.shape(v))
                    raise
            else:
                v = numpy.asarray(v)
                try:
                    self._buf[d.slice] = v.flat
                except ValueError:
                    print("*** Shape mismatch?",v.shape,"not",d.shape)
                    raise
    ##
    def __delitem__(self,k):
        raise NotImplementedError("Cannot delete items from BufferDict.")
    ##
    def __len__(self):
        """ Number of keys. """
        return len(self._keys)
    ##
    def __iter__(self):
        """ Iterator over the keys. """
        return iter(self._keys)
    ##
    def __contains__(self,k):
        """ True if k is a key in ``self``. """
        return k in self._data
    ##
    def __str__(self):
        return str(dict(self.items()))
    ##
    def __repr__(self):
        cn = self.__class__.__name__
        return cn+"("+repr([k for k in self.items()])+")"
    ##
    def _getflat(self):
        return self._buf.flat
    ##
    def _setflat(self,buf):
        """ Replaces buffer with buf if same size. """
        if len(buf)==len(self._buf):
            self._buf = numpy.asarray(buf)
            if self._buf.ndim!=1:
                raise ValueError("Buffer is not 1-d: "+str(self._buf.shape))
        else:
            raise ValueError("Buffer wrong size: %d not %d"
                            %(len(buf),len(self._buf)))
    ##
    flat = property(_getflat,_setflat,doc='Buffer array iterator.')
    def flatten(self):
        """ Copy of buffer array. """
        return numpy.array(self._buf)
    ##
    def _getbuf(self):      # obsolete --- for backwards compatibility
        return self._buf
    ##
    buf = property(_getbuf,_setflat,doc='Similar to flatten(), but reveals real buffer')
    def _getsize(self):
        """ Length of buffer. """
        return len(self._buf)
    ##
    size = property(_getsize,doc='Size of buffer array.')
    def slice(self,k):
        """ Return slice/index in ``self.flat`` corresponding to key ``k``."""
        return self._data[k].slice
    ##
    def isscalar(self,k):
        """ Return ``True`` if ``self[k]`` is scalar else ``False``."""
        return self._data[k].shape is None
    ##
##
##
    
## utility functions ##
def mean(g):
    """ Extract means from :class:`gvar.GVar`\s in ``g``.
        
    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Result has the same layout as ``g``.
    """
    cdef unsigned int i
    cdef GVar gi
    cdef numpy.ndarray[numpy.double_t,ndim=1] buf
    if isinstance(g,GVar):
        return g.mean
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = numpy.zeros(g.size,float)
    for i,gi in enumerate(g.flat):
        buf[i] = gi.v
    return BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)
##
def sdev(g):
    """ Extract standard deviations from :class:`gvar.GVar`\s in ``g``.
        
    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Result has the same layout as ``g``.
    """
    cdef unsigned int i
    cdef GVar gi
    cdef numpy.ndarray[numpy.double_t,ndim=1] buf
    if isinstance(g,GVar):
        return g.sdev
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = numpy.zeros(g.size,float)
    for i,gi in enumerate(g.flat):
        buf[i] = gi.sdev
    return BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)
##
def var(g):
    """ Extract variances from :class:`gvar.GVar`\s in ``g``.
        
    ``g`` can be a |GVar|, an array of |GVar|\s, or a dictionary containing
    |GVar|\s or arrays of |GVar|\s. Result has the same layout as ``g``.
    """
    cdef unsigned int i
    cdef GVar gi
    cdef numpy.ndarray[numpy.double_t,ndim=1] buf
    if isinstance(g,GVar):
        return g.var
    if hasattr(g,'keys'):
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    buf = numpy.zeros(g.size,float)
    for i,gi in enumerate(g.flat):
        buf[i] = gi.var
    return BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)
##
def orthogonal(g1,g2):
    """ Return ``True`` if ``g1`` and ``g2`` involve unrelated |GVar|\s
        
    ``g1`` and ``g2`` can be |GVar|\s, arrays of |GVar|\s, or dictionaries
    containing |GVar|\s or arrays of |GVar|\s.
    """
    cdef GVar g
    s = [set(),set()]
    for i,gi in enumerate([g1,g2]):
        if not hasattr(gi,'flat'):
            if isinstance(gi,GVar):
                gi = numpy.array([gi])
            elif hasattr(gi,'keys'):
                gi = BufferDict(gi)
            else:
                gi = numpy.asarray(gi)
        for g in gi.flat:
            s[i].update(g.d.indices())
    return s[0].isdisjoint(s[1])
##
def evalcov(g):
    """ Compute covariance matrix for elements of 
    array/dictionary ``g``.
        
    If ``g`` is an array of |GVar|\s, ``evalcov`` returns the
    covariance matrix as an array with shape ``g.shape+g.shape``.
    If ``g`` is a dictionary whose values are |GVar|\s or arrays of 
    |GVar|\s, the result is a doubly-indexed dictionary where 
    ``cov[k1,k2]`` is the covariance for ``g[k1]`` and ``g[k2]``.
    """
    cdef int a,b,ng,i,j,nc
    cdef numpy.ndarray[numpy.double_t,ndim=2] ans
    cdef numpy.ndarray[numpy.double_t,ndim=1] rowda
    cdef numpy.ndarray[numpy.int8_t,ndim=1] rowda_empty
    cdef GVar ga,gb
    cdef svec da,db,row
    cdef smat cov
    if hasattr(g,"keys"):
        ## convert g to list and call evalcov; repack as double dict ##
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        gcov = evalcov(g.flat)
        ansd = BufferDict()
        for k1 in g:
            for k2 in g:
                ansd[k1,k2] = gcov[g.slice(k1),g.slice(k2)]
        return ansd
        ##
    g = numpy.asarray(g)
    g_shape = g.shape
    g = g.flat
    ng = len(g)
    ans = numpy.zeros((ng,ng),float)
    cov = g[0].cov 
    nc = len(cov.rowlist)
    covd = []
    if True:
        rowda = numpy.zeros(nc,float)   ## stores rowlist[i].dot(da)s
        rowda_empty = numpy.ones(nc,numpy.int8)
        for a in range(ng):
            ga = g[a]
            da = ga.d
            rowda_empty.fill(True)  ## reset
            for b in range(a,ng):
                gb = g[b]
                db = gb.d
                for i in range(db.size):
                    j = db.v[i].i
                    if rowda_empty[j]:   
                        row = cov.rowlist[j]
                        rowda_empty[j] = False
                        rowda[j] = row.dot(da)
                    ans[a,b] += rowda[j]*db.v[i].v
                if a!=b:
                    ans[b,a] = ans[a,b]
    else:      
        for a in range(ng):
            ga = g[a]
            covd.append(cov.dot(ga.d))
            ans[a,a] = ga.d.dot(covd[-1])
            for b in range(a):
                ans[a,b] = ga.d.dot(covd[b])
                ans[b,a] = ans[a,b]
    return ans.reshape(2*g_shape)
##
def wsum_der(numpy.ndarray[numpy.double_t,ndim=1] wgt,glist):
    """ weighted sum of |GVar| derivatives """
    cdef GVar g
    cdef smat cov
    cdef double w
    cdef unsigned int ng,i
    cdef numpy.ndarray[numpy.double_t,ndim=1] ans
    ng = len(glist)
    assert ng==len(wgt),"wgt and glist have different lengths."
    cov = glist[0].cov
    ans = numpy.zeros(len(cov),float)
    for i in range(wgt.shape[0]):
        w = wgt[i]
        g = glist[i]
        assert g.cov is cov,"Incompatible |GVar|\s."
        for i in range(g.d.size):
            ans[g.d.v[i].i] += w*g.d.v[i].v
    return ans 
##
def wsum_gvar(numpy.ndarray[numpy.double_t,ndim=1] wgt,glist):
    """ weighted sum of |GVar|\s """
    cdef svec wd
    cdef double wv,w
    cdef GVar g
    cdef smat cov
    cdef unsigned int ng,i,nd,size
    cdef numpy.ndarray[numpy.double_t,ndim=1] der
    cdef numpy.ndarray[numpy.int_t,ndim=1] idx
    ng = len(glist)
    assert ng==len(wgt),"wgt and glist have different lengths."
    cov = glist[0].cov
    der = numpy.zeros(len(cov),float)
    wv = 0.0
    for i in range(ng): #w,g in zip(wgt,glist):
        w = wgt[i]
        g = glist[i]
        assert g.cov is cov,"Incompatible |GVar|\s."
        wv += w*g.v
        for i in range(g.d.size):
            der[g.d.v[i].i] += w*g.d.v[i].v
    idx = numpy.zeros(len(cov),int) # der.nonzero()[0]
    nd = 0
    for i in range(der.shape[0]):
        if der[i]!=0:
            idx[nd] = i
            nd += 1
    wd = svec(nd)
    for i in range(nd):
        wd.v[i].i = idx[i]
        wd.v[i].v = der[idx[i]]
    return GVar(wv,wd,cov)
##
def fmt_values(outputs,ndigit=3):
    """ Tabulate :class:`gvar.GVar`\s in ``outputs``. 
        
    :param outputs: A dictionary of :class:`gvar.GVar` objects. 
    :param ndigit: Number of digits displayed in table; if ``None``, 
        use ``str(outputs[k])`` instead.
    :type ndigit: ``int`` or ``None``
    :returns: A table (``str``) containing values and standard 
        deviations for variables in ``outputs``, labeled by the keys
        in ``outputs``.
    """
    ans = "Values:\n"
    for vk in outputs:
        ans += "%19s: %-20s\n" % (vk,outputs[vk].fmt(ndigit))
    return ans
##
def fmt_errorbudget(outputs,inputs,ndigit=2,percent=True):
    """ Tabulate error budget for ``outputs[ko]`` due to ``inputs[ki]``.
        
    :param outputs: Dictionary of |GVar|\s for which partial standard 
        deviations are computed.
    :param inputs: Dictionary of lists of: |GVar|\s or arrays/dictionaries 
        of |GVar|\s. The partial standard deviation due to the input
        quantities in each list is tabulated for each output quantity in
        ``outputs``.
    :param ndigit: Number of decimal places displayed in table.
    :type ndigit: ``int``
    :param percent: Tabulate % errors if ``percent is True``; otherwise
        tabulate the errors themselves.
    :type percent: boolean
    :returns: A table (``str``) containing the error budget. 
        Output variables are labeled by the keys in ``outputs``
        (columns); sources of uncertainty are labeled by the keys in
        ``inputs`` (rows).
    """
    ## collect partial errors ##
    err = {}
    for ko in outputs:
        for ki in inputs:
            inputs_ki = inputs[ki]
            if hasattr(inputs_ki,'keys') or not hasattr(inputs_ki,'__iter__'):
                inputs_ki = [inputs_ki]
            err[ko,ki] = outputs[ko].partialvar(*inputs_ki)**0.5                
    ##
    ## form table ##
    lfmt = "%19s:"+len(outputs)*("%10."+str(ndigit)+"f")+"\n"
    hfmt = "%20s"+len(outputs)*("%10s")+"\n"
    if percent:
        val = numpy.array([abs(outputs[vk].mean) 
                                for vk in outputs])/100.
        ans = "Partial % Errors:\n"
    else:
        val = 1.
        ans = "Partial Errors:\n"
    ans += hfmt % (("",)+tuple(outputs.keys()))
    ans += (20+len(outputs)*10)*'-'+"\n"
    for ck in inputs:
        verr = numpy.array([err[vk,ck] for vk in outputs])/val
        ans += lfmt%((ck,)+tuple(verr))
    ans += (20+len(outputs)*10)*'-'+"\n"
    ans += lfmt%(("total",)+tuple(numpy.array([outputs[vk].sdev 
                                    for vk in outputs])/val))
    ##
    return ans
##
##

## GVar factory functions ##
_GDEV_LIST = []
    
def switch_gvar():
    """ Switch :func:`gvar.gvar` to new :class:`gvar.GVarFactory`.
        
    :returns: New :func:`gvar.gvar`.
    """
    global gvar
    _GDEV_LIST.append(gvar)
    gvar = gvar_factory()
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
        raise IndexError("no previous gvar")
    return gvar
##
    
def gvar_factory(cov=None):
    """ Return new function for creating |GVar|\s (to replace 
    :func:`gvar.gvar`). 
        
    If ``cov`` is specified, it is used as the covariance matrix
    for new |GVar|\s created by the function returned by 
    ``gvar_factory(cov)``.
    """
    return GVarFactory(cov)
##
    
_GDEVre1 = _compile(r"([-+]?[0-9]*)[.]([0-9]+)\s*\(([.0-9]+)\)")
_GDEVre2 = _compile(r"([-+]?[0-9]+)\s*\(([0-9]*)\)")
_GDEVre3 = _compile(r"([-+]?[0-9.]*[e]*[+-]?[0-9]*)" + "\s*[+][-]\s*"
                        +"([-+]?[0-9.]*[e]*[+-]?[0-9]*)")
                        
class GVarFactory:
    """ Create one or more new |GVar|\s.
        
    Each of the following creates new |GVar|\s:
        
    .. function:: gvar(x,xsdev)
        
        Returns a |GVar| with mean ``x`` and standard deviation ``xsdev``.
        Returns an array of |GVar|\s if ``x`` and ``xsdev`` are arrays
        with the same shape; the shape of the result is the same as the
        shape of ``x``. 
        
    .. function:: gvar(x,xcov)
        
        Returns an array of |GVar|\s with means given by array ``x`` 
        and a covariance matrix given by array ``xcov``, where 
        ``xcov.shape = x.shape+x.shape``. The result has the same shape
        as ``x``.
        
    .. function:: gvar((x,xsdev))
        
        Returns a |GVar| with mean ``x`` and standard deviation ``xsdev``.
        
    .. function:: gvar(xstr)
        
        Returns a |GVar| corresponding to string ``xstr`` which is 
        either of the form ``"xmean +- xsdev"`` or ``"x(xerr)"`` (see
        :meth:`GVar.fmt`.)
        
    .. function:: gvar(xgvar)
        
        Returns |GVar| ``xgvar`` unchanged.
        
    .. function:: gvar(dict(x=xstr,y=(x,xsdev)...))
        
        Returns a dictionary (:class:`BufferDict`) ``b`` where
        ``b['x'] = gvar(xstr)``, ``b['y']=gvar(x,xsdev)``...
        
    .. function:: gvar([[(x1,x1sdev)...]...])
        
        Returns an array ``numpy.array([[gvar(x1,x1sdev)...]...])``. Works
        for arrays of any shape.
            
    .. function:: gvar([[x1str...]...])
        
        Returns an array ``numpy.array([[gvar(xstr1)...]...])`` where
        ``x1str...`` are strings. Works for arrays of any shape.
            
    ``gvar.gvar`` is actually an object of type :class:`gvar.GVarFactory`.  
    """
    def __init__(self,cov=None):
        if cov is None:
            self.cov = smat()
        else:
            assert isinstance(cov,smat),"cov not type gvar.smat"
            self.cov = cov
    ##
    def __call__(self,*args):
        cdef unsigned int nx,i,nd
        cdef svec der
        cdef smat cov
        cdef GVar gd
        cdef numpy.ndarray[numpy.double_t,ndim=1] d
        cdef numpy.ndarray[numpy.double_t,ndim=1] d_v
        cdef numpy.ndarray[numpy.int_t,ndim=1] d_idx
        
        if len(args)==2:
            ## (x,xsdev) or (xarray,sdev-array) or (xarray,cov) ##
            ## unpack arguments and verify types ##
            try:
                x = numpy.asarray(args[0],float)
                xsdev = numpy.asarray(args[1],float)
            except (ValueError,TypeError):
                raise TypeError(    #):
                        "Arguments must be numbers or arrays of numbers")
            ##
            if len(x.shape)==0:
                ## single gvar from x and xsdev ##
                if len(xsdev.shape)!=0:
                    raise ValueError("x and xsdev different shapes.")
                idx = self.cov.append_diag(numpy.array([xsdev**2]))
                der = svec(1)
                der.v[0].i = idx[0]
                der.v[0].v = 1.0
                return GVar(x,der,self.cov)
                ##
            else:
                ## array of gvars from x and sdev/cov arrays ##
                nx = len(x.flat)
                if x.shape==xsdev.shape:  # x,sdev
                    idx = self.cov.append_diag(xsdev.reshape(nx)**2)
                elif xsdev.shape==2*x.shape: # x,cov
                    idx = self.cov.append_diag_m(xsdev.reshape(nx,nx))
                else:
                    raise ValueError("Argument shapes mismatched: "+
                        str(x.shape)+' '+str(xsdev.shape))
                d = numpy.ones(nx,float)
                ans = []
                for i in range(nx):
                    der = svec(1)
                    der.v[0].i = idx[i]
                    der.v[0].v = 1.0
                    ans.append(GVar(x.flat[i],der,self.cov))
                return numpy.array(ans).reshape(x.shape)
                ##
            ##
        elif len(args)==1:
            x = args[0]
            if isinstance(x,str):
                ## case 1: x is a string like "3.72(41)" or "3.2 +- 4" ##
                x = x.strip()
                try:
                    x,y,z = _GDEVre1.match(x).groups()
                    if x=='':
                        x = '0'
                    fac = 1./10.**len(y)
                    if y=='':
                        y = '0'
                    if '.' in z:
                        efac = 1.
                    else:
                        efac = fac
                    x,y,z = float(x),float(y),float(z)
                    if x>=0:
                        return self(x+y*fac,z*efac)
                    else:
                        return self(x-y*fac,z*efac)
                except AttributeError:
                    try:
                        x,z = _GDEVre2.match(x).groups()
                        return self(float(x),float(z))
                    except AttributeError:
                        try:
                            x,z = _GDEVre3.match(x).groups()
                            return self(float(x),float(z))
                        except:
                            raise ValueError(  # )
                                    "Poorly formatted gvar string: "+x)
                ##
            elif isinstance(x,GVar):
                ## case 2: x is a GVar ##
                return x
                ##
            elif isinstance(x,tuple) and len(x)==2:
                ## case 3: x = (x,sdev) tuple ##
                return self(x[0],x[1])
                ##
            elif hasattr(x,'keys'):
                ## case 4: x is a dictionary ##
                ans = BufferDict()
                for k in x:
                    ans[k] = self(x[k])
                return ans
                ##
            elif hasattr(x,'__iter__'):
                ## case 5: x is an array ##
                try:
                    xa = numpy.asarray(x)
                except ValueError:
                    xa = numpy.asarray(x,object)
                if xa.size==0:
                    return xa
                if xa.shape[-1]==2 and xa.dtype!=object and xa.ndim>1:
                    ## array of tuples? ##
                    xxa = numpy.empty(xa.shape[:-1],object)
                    xxa[:] = x
                    if all(type(xxai)==tuple for xxai in xxa.flat):
                        return self(xa[...,0],xa[...,1])
                    ##
                return numpy.array([xai if isinstance(xai,GVar) else self(xai) 
                                    for xai in xa.flat]).reshape(xa.shape)
                ##        
            else:   ## case 6: a number
                return self(x,0.0)
        elif len(args)==3:
            ## (x,der,cov) ##
            try:
                x = numpy.asarray(args[0],float)
                d = numpy.asarray(args[1],float).flatten()
            except (ValueError,TypeError,AssertionError):
                raise TypeError("Value and derivatives not numbers.")
            assert len(x.shape)==0,"Value not a number."
            cov = args[2]
            assert isinstance(cov,smat),"cov not type gvar.smat."
            assert len(cov)>=len(d),"length mismatch between d and cov"
            d_idx = d.nonzero()[0]
            d_v = d[d_idx]
            nd = len(d_idx)
            der = svec(nd)
            for i in range(nd):
                der.v[i].i = d_idx[i]
                der.v[i].v = d_v[i]
            return GVar(x,der,cov)
            ##
        else:
            raise ValueError("Wrong number of arguments: "+str(len(args)))
    ## 
##
    
gvar = gvar_factory()   
    
def rebuild(g,corr=0.0,gvar=gvar):
    """ Rebuild ``g`` stripping correlations with variables not in ``g``.
        
    ``g`` is either an array of |GVar|\s or a dictionary containing |GVar|\s
    and/or arrays of |GVar|\s. ``rebuild(g)`` creates a new collection 
    |GVar|\s with the same layout, means and covariance matrix as those 
    in ``g``, but discarding all correlations with variables not in ``g``. 
        
    If ``corr`` is nonzero, ``rebuild`` will introduce correlations 
    wherever there aren't any using ::
        
        cov[i,j] -> corr * sqrt(cov[i,i]*cov[j,j]) 
        
    wherever ``cov[i,j]==0.0`` initially. Positive values for ``corr`` 
    introduce positive correlations, negative values anti-correlations.
        
    Parameter ``gvar`` specifies a function for creating new |GVar|\s that
    replaces :func:`gvar.gvar` (the default).
            
    :param g: |GVar|\s to be rebuilt.
    :type g: array or dictionary
    :param gvar: Replacement for :func:`gvar.gvar` to use in rebuilding.
        Default is :func:`gvar.gvar`.
    :type gvar: :class:`gvar.GVarFactory` or ``None``
    :param corr: Size of correlations to introduce where none exist
        initially.
    :type corr: number
    :returns: Array or dictionary (gvar.BufferDict) of |GVar|\s  (same layout 
        as ``g``) where all correlations with variables other than those in 
        ``g`` are erased.
    """
    cdef numpy.ndarray[numpy.double_t,ndim=2] gcov
    cdef unsigned int i,j,ng
    cdef float cr
    if hasattr(g,'keys'):
        ## g is a dict ##
        if not isinstance(g,BufferDict):
            g = BufferDict(g)
        buf = rebuild(g.flat,corr=corr,gvar=gvar)
        return BufferDict(g,buf=buf)
        ##
    else:
        ## g is an array ##
        g = numpy.asarray(g)
        if corr!=0.0:
            ng = g.size
            gcov = evalcov(g).reshape(ng,ng)
            cr = corr
            for i in range(ng):
                for j in range(i+1,ng):
                    if gcov[i,j]==0:
                        gcov[i,j] = cr*c_sqrt(gcov[i,i]*gcov[j,j])
                        gcov[j,i] = gcov[i,j]
            return gvar(mean(g),gcov.reshape(2*g.shape))
        else:
            return gvar(mean(g),evalcov(g))
        ##
##
            
def asgvar(x):
    """ Return x if it is type |GVar|; otherwise return 'gvar.gvar(x)`."""
    if isinstance(x,GVar):
        return x
    else:
        return gvar(x)
##
    
##   

## tools for random data: Dataset, avg_data, bin_data ## 
 
def _vec_median(v,spread=False):
    """ estimate the median, with errors, of data in 1-d vector ``v``. 
        
    If ``spread==True``, the error on the median is replaced by the spread
    of the data (which is larger by ``sqrt(len(v))``).
    """
    nv = len(v)
    v = sorted(v)
    if nv%2==0:
        im = int(nv/2)
        di = int(0.341344746*nv)
        median = 0.5*(v[im-1]+v[im])
        sdev = max(v[im+di]-median,median-v[im-di-1])
    else:
        im = int((nv-1)/2)
        di = int(0.341344746*nv+0.5)
        median = v[im]
        sdev = max(v[im+di]-median,median-v[im-di])
    if not spread:
        sdev = sdev/nv**0.5
    return gvar(median,sdev)
##
    
def bin_data(data,binsize=2):
    """ Bin random data.
        
    ``data`` is a list of random numbers or random arrays, or a dictionary of
    lists of random numbers/arrays. ``bin_data(data,binsize)`` replaces
    consecutive groups of ``binsize`` numbers/arrays by the average of those
    numbers/arrays. The result is new data list (or dictionary) with
    ``1/binsize`` times as much random data: for example, ::
        
        >>> print(bin_data([1,2,3,4,5,6,7],binsize=2))
        [1.5, 3.5, 5.5]
        >>> print(bin_data(dict(s=[1,2,3,4,5],v=[[1,2],[3,4],[5,6],[7,8]]),binsize=2))
        {'s': [1.5, 3.5], 'v': [array([ 2.,  3.]), array([ 6.,  7.])]}
        
    Data is dropped at the end if there is insufficient data to from complete
    bins. Binning is used to make calculations
    faster and to reduce measurement-to-measurement correlations, if they
    exist. Over-binning erases useful information.
    """
    if hasattr(data,'keys'):
        ## data is a dictionary ##
        if not data:
            return Dataset()
        newdata = {}
        for k in data:
            newdata[k] = bin_data(data[k],binsize=binsize)
        return newdata
        ##
    ## data is a list ##
    if not data:
        return []
    ## force data into a numpy array of floats ##
    try:
        data = numpy.array(data,float)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in data.")
    ##
    nd = data.shape[0] - data.shape[0]%binsize
    accum = 0.0
    for i in range(binsize):
        accum += data[i:nd:binsize]
    return list(accum/float(binsize))
    ##
##
    
def avg_data(data,median=False,spread=False,bstrap=False):
    """ Average random data to estimate mean.
        
    ``data`` is a list of random numbers or random arrays, or a dictionary of
    lists of random numbers/arrays: for example, ::
        
        >>> random_numbers = [1.60, 0.99, 1.28, 1.30, 0.54, 2.15]
        >>> random_arrays = [[12.2,121.3],[13.4,149.2],[11.7,135.3],
        ...                  [7.2,64.6],[15.2,69.0],[8.3,108.3]]
        >>> random_dict = dict(n=random_numbers,a=random_arrays)
        
    where in each case there are six random numbers/arrays. ``avg_data``
    estimates the means of the distributions from which the random
    numbers/arrays are drawn, together with the uncertainties in those
    estimates. The results are returned as a |GVar| or an array of |GVar|\s,
    or a dictionary of |GVar|\s or arrays of |GVar|\s::
        
        >>> print(avg_data(random_numbers))
        1.31 +- 0.203169
        >>> print(avg_data(random_arrays))
        [11.3333 +- 1.13521 107.95 +- 12.936]
        >>> print(avg_data(random_dict))
        {'n': 1.31 +- 0.203169, 'a': array([11.3333 +- 1.13521, 107.95 +- 12.936], dtype=object)}
        
    The arrays in ``random_arrays`` are one dimensional; in general, they can 
    have any shape.
        
    ``avg_data(data)`` also estimates any correlations between different 
    quantities in ``data``. When ``data`` is a dictionary, it does this by 
    assuming that the lists of random numbers/arrays for the different 
    ``data[k]``\s are synchronized, with the first element in one list 
    corresponding to the first elements in all other lists, and so on. If
    some lists are shorter than others, the longer lists are truncated to 
    the same length as the shortest list (discarding data samples).
        
    There are three optional arguments. If argument ``spread=True`` each
    standard deviation in the results refers to the spread in the data, not
    the uncertainty in the estimate of the mean. The former is ``sqrt(N)``
    larger where ``N`` is the number of random numbers (or arrays) being
    averaged::
        
        >>> print(avg_data(random_numbers,spread=True))
        1.31 +- 0.497661
        
    This is useful, for example, when averaging bootstrap data. The default
    value is ``spread=False``.
            
    The second option is triggered by setting ``median=True``. This replaces
    the means in the results by medians, while the standard deviations are
    determined from the half-width of the interval, centered around the
    median, that contains 68% of the data. These estimates are more robust
    than the mean and standard deviation when averaging over small amounts of
    data; in particular, they are unaffected by extreme outliers in the data.
    The default is ``median=False``.
        
    The third option is triggered by setting ``bstrap=True``. This is
    shorthand for setting ``median=True`` and ``spread=True``, and overrides
    any explicit setting for these keyword arguments. This is the typical
    choice for analyzing bootstrap data --- hence its name. The default value
    is ``bstrap=False``.
    """
    if bstrap:
        median = True
        spread = True
    if hasattr(data,'keys'):
        ## data is a dictionary ##
        if not data:
            return BufferDict()
        newdata = []                    # data repacked as a list of arrays
        i = 0                           # i = measurement number
        lastm = None                    # BufferDict for last measurement
        while i>=0:
            ## iterate over each measurement, building newdata array ##
            m = BufferDict()
            for k in data:
                try:
                    m[k] = data[k][i]
                except IndexError:
                    i = -1              # stops the while loop
                    break
            else:
                lastm = m
                newdata.append(m.buf)
                i += 1
            ##
        if lastm is None:
            return BufferDict()
        else:
            return BufferDict(lastm,
                        buf=avg_data(newdata,median=median,spread=spread))
        ##
    ## data is list ## 
    if not data:
        return None
    ## force data into a numpy array of floats ##
    try:
        data = numpy.array(data,float)
    except ValueError:
        raise ValueError("Inconsistent array shapes or data types in data.")
    ##
    # avg_data.nmeas = len(data)
    if median:
        ## use median and spread ## 
        if len(data.shape)==1:
            return _vec_median(data,spread=spread)
        else:
            tdata = data.transpose()
            tans = numpy.empty(data.shape[1:],object).transpose()
            for ij in numpy.ndindex(tans.shape):
                tans[ij] = _vec_median(tdata[ij],spread=spread)
            ans = tans.transpose()
            cov = numpy.cov(data.reshape(data.shape[0],ans.size),
                            rowvar=False,bias=True)
            if ans.size==1:                 # rescale std devs
                D = sdev(ans)/cov**0.5
            else:
                D = sdev(ans).reshape(ans.size)/numpy.diag(cov)**0.5 
            cov = ((cov*D).transpose()*D).transpose()
            return gvar(mean(ans),cov.reshape(ans.shape+ans.shape))
        ##
    else:
        ## use mean and standard deviation ##
        means = data.mean(axis=0)
        norm = 1.0 if spread else float(len(data))
        if len(data)>=2:
            cov = numpy.cov(data.reshape(data.shape[0],means.size),
                            rowvar=False,bias=True)/norm
        else:
            cov = numpy.zeros(means.shape+means.shape,float)
        if cov.shape==() and means.shape==():
            cov = cov**0.5
        return gvar(means,cov.reshape(means.shape+means.shape))
        ##
    ##
##
    
class Dataset(dict):
    """ Dictionary for collecting random data.
        
    This dictionary class simplifies the collection of random data. The random
    data are stored in a dictionary, with each piece of random data being a
    number or an array of numbers. For example, consider a situation where
    there are four random values for a scalar ``s`` and four random values for
    vector ``v``. These can be collected as follows::
        
        >>> a = gvar.Dataset()
        >>> a.append(s=1.1,v=[12.2,20.6])
        >>> a.append(s=0.8,v=[14.1,19.2])
        >>> a.append(s=0.95,v=[10.3,19.7])
        >>> a.append(s=0.91,v=[8.2,21.0])
        >>> print(a['s'])       # 4 random values of s
        [ 1.1, 0.8, 0.95, 0.91]
        >>> print(a['v'])       # 4 random vector-values of v
        [array([ 12.2,  20.6]), array([ 14.1,  19.2]), array([ 10.3,  19.7]), array([  8.2,  21. ])]
        
    The argument to ``a.append()`` could be a dictionary: for example,
    ``data = dict(s=1.1,v=[12.2,20.6]); a.append(data)`` is equivalent
    to the first ``append`` statement above. This is useful, for 
    example, if the data comes from a function (that returns a dictionary).
        
    One can also append data key by key: for example, ``a.append('s',1.1);
    a.append('v',[12.2,20.6])`` is equivalent to the first ``append`` in the
    example above. One could also achieve this using, for example,
    ``a['s'].append(1.1);a['v'].append([12.2,20.6])``, since each dictionary
    value is a list, but :class:`gvar.Dataset`'s ``append`` checks for
    consistency between the new data and data already collected.
        
    Use ``extend`` in place of ``append`` to collect data in batches: for
    example, ::
        
        >>> a = gvar.Dataset()
        >>> a.extend(s=[1.1,0.8],v=[[12.2,20.6],[14.1,19.2]])
        >>> a.extend(s=[0.95,0.91],v=[[10.3,19.7],[8.2,21.0]])
        >>> print(a['s'])       # 4 random values of s
        [ 1.1, 0.8, 0.95, 0.91]
        
    gives the same dataset as the first example above.
        
    A :class:`gvar.Dataset` can also be created from a file where every 
    line is a new random measurement. The data in the first example above
    could have been stored in a file with the following content::
            
        # file: datafile
        s 1.1
        v [12.2,20.6]
        s 0.8
        v [14.1,19.2]
        s 0.95
        v [10.3,19.7]
        s 0.91
        v [8.2,21.0]
        
    Lines that begin with ``#`` are ignored. Assuming the file is called
    ``datafile``, we create a dataset identical to that above using the code::
        
        >>> a = Dataset('datafile')
        >>> print(a['s'])
        [ 1.1, 0.8, 0.95, 0.91]
        
    Data can be binned while reading it in, which might be useful if there the
    data set is huge. To bin the data contained in file ``datafile`` in bins
    of binsize 2 we use::
        
        >>> a = Dataset('datafile',binsize=2)
        >>> print(a['s'])
        [0.95, 0.93]
        
    Finally the keys read from a data file are restricted to those listed
    in keyword ``keys`` if it is specified: for example, ::
        
        >>> a = Dataset('datafile')
        >>> print([k for k in a])
        ['s', 'v']
        >>> a = Dataset('datafile',keys=['v'])
        >>> print([k for k in a])
        ['v']
        
    """
    def __init__(self,*args,**kargs):
        super(Dataset, self).__init__()
        if not args:
            return
        elif len(args)>1:
            raise TypeError("Expected at most 1 argument, got %d."%len(args))
        if 'nbin' in kargs and 'binsize' not in kargs:
            binsize = kargs.get('nbin',1)   # for legacy code
        else:
            binsize = kargs.get('binsize',1)
        if binsize>1: 
            acc = {}
        keys = set(kargs.get('keys',[]))
        for line in fileinput.input(args[0]):
            f = line.split()
            if len(f)<2 or f[0][0]=='#':
                continue
            k = f[0]
            if keys and k not in keys:
                continue
            if len(f)==2:
                d = eval(f[1])
            elif f[1][0] in "[(":
                d = eval(" ".join(f[1:]),{},{})
            else: # except (NameError,SyntaxError):
                try:
                    d = [float(x) for x in f[1:]]
                except ValueError:
                    raise ValueError('Bad input line: "%s"'%line[:-1])
            if binsize<=1:
                self.append(k,d)
            else:
                acc.setdefault(k,[]).append(d)
                if len(acc[k])==binsize:
                    d = numpy.sum(acc[k],axis=0)/float(binsize)
                    del acc[k]
                    self.append(k,d)
    ##
    def toarray(self):
        """ Create copy of ``self`` where the ``self[k]`` are numpy arrays. """
        ans = dict()
        for k in self:
            ans[k] = numpy.array(self[k],float)
        return ans
    ##
    def append(self,*args,**kargs):
        """ Append data to dataset. 
            
        There are three equivalent ways of adding data to a dataset
        ``dset``: for example, each of ::
            
            dset.append(n=1.739,a=[0.494,2.734])        # method 1
            
            dset.append(n,1.739)                        # method 2
            dset.append(a,[0.494,2.734])
            
            data = dict(n=1.739,a=[0.494,2.734])        # method 3
            dset.append(data)
            
        adds one new random number (or array) to ``dset['n']`` (or
        ``dset['a']``).
        """
        if len(args)>2 or (args and kargs):
            raise ValueError("Too many arguments.")
        if len(args)==2:
            ## append(k,m) ##
            k = args[0]
            d = numpy.asarray(args[1],float)
            if d.shape==():
                d = d.flat[0]
            if k not in self:
                self[k] = [d]
            elif d.shape!=self[k][0].shape:
                raise ValueError(     #
                    "Shape mismatch between measurements %s: %s,%s"%
                    (k,d.shape,self[k][0].shape))
            else:
                self[k].append(d)
            ##
            return
        if len(args)==1:
            ## append(kmdict) ##
            kargs = args[0]
            if not hasattr(kargs,'keys'):
                raise ValueError("Argument not a dictionary.")
            ##
        for k in kargs:
            self.append(k,kargs[k])
    ##
    def extend(self,*args,**kargs):
        """ Add batched data to dataset. 
            
        There are three equivalent ways of adding batched data, containing
        multiple samples for each quantity, to a dataset ``dset``: for
        example, each of ::
            
            dset.extend(n=[1.739,2.682],
                        a=[[0.494,2.734],[ 0.172, 1.400]])  # method 1
            
            dset.extend(n,[1.739,2.682])                    # method 2
            dset.extend(a,[[0.494,2.734],[ 0.172, 1.400]])
            
            data = dict(n=[1.739,2.682],
                        a=[[0.494,2.734],[ 0.172, 1.400]])  # method 3
            dset.extend(data)
            
        adds two new random numbers (or arrays) to ``dset['n']`` (or
        ``dset['a']``).
            
        This method can be used to merge two datasets, whether or not they
        share keys: for example, ::
            
            a = Dataset("file1")
            b = Dataset("file2")
            a.extend(b)                 # a now contains everything in b
        """
        if len(args)>2 or (args and kargs):
            raise ValueError("Too many arguments.")
        if len(args)==2:
            ## extend(k,m) ## 
            k = args[0]
            try:
                d = [numpy.asarray(di,float) for di in args[1]]
            except TypeError:
                raise TypeError('Bad argument.')
            if not d:
                return
            if any(d[0].shape!=di.shape for di in d):
                raise ValueError("Inconsistent shapes.")
            if d[0].shape==():
                d = [di.flat[0] for di in d]
            if k not in self:
                self[k] = d
            elif self[k][0].shape!=d[0].shape:
                raise ValueError( #
                    "Shape mismatch between measurements %s: %s,%s"%
                    (k,d[0].shape,self[k][0].shape))
            else:
                self[k].extend(d)
            ##
            return
        if len(args)==1:
            ## extend(kmdict) ##
            kargs = args[0]
            if not hasattr(kargs,'keys'):
                raise ValueError("Argument not a dictionary.")
            ##
        for k in kargs:
            self.extend(k,kargs[k])
    ##           
    def bootstrap_iter(self,n=None):
        """ Create iterator that returns bootstrap copies of ``self``.
            
        The iterator returns copies of ``self`` whose random numbers/arrays
        are drawn at random (with repetition) from among the samples in
        ``self``. These are fake datasets that should be similar to the
        original (that is, ``self``). Correlations between different
        quantities are preserved. Parameter ``n`` specifies the maximum number
        of copies; there is no maximum if ``n is None``.
        """
        ns = min(len(self[k]) for k in self)  # number of samples
        datadict = self.toarray()
        ct = 0
        while (n is None) or (ct<n):
            ct += 1
            idx = numpy.random.random_integers(0,ns-1,ns)
            ans = Dataset()
            for k in datadict:
                ans[k] = datadict[k][idx]
            yield ans
    ##
## 
    
##

## bootstrap_iter, raniter, ranseed, svd, valder_var ##
    
def bootstrap_iter(g,n=None,svdcut=None,svdnum=None,rescale=True):
    """ Return iterator for bootstrap copies of ``g``. 
        
    :param g: An array (or dictionary) of objects of type |GVar|.
    :type g: array or dictionary or BufferDict
    :param n: Maximum number of random iterations. Setting ``n=None``
        (the default) implies there is no maximum number.
    :param svdcut: If positive, replace eigenvalues of the covariance
        matrix of ``g`` with ``svdcut*(max eigenvalue)``; if negative,
        discards eigenmodes with eigenvalues smaller than 
        ``svdcut*(max eigenvalue)``; ignore if set to ``None``.
    :type svdcut: ``None`` or number
    :param svdnum: If positive, keep only the modes with the largest 
        ``svdnum`` eigenvalues in the covariance matrix for ``g``; 
        ignore if set to ``None`` or negative.
    :type svdnum: ``None`` or positive ``int``
    :param rescale: Covariance matrix is rescaled so that diagonal elements
        equal ``1`` if ``rescale=True``.
    :type rescale: bool
    :returns: An iterator that returns bootstrap copies of ``g``.
    """
    if hasattr(g,'keys'):
        g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    s = svd(evalcov(g.flat),svdcut=svdcut,svdnum=svdnum,rescale=rescale,compute_delta=True)
    g_flat = g.flat if s.delta is None else (g.flat + s.delta)
    wgt = s.decomp(1)
    nwgt = len(wgt)
    count = 0
    while (n is None) or (count<n):
        count += 1
        z = numpy.random.normal(0.0,1.,nwgt)
        buf = g_flat + sum(zi*wi for zi,wi in zip(z,wgt))
        yield BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)
    raise StopIteration
##
    
def raniter(g,n=None,svdcut=None,svdnum=None,rescale=True):
    """ Return iterator for random samples from distribution ``g``
        
    The gaussian deviates (|GVar| objects) in array (or dictionary) ``g`` 
    collectively define a multidimensional gaussian distribution. The 
    iterator defined by :func:`raniter` generates an array (or dictionary)
    containing random numbers drawn from that distribution, with 
    correlations intact. 
        
    The layout for the result is the same as for ``g``. So an array of the
    same shape is returned if ``g`` is an array. When ``g`` is a dictionary, 
    individual entries ``g[k]`` may be |GVar|\s or arrays of |GVar|\s, 
    with arbitrary shapes.
        
    :param g: An array (or dictionary) of objects of type |GVar|.
    :type g: array or dictionary or BufferDict
    :param n: Maximum number of random iterations. Setting ``n=None``
        (the default) implies there is no maximum number.
    :param svdcut: If positive, replace eigenvalues of the covariance
        matrix of ``g`` with ``svdcut*(max eigenvalue)``; if negative,
        discards eigenmodes with eigenvalues smaller than 
        ``svdcut*(max eigenvalue)``; ignore if set to ``None``.
    :type svdcut: ``None`` or number
    :param svdnum: If positive, keep only the modes with the largest 
        ``svdnum`` eigenvalues in the covariance matrix for ``g``; 
        ignore if set to ``None`` or negative.
    :type svdnum: ``None`` or positive ``int``
    :param rescale: Covariance matrix is rescaled so that diagonal elements
        equal ``1`` if ``rescale=True``.
    :type rescale: bool
    :returns: An iterator that returns random arrays or dictionaries
        with the same shape as ``g`` drawn from the gaussian distribution 
        defined by ``g``.
    """
    if hasattr(g,'keys'):
        g = BufferDict(g)
    else:
        g = numpy.asarray(g)
    g_mean = mean(g.flat)
    s = svd(evalcov(g.flat),svdcut=svdcut,svdnum=svdnum,rescale=rescale)
    wgt = s.decomp(1)
    nwgt = len(wgt)
    count = 0
    while count!=n:
        count += 1
        z = numpy.random.normal(0.0,1.,nwgt)
        buf = g_mean + sum(zi*wi for zi,wi in zip(z,wgt))
        yield BufferDict(g,buf=buf) if g.shape is None else buf.reshape(g.shape)
    raise StopIteration
##
    
def ranseed(a):
    """ Seed random number generators with ``a``.
        
    Argument ``a`` is a :class:`tuple` of integers that is used to seed
    the random number generators used by :mod:`numpy` and  
    :mod:`random` (and therefore by :mod:`gvar`). Reusing 
    the same ``a`` results in the same set of random numbers.
        
    :param a: A tuple of integers.
    :type a: tuple
    """
    a = tuple(a)
    numpy.random.seed(a)
##
   
class svd(object):
    """ Compute eigenvalues and eigenvectors of a pos. sym. matrix. 
        
    :class:`svd` is a function-class that computes the eigenvalues and 
    eigenvectors of a positive symmetric matrix ``mat``. Eigenvalues that are
    small (or negative, because of roundoff) can be eliminated or modified 
    using *svd* cuts. Typical usage is::
            
        >>> mat = [[1.,.25],[.25,2.]]
        >>> s = svd(mat)
        >>> print(s.val)             # eigenvalues
        [ 0.94098301  2.05901699]
        >>> print(s.vec[0])          # 1st eigenvector (for s.val[0])
        [ 0.97324899 -0.22975292]
        >>> print(s.vec[1])          # 2nd eigenvector (for s.val[1])
        [ 0.22975292  0.97324899]
            
        >>> s = svd(mat,svdcut=0.6)  # force s.val[i]>=s.val[-1]*0.6
        >>> print(s.val)
        [ 1.2354102   2.05901699]
        >>> print(s.vec[0])          # eigenvector unchanged
        [ 0.97324899 -0.22975292]
        
        >>> s = svd(mat)
        >>> w = s.decomp(-1)         # decomposition of inverse of mat
        >>> invmat = sum(numpy.outer(wj,wj) for wj in w)
        >>> print(numpy.dot(mat,invmat))    # should be unit matrix
        [[  1.00000000e+00   2.77555756e-17]
         [  1.66533454e-16   1.00000000e+00]]
            
    Input parameters are:
        
    :param mat: Positive, symmetric matrix.
    :type mat: 2-d sequence (``numpy.array`` or ``list`` or ...)
    :param svdcut: If positive, replace eigenvalues of ``mat`` with 
        ``svdcut*(max eigenvalue)``; if negative, discard eigenmodes with 
        eigenvalues smaller than ``svdcut`` times the maximum eigenvalue.
    :type svdcut: ``None`` or number ``(|svdcut|<=1)``.
    :param svdnum: If positive, keep only the modes with the largest 
        ``svdnum`` eigenvalues; ignore if set to ``None``.
    :type svdnum: ``None`` or int
    :param compute_delta: Compute ``delta`` (see below) if ``True``; set 
        ``delta=None`` otherwise.
    :type compute_delta: boolean
    :param rescale: Rescale the input matrix to make its diagonal elements 
        equal to 1.0 before diagonalizing.
        
    The results are accessed using:
        
    ..  attribute:: val
        
        An ordered array containing the eigenvalues or ``mat``. Note
        that ``val[i]<=val[i+1]``.
        
    ..  attribute:: vec
        
        Eigenvectors ``vec[i]`` corresponding to the eigenvalues 
        ``val[i]``. 
        
    ..  attribute:: D
        
        The diagonal matrix used to precondition the input matrix if
        ``rescale==True``. The matrix diagonalized is ``D M D`` where ``M`` is
        the input matrix. ``D`` is stored as a one-dimensional vector of
        diagonal elements. ``D`` is ``None`` if ``rescale==False``.
        
    ..  attribute:: kappa 
        
        Ratio of the smallest to the largest eigenvector in the 
        unconditioned matrix.
        
    ..  attribute:: delta
        
        A vector of ``gvar``\s whose means are zero and whose 
        covariance matrix is what was added to ``mat`` to condition 
        its eigenvalues. Is ``None`` if ``svdcut<0`` or 
        ``compute_delta==False``.
    """
    def __init__(self, mat,svdcut=None,svdnum=None,compute_delta=False,rescale=False):
        super(svd,self).__init__()
        self.svdcut = svdcut
        self.svdnum = svdnum
        if rescale:
            mat = numpy.asarray(mat)
            D = (mat.diagonal())**(-0.5)
            DmatD = mat*D
            DmatD = (DmatD.transpose()*D).transpose()
            self.D = D
        else:
            DmatD = mat
            self.D = None
        vec,val,dummy = numpy.linalg.svd(DmatD) 
        vec = numpy.transpose(vec) # now 1st index labels eigenval
        ## guarantee that sorted, with smallest val[i] first ##
        vec = numpy.array(vec[-1::-1])
        val = numpy.array(val[-1::-1])
        self.kappa = val[0]/val[-1] if val[-1]!=0 else None  # min/max eval
        self.delta = None
        ##
        ## svd cuts ##
        if (svdcut is None or svdcut==0.0) and (svdnum is None or svdnum<=0):
            self.val = val
            self.vec = vec
            return
        ## restrict to svdnum largest eigenvalues ##
        if svdnum is not None and svdnum>0:
            val = val[-svdnum:]
            vec = vec[-svdnum:]
        ##
        ## impose svdcut on eignevalues ##
        if svdcut is None or svdcut==0:
            self.val = val
            self.vec = vec
            return 
        valmin = abs(svdcut)*val[-1]
        if svdcut>0:
            ## force all eigenvalues >= valmin ##
            dely = None
            for i in range(len(val)): 
                if val[i]<valmin:
                    if compute_delta:
                        if dely is None:
                            dely = vec[i]*gvar(0.0,(valmin-val[i])**0.5)
                        else:
                            dely += vec[i]*gvar(0.0,(valmin-val[i])**0.5)
                    val[i] = valmin
                else:
                    break
            self.val = val
            self.vec = vec
            self.delta = dely if (self.D is None or dely is None) else dely/self.D
            return 
            ##
        else:
            ## discard modes with eigenvalues < valmin ##
            for i in range(len(val)): 
                if val[i]>=valmin:
                    break
            self.val = val[i:]
            self.vec = vec[i:]
            return  # val[i:],vec[i:],kappa,None
            ##
        ##
        ##
    ##
    def decomp(self,n=1):
        """ Compute vector decomposition of input matrix raised to power ``n``.
            
        Computes vectors ``w[j]`` such that
            
            mat**n = sum_j numpy.outer(w[j],w[j])
                
        where ``mat`` is the original input matrix to :class:`svd`. This 
        decomposition cannot be computed if the input matrix was rescaled
        (``rescale=True``) except for ``n=1`` and ``n=-1``.
            
        :param n: Power of input matrix.
        :type n: number
        :returns: Array ``w`` of vectors.
        """
        if self.D is None:
            w = numpy.array(self.vec)
            for j,valj in enumerate(self.val):
                w[j] *= valj**(n/2.)
        else:
            if n!=1 and n!=-1:
                raise ValueError("Can't compute decomposition for rescaled matrix.")
            w = numpy.array(self.vec)
            Dfac = self.D**(-n)
            for j,valj in enumerate(self.val):
                w[j] *= Dfac*valj**(n/2.)
        return w
    ##
##        
        
def valder_var(vv): 
    """ Convert array ``vv`` of numbers into an array of |GVar|\s.
        
    The |GVar|\s created by ``valder_var(vv)`` have means equal to the
    values ``vv[i]`` and standard deviations of zero. If ``vv`` is
    one-dimensional, for example, ``valder_var(vv)`` is functionally more
    or less the same as::
        
        numpy.array([gvar.gvar(vvi,0.0) for vvi in vv])
        
    In general, the shape of the array returned by ``valder_var`` is the
    same as that of ``vv``.
    """
    try:
        vv = numpy.asarray(vv,float)
    except ValueError:
        raise ValueError("Values aren't numbers.")
    gd = GVarFactory(smat())
    return gd(vv,vv*0.0)
##
    
##
 