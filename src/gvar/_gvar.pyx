# c#ython: profile=True
# remove extra # above for profiling

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

import re
from ._svec_smat import svec, smat
from ._bufferdict import BufferDict

cimport numpy
cimport cython
from ._svec_smat cimport svec, smat


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
# from re import compile as _compile
import copy

try:
    import powerseries
    _ARRAY_TYPES = [numpy.ndarray,powerseries.PowerSeries]
except:
    _ARRAY_TYPES = [numpy.ndarray]
   

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
    
_GDEVre1 = re.compile(r"([-+]?[0-9]*)[.]([0-9]+)\s*\(([.0-9]+)\)")
_GDEVre2 = re.compile(r"([-+]?[0-9]+)\s*\(([0-9]*)\)")
_GDEVre3 = re.compile(r"([-+]?[0-9.]*[e]*[+-]?[0-9]*)" + "\s*[+][-]\s*"
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
    
# def rebuild(g,corr=0.0,gvar=gvar):
#     """ Rebuild ``g`` stripping correlations with variables not in ``g``.
#         
#     ``g`` is either an array of |GVar|\s or a dictionary containing |GVar|\s
#     and/or arrays of |GVar|\s. ``rebuild(g)`` creates a new collection 
#     |GVar|\s with the same layout, means and covariance matrix as those 
#     in ``g``, but discarding all correlations with variables not in ``g``. 
#         
#     If ``corr`` is nonzero, ``rebuild`` will introduce correlations 
#     wherever there aren't any using ::
#         
#         cov[i,j] -> corr * sqrt(cov[i,i]*cov[j,j]) 
#         
#     wherever ``cov[i,j]==0.0`` initially. Positive values for ``corr`` 
#     introduce positive correlations, negative values anti-correlations.
#         
#     Parameter ``gvar`` specifies a function for creating new |GVar|\s that
#     replaces :func:`gvar.gvar` (the default).
#             
#     :param g: |GVar|\s to be rebuilt.
#     :type g: array or dictionary
#     :param gvar: Replacement for :func:`gvar.gvar` to use in rebuilding.
#         Default is :func:`gvar.gvar`.
#     :type gvar: :class:`gvar.GVarFactory` or ``None``
#     :param corr: Size of correlations to introduce where none exist
#         initially.
#     :type corr: number
#     :returns: Array or dictionary (gvar.BufferDict) of |GVar|\s  (same layout 
#         as ``g``) where all correlations with variables other than those in 
#         ``g`` are erased.
#     """
#     cdef numpy.ndarray[numpy.double_t,ndim=2] gcov
#     cdef unsigned int i,j,ng
#     cdef float cr
#     if hasattr(g,'keys'):
#         ## g is a dict ##
#         if not isinstance(g,BufferDict):
#             g = BufferDict(g)
#         buf = rebuild(g.flat,corr=corr,gvar=gvar)
#         return BufferDict(g,buf=buf)
#         ##
#     else:
#         ## g is an array ##
#         g = numpy.asarray(g)
#         if corr!=0.0:
#             ng = g.size
#             gcov = evalcov(g).reshape(ng,ng)
#             cr = corr
#             for i in range(ng):
#                 for j in range(i+1,ng):
#                     if gcov[i,j]==0:
#                         gcov[i,j] = cr*c_sqrt(gcov[i,i]*gcov[j,j])
#                         gcov[j,i] = gcov[i,j]
#             return gvar(mean(g),gcov.reshape(2*g.shape))
#         else:
#             return gvar(mean(g),evalcov(g))
#         ##
# ##
            
def asgvar(x):
    """ Return x if it is type |GVar|; otherwise return 'gvar.gvar(x)`."""
    if isinstance(x,GVar):
        return x
    else:
        return gvar(x)
##
    
##   

 