""" Introduction 
------------
This module contains tools for nonlinear least-squares curve fitting of data.
In general a fit has four inputs:
        
    1) The dependent data ``y`` that is to be fit --- typically ``y`` 
       is a Python dictionary in an :mod:`lsqfit` analysis. Its values
       ``y[k]`` are either |GVar|\s or arrays (any shape or dimension) of
       |GVar|\s that specify the values of the dependent variables and their
       errors.
       
    2) A collection ``x`` of independent data --- ``x`` can have any structure 
       or it can be ``None``.
       
    3) A fit function ``f(x,p)`` whose parameters ``p`` are adjusted by the fit 
       until ``f(x,p)`` equals ``y`` to within ``y``\s errors --- parameters
       ``p`` are usually specified by a dictionary whose values ``p[k]`` are
       individual parameters or (:mod:`numpy`) arrays of parameters.
       
    4) Initial estimates or *priors* for each parameter in ``p`` --- priors are
       usually specified using a dictionary ``prior`` whose values
       ``prior[k]`` are |GVar|\s or arrays of |GVar|\s that give initial
       estimates (values and errors) for parameters ``p[k]``.
       
A typical code sequence has the structure::
        
    ... collect x,y,prior ...
    
    def f(x,p):
        ... compute fit to y[k], for all k in y, using x,p ...
        ... return dictionary containing the fit values for the y[k]s ...
    
    fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=f)
    print(fit)      # variable fit is of type nonlinear_fit
        
The parameters ``p[k]`` are varied until the ``chi**2`` for the fit is 
minimized.
    
The best-fit values for the parameters are recovered after fitting 
using, for example, ``p=fit.p``. Then the ``p[k]`` are |GVar|\s or arrays 
of |GVar|\s that give best-fit estimates and fit uncertainties in those 
estimates. The ``print(fit)`` statement prints a summary of the fit results.
    
The dependent variable ``y`` above could be an array instead of a dictionary, 
which is less flexible in general but possibly more convenient in simpler fits.
Then the approximate ``y`` returned by fit function ``f(x,p)`` must be an
array with the same shape as the dependent variable. The prior ``prior`` 
could also be represented by an array instead of a dictionary.
    
The :mod:`lsqfit` tutorial contains extended explanations and examples.
"""

# Created by G. Peter Lepage (Cornell University) on 2008-02-12.
# Copyright (c) 2008-2012 G. Peter Lepage. 
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
import warnings
import numpy
import math,copy,pickle,time

import gvar
import lsqfit_util
from lsqfit_util import multifit,multiminex

from lsqfit__version__ import __version__

_FData = collections.namedtuple('_FData',['mean','wgt'])
""" Internal data type for nonlinear_fit.unpack_data() """

class nonlinear_fit(object):
    """ Nonlinear least-squares fit.
        
    :class:`lsqfit.nonlinear_fit` fits a (nonlinear) function ``f(x,p)`` to 
    data ``y`` by varying parameters ``p``, and stores the results: 
    for example, ::
        
        fit = nonlinear_fit(data=(x,y),fcn=f,prior=prior)   # do fit
        print(fit)                                          # print fit results
         
    The best-fit values for the parameters are in ``fit.p``, while the
    ``chi**2``, the number of degrees of freedom, the logarithm of Gaussian
    Bayes Factor, the number of iterations, and the cpu time for the
    fit are in ``fit.chi2``, ``fit.dof``, ``fit.logGBF``, ``fit.nit``, and
    ``fit.time``, respectively. Results for individual parameters in ``fit.p``
    are of type |GVar|, and therefore carry information about errors and
    correlations with other parameters.
        
    :param data: Fit data consisting of ``(x,y)`` where ``x`` is the 
        independent data that is passed to the fit function, and ``y`` is a
        dictionary whose values are |GVar|\s or arrays of |GVar|\s specifying
        the means and covariance matrix for the dependent data (*i.e.*, the
        data being fit). ``y`` could instead be an array of |GVar|\s, rather
        than a dictionary. Another format for ``data`` is the 3-tuples
        ``(x,ymean,ycov)`` (or ``(x,ymean,ysdev)``) where ``ymean`` is an
        array containing the mean ``y`` values, and ``ycov`` is the
        corresponding covariance matrix (or ``ysdev`` the corresponding array
        of standard deviations, if there are no correlations). In this second
        case, ``ycov.shape`` must equal ``ymean.shape+ymean.shape``.
    :type data: 2-tuple or 3-tuple
    :param fcn: Fit function ``fcn(x,p)`` of the independent data ``x`` and
        the parameters ``p``. The function should return approximations to the
        ``y`` data in the same format used for ``y`` in ``data=(x,y)``
        (*i.e.*, a dictionary or array). Fit parameters are stored in ``p``,
        which is either a dictionary, where ``p[k]`` is a single parameter or 
        an array of parameters (any shape), or an array of parameters.
    :type fcn: function
    :param prior: A dictionary (or array) containing *a priori* estimates for 
        all parameters ``p`` used by fit function ``fcn(x,p)``. Fit parameters
        ``p`` are stored in a dictionary (or array) with the same keys and
        structure (or shape) as ``prior``. The default value is ``None``;
        ``prior`` must be defined if ``p0`` is ``None``.
    :type prior: dictionary, array, or ``None``
    :param p0: Starting values for fit parameters in fit. p0 should be a
        dictionary with the same keys and structure as ``prior`` (or an array
        of the same shape if ``prior`` is an array). If ``p0`` is a string, it
        is taken as a file name and :class:`lsqfit.nonlinear_fit` attempts 
        to read starting values from that file; best-fit parameter values are
        written out to the same file after the fit (for priming future fits).
        If ``p0`` is ``None`` or the attempt to read the file fails, starting
        values are extracted from the prior. The default value is ``None``;
        ``p0`` must be defined if ``prior`` is ``None``.
    :type p0: dictionary, array, string or ``None``
    :param svdcut: If positive, eigenvalues of the (rescaled) ``y`` 
        covariance matrix that are smaller than ``svdcut`` times the maximum
        eigenvalue are replaced by ``svdcut`` times the maximum eigenvalue. If
        negative, eigenmodes with eigenvalues smaller than ``|svdcut|`` times
        the largest eigenvalue are discarded. If zero or ``None``, the
        covariance matrix is left unchanged. If ``svdcut`` is a 2-tuple, the
        first entry is ``svdcut`` for the ``y`` covariance matrix and the
        second entry is ``svdcut`` for the prior's covariance matrix.
    :type svdcut: ``None`` or ``float`` or 2-tuple
    :param svdnum: If positive, at most ``svdnum`` eigenmodes of the 
        (rescaled) ``y`` covariance matrix are retained; the modes with the
        smallest eigenvalues are discarded. ``svdnum`` is ignored if set to
        ``None``. If ``svdnum`` is a 2-tuple, the first entry is ``svdnum``
        for the ``y`` covariance matrix and the second entry is ``svdnum`` for
        the prior's covariance matrix.
    :type svdnum: ``None`` or ``int`` or 2-tuple
    :param debug: Set to ``True`` for extra debugging of the fit function and
        a check for roundoff errors. (Default is ``False``.)
    :type debug: boolean
    :param fitterargs: Dictionary of arguments passed on to 
        :class:`lsqfit.multifit`, which does the fitting.
    """
        
    fmt_parameter='%12g +- %8.2g'
    """ Format used for parameters by :func:`lsqfit.nonlinear_fit.format`. """
    fmt_label="%16s"
    """ Format used for parameter labels by :func:`lsqfit.nonlinear_fit.format`."""
    fmt_prior='(%8.2g +- %8.2g)'
    """ Format used for priors by :func:`lsqfit.nonlinear_fit.format`. """
    fmt_table_header='%12s%12s%12s%12s\n'
    """ Format used for table header by :func:`lsqfit.nonlinear_fit.format`. """
    fmt_table_line='%12.5g%12.5g%12.5g%12.5g\n'
    """ Format used for table lines by :func:`lsqfit.nonlinear_fit.format`. """
    alt_fmt_table_line='%11s_%12.5g%12.5g%12.5g\n'
    """ Alt. format used for table lines by :func:`lsqfit.nonlinear_fit.format`. """
        
    def __init__(self,data=None,fcn=None,prior=None,p0=None, #):
                svdcut=None,svdnum=None,debug=False,**kargs): 
        ## capture arguments; initialize parameters ##
        self.fitterargs = kargs
        self.svdcut = svdcut if isinstance(svdcut,tuple) else (svdcut,None)
        self.svdnum = svdnum if isinstance(svdnum,tuple) else (svdnum,None)
        self.data = data
        self.p0file = p0 if isinstance(p0,str) else None
        self.p0 = p0 if self.p0file is None else None
        self.fcn = fcn
        self.prior = prior
        self._p = None
        self._palt = None
        self.debug = debug
        cpu_time = time.clock()
        ##
        ## unpack prior,data,fcn,p0 to reconfigure for multifit ## 
        prior = self._unpack_gvars(self.prior)
        if (debug and prior is not None and
            not all(isinstance(pri,gvar.GVar) for pri in prior.flat)):
            raise TypeError("Priors must be GVars.")
        x,y,prior,fdata = self._unpack_data(data=self.data,prior=prior,
                                        svdcut=self.svdcut,svdnum=self.svdnum)  
        self.x = x 
        self.y = y   
        self.prior = prior  
        self.svdcorrection = (sum(fdata['svdcorrection']) 
                                if len(fdata['svdcorrection'])!=0 else [])
        if 'all' in fdata:
            self.descr = " (input data correlated with prior)" 
        elif 'prior' not in fdata:
            self.descr = " (no prior)"
        else:
            self.descr = ""
        self.p0 = self._unpack_p0(p0=self.p0,p0file=self.p0file,prior=self.prior)
        p0 = self.p0.flatten()  # only need the buffer for multifit 
        fcn = self._unpack_fcn(fcn=self.fcn,p0=self.p0,y=self.y)
        ##
        ## create fit function chiv for multifit ## 
        self._chiv = self._build_chiv(x=x,fdata=fdata,fcn=fcn)
        self._chivw = self._chiv.chivw
        self.dof = self._chiv.nf - self.p0.size
        nf = self._chiv.nf
        ##
        ## trial run if debugging ##
        if self.debug:
            if self.prior is None:
                p0gvar = numpy.array([p0i*gvar.gvar(1,1) 
                                for p0i in p0.flat])
                nchivw = self.y.size
            else:
                p0gvar = self.prior.flatten() + p0
                nchivw = self.y.size + self.prior.size
            for p in [p0,p0gvar]:
                f = fcn(x,p)
                if len(f)!=self.y.size:
                    raise ValueError("fcn(x,p) differs in size from y: %s,%s"
                                        % (len(f),y.size))
                v = self._chiv(p)
                vw = self._chivw(p)
                if nf!=len(v) or nchivw!=len(vw):
                    raise RuntimeError(
                    "Internal error, len(chiv) or len(chivw): (%s,%s) (%s,%s)"
                    %(len(v),nf,len(vw),nchivw))
        ##
        ## do the fit and save results ## 
        fit = lsqfit_util.multifit(p0,nf,self._chiv,**self.fitterargs)
        self.pmean = self._reformat(self.p0,fit.x.flat)
        self.error = fit.error
        self.cov = fit.cov
        self.chi2 = numpy.sum(fit.f**2)
        self.Q = lsqfit_util.gammaQ(self.dof/2.,self.chi2/2.)
        self.nit = fit.nit
        self._p = None          # lazy evaluation
        self._palt = None       # lazy evaluation
        self.psdev = self._reformat(self.p0,[covii**0.5 
                                    for covii in self.cov.diagonal()])
        ## compute logGBF ## 
        if 'logdet_prior' not in fdata: 
            self.logGBF = None
        else:
            logdet_cov = numpy.sum(numpy.log(        #...)
                                numpy.linalg.svd(self.cov,compute_uv=False)))
            self.logGBF = 0.5*(-self.chi2+logdet_cov-fdata['logdet_prior'])
        ##
        ## archive final parameter values if requested ##
        if self.p0file is not None:
            self.dump_parameters(self.p0file)
        ##
        ##
        self.time = time.clock()-cpu_time
        if self.debug:
            self.check_roundoff()
    ##
    def __str__(self): 
        return self.format()
    ##
    @staticmethod
    def _reformat(p,buf):
        """ Transfer format of ``p`` to data in 1-d array ``buf``. """
        assert numpy.ndim(buf)==1,"Buffer ``buf`` must be 1-d."
        if hasattr(p,'keys'):
            ans = gvar.BufferDict(p)
            assert ans.size==len(buf),"p,buf size mismatch: %d,%d"%(ans.size,len(buf))
            ans = BufferDict(ans,buf=buf)
        else:
            assert numpy.size(p)==len(buf),"p,buf size mismatch: %d,%d"%(numpy.size(p),len(buf))
            ans = numpy.array(buf).reshape(numpy.shape(p))
        return ans
    ##
    @staticmethod
    def _unpack_data(data,svdcut,svdnum,prior): 
        """ Unpack data and prior into ``(x,ydict,prior,fdata,logdet_prior)``. 
            
        This routine unpacks ``data`` into ``x,ydict,prior,fdata,logdet_prior`` 
        where ``x`` is whatever was in ``data`` (unprocessed), ``ydict`` is a
        buffer containing the ``y``\s as |GVar|\s, and ``prior`` is the prior.
            
        ``fdata`` is a dictionary containing entries for the ``y``\s (key
        ``"y"``) and the prior (key ``"prior"``), or for both combined (key
        ``"all"``) when the two are correlated. Each entry has a vector
        ``fdata[k].mean`` containing all the corresponding mean values, and an
        array ``fdata[k].wgt`` of vectors whose outer products when summed
        reproduce the corresponding inverse covariance matrix. (When the 
        covariance matrix is diagonal, ``fdata[k].wgt`` is a vector containing
        the square root of the diagonal elements.)
            
        SVD cuts, if specified, are applied before forming the
        ``fdata[k].wgt``\s. In such cases the output ``ydict`` and ``prior``
        are adjusted to reflect the svd cuts.
            
        The logarithm of the determinant of the prior's covariance matrix
        is also returned as ``fdata['logdet_prior']``.
            
        ``data`` is one of: 
            
        .. describe:: ``x,y``
            
            ``y`` is an array of |GVar|\s that is converted into a
            flattened ``numpy.ndarray`` containing the mean values. The
            covariance matrix ``ycov`` is determined using
            :func:`gvar.evalcov`.
                
        .. describe::   ``x,y,ysdev``
            
            ``y`` is an array of numbers that is converted into a flattened
            ``numpy.ndarray``. ``ysdev`` is an array of standard deviations
            that has the same shape as the input ``y`` array. It is
            converted into a diagonal ``numpy.ndarray`` covariance matrix
            ``ycov``.
            
        .. describe:: ``x,y,ycov``
            
            ``y`` is an array of numbers that is converted into a flattened
            ``numpy.ndarray``. ``ycov`` is the covariance matrix for the
            ``y``\s and has shape equal to ``y.shape+y.shape``. It is
            converted into a two-dimensional ``numpy.ndarray`` for internal
            processing.
        """
        if len(data) not in [2,3]:
            raise ValueError("Data tuple wrong length: "+str(len(data)))
        fdata = dict(svdcorrection=[])
        if len(data)==3:
            ## data=x,y,ycov => no correlations with priors ##
            x,ym,ycov = data
            ym = numpy.asarray(ym)
            ycov = numpy.asarray(ycov)
            y = gvar.gvar(ym,ycov)
            if ym.shape==ycov.shape:
                ycov = ycov.flatten()**2
            elif ycov.shape==ym.shape+ym.shape:
                ycov = ycov.reshape((y.size,y.size))
            else:
                raise ValueError("y,ycov shapes mismatched: %s %s"
                                % (y_shape,ycov_shape))
            wgt = decomp_cov(ycov,svdcut=svdcut[0],svdnum=svdnum[0])
            fdata['y'] = _FData(mean=ym.flatten(),wgt=wgt)
            if decomp_cov.svdcorrection is not None:
                fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                y.flat += decomp_cov.svdcorrection
            if prior is not None:
                ## include prior ##
                wgt = decomp_cov(gvar.evalcov(prior.flat),svdcut=svdcut[1],
                                svdnum=svdnum[1])
                fdata['prior'] = _FData(mean=gvar.mean(prior.flat),wgt=wgt)
                fdata['logdet_prior'] = decomp_cov.logdet
                if decomp_cov.svdcorrection is not None:
                    fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                    prior.flat += decomp_cov.svdcorrection
                ##
            ##
        else:
            x,y = data
            y = nonlinear_fit._unpack_gvars(y)
            if prior is not None:
                if gvar.orthogonal(y.flat,prior.flat): 
                    ## data=x,y and y uncorrelated with prior ##
                    wgt = decomp_cov(gvar.evalcov(y.flat),svdcut=svdcut[0],
                                    svdnum=svdnum[0])         
                    fdata['y'] = _FData(mean=gvar.mean(y.flat),wgt=wgt)
                    if decomp_cov.svdcorrection is not None:
                        fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                        y.flat += decomp_cov.svdcorrection
                    wgt = decomp_cov(gvar.evalcov(prior.flat),svdcut=svdcut[1],
                                    svdnum=svdnum[1])
                    fdata['prior'] = _FData(mean=gvar.mean(prior.flat),wgt=wgt)
                    fdata['logdet_prior'] = decomp_cov.logdet
                    if decomp_cov.svdcorrection is not None:
                        fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                        prior.flat += decomp_cov.svdcorrection
                    ##
                else:
                    ## data=x,y and y correlated with prior ##
                    yp = numpy.concatenate((y.flat,prior.flat))
                    wgt = decomp_cov(gvar.evalcov(yp),svdcut=svdcut[0],
                                    svdnum=svdnum[0])
                    fdata['all'] = _FData(mean=gvar.mean(yp),wgt=wgt)
                    if decomp_cov.svdcorrection is not None:
                        fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                        y.flat += decomp_cov.svdcorrection[:y.size]
                        prior.flat += decomp_cov.svdcorrection[y.size:]
                    ## log(det(cov_pr)) where cov_pr = prior part of cov ##
                    invcov = numpy.sum(numpy.outer(wi,wi) for wi in wgt)
                    dummy = decomp_cov(invcov[y.size:,y.size:])
                    fdata['logdet_prior'] = -decomp_cov.logdet  # minus!
                    ##
                    ##
            else:
                ## data=x,y and no prior ##
                wgt = decomp_cov(gvar.evalcov(y.flat),svdcut=svdcut[0],
                                svdnum=svdnum[1])         
                fdata['y'] = _FData(mean=gvar.mean(y.flat),wgt=wgt)
                if decomp_cov.svdcorrection is not None:
                    fdata['svdcorrection'] += decomp_cov.svdcorrection.tolist()
                    y.flat += decomp_cov.svdcorrection
                ##
        return x,y,prior,fdata
    ##        
    @staticmethod
    def _unpack_gvars(g):
        """ Unpack collection of GVars to BufferDict or numpy array. """
        if g is not None:
            g = gvar.gvar(g)
            if not hasattr(g,'flat'):
                g = numpy.asarray(g)
        return g
    ##  
    @staticmethod
    def _unpack_p0(p0,p0file,prior):
        """ Create proper p0. """
        if p0file is not None:
            ## p0 is a filename; read in values ##
            try:
                with open(p0file,"rb") as f:
                    p0 = pickle.load(f)
            except (IOError,EOFError):
                if prior is None:
                    raise IOError("Can't read parameters from "+p0)
                else:
                    p0 = None
            ##
        if p0 is not None:
            ## repackage as BufferDict or numpy array ##
            if hasattr(p0,'keys'):
                p0 = gvar.BufferDict(p0)
            else:
                p0 = numpy.array(p0)
            ##
        if prior is not None:
            ## build new p0 from p0, plus the prior as needed ##
            pp = nonlinear_fit._reformat(prior,buf=[x.mean if x.mean!=0.0 
                            else x.mean+0.1*x.sdev for x in prior.flat])
            if p0 is None:
                p0 = pp
            else:
                if pp.shape is not None:
                    ## pp and p0 are arrays ##
                    pp_shape = pp.shape
                    p0_shape = p0.shape
                    if len(pp_shape)!=len(p0_shape):
                        raise ValueError("p0 and prior shapes incompatible: %s,%s"
                                            % (str(p0_shape),str(pp_shape)))
                    idx = []
                    for npp,np0 in zip(pp_shape,p0_shape):
                        idx.append(slice(0,min(npp,np0)))
                    idx = tuple(idx)    # overlapping slices in each dir
                    pp[idx] = p0[idx]
                    p0 = pp
                    ##
                else:
                    ## pp and p0 are dicts ##
                    if set(pp.keys())!=set(p0.keys()):
                        ## mismatch in keys between prior and p0 ## 
                            raise ValueError("Key mismatch between prior and p0: "
                                + ' '.join(str(k) for k in 
                                            set(prior.keys())^set(p0.keys())))
                        ##   
                    ## adjust p0[k] to be compatible with shape of prior[k] ## 
                    for k in pp:
                        pp_shape = numpy.shape(pp[k])
                        p0_shape = numpy.shape(p0[k])
                        if len(pp_shape)!=len(p0_shape):
                            raise ValueError("p0 and prior incompatible: "+str(k))
                        if pp_shape==p0_shape:
                            pp[k] = p0[k]
                        else:
                            ## find overlap between p0 and pp ##
                            pp_shape = pp[k].shape
                            p0_shape = p0[k].shape
                            if len(pp_shape)!=len(p0_shape):
                                raise ValueError("p0 and prior incompatible: "+str(k))
                            idx = []
                            for npp,np0 in zip(pp_shape,p0_shape):
                                idx.append(slice(0,min(npp,np0)))
                            idx = tuple(idx)    # overlapping slices in each dir
                            ##
                            pp[k][idx] = p0[k][idx]
                    p0 = pp
                    ##
                    ##
            ##
        if p0 is None:
            raise ValueError("No starting values for parameters")
        return p0
    ##
    @staticmethod
    def _unpack_fcn(fcn,p0,y):
        """ reconfigure fitting fcn so inputs,outputs = flat arrays """
        if y.shape is not None:
            if p0.shape is not None:
                def fcn(x,p,fcn=fcn,pshape=p0.shape):
                    po = p.reshape(pshape)
                    ans = fcn(x,po)
                    if hasattr(ans,'flat'):
                        return ans.flat
                    else:
                        return numpy.array(ans).flat
                ##
            else:
                po = BufferDict(p0,buf=numpy.zeros(p0.size,object))
                def fcn(x,p,fcn=fcn,po=po):
                    po.flat = p
                    ans = fcn(x,po)
                    if hasattr(ans,'flat'):
                        return ans.flat
                    else:
                        return numpy.array(ans).flat
                ##
        else:
            yo = BufferDict(y,buf=y.size*[None])
            if p0.shape is not None:
                def fcn(x,p,fcn=fcn,pshape=p0.shape,yo=yo):
                    po = p.reshape(pshape)
                    fxp = fcn(x,po)
                    for k in yo:
                        yo[k] = fxp[k]
                    return yo.flat
                ##
            else:
                po = BufferDict(p0,buf=numpy.zeros(p0.size,object))
                def fcn(x,p,fcn=fcn,po=po,yo=yo):
                    po.flat = p
                    fxp = fcn(x,po)
                    for k in yo:
                        yo[k] = fxp[k]
                    return yo.flat
                ##
        return fcn
    ##
    def _build_chiv(self,x,fdata,fcn):
        """ Build ``chiv`` where ``chi**2=sum(chiv(p)**2)``. """
        if 'all' in fdata:
            ## y and prior correlated ##
            def chiv(p,x=x,fcn=fcn,fd=fdata['all']):
                delta = numpy.concatenate((fcn(x,p),p))-fd.mean
                return (lsqfit_util.dot(fd.wgt,delta) if fd.wgt.ndim==2 
                        else fd.wgt*delta)
            ##
            def chivw(p,x=x,fcn=fcn,fd=fdata['all']):
                delta = numpy.concatenate((fcn(x,p),p))-fd.mean
                if fd.wgt.ndim==2:
                    wgt2 = numpy.sum(numpy.outer(wj,wj) 
                                    for wj in reversed(fd.wgt))
                    return lsqfit_util.dot(wgt2,delta)
                else:
                    return fd.wgt*fd.wgt*delta
            ##
            chiv.nf = len(fdata['all'].wgt)
            ##
        elif 'prior' in fdata:
            ## y and prior uncorrelated ##
            def chiv(p,x=x,fcn=fcn,yfd=fdata['y'],pfd=fdata['prior']):
                ans = []
                for d,w in [(fcn(x,p)-yfd.mean,yfd.wgt),(p-pfd.mean,pfd.wgt)]:
                    ans.append(lsqfit_util.dot(w,d) if w.ndim==2 else w*d)
                return numpy.concatenate(tuple(ans))
            ##
            def chivw(p,x=x,fcn=fcn,yfd=fdata['y'],pfd=fdata['prior']):
                ans = []
                for d,w in [(fcn(x,p)-yfd.mean,yfd.wgt),(p-pfd.mean,pfd.wgt)]:
                    if w.ndim==2:
                        w2 = numpy.sum(numpy.outer(wj,wj) for wj in reversed(w))
                        ans.append(lsqfit_util.dot(w2,d))
                    else:
                        ans.append(w*w*d)
                return numpy.concatenate(tuple(ans))
            ##
            chiv.nf = len(fdata['y'].wgt)+len(fdata['prior'].wgt)
            ##
        else:
            ## no prior ##
            def chiv(p,fcn=fcn,fd=fdata['y']):
                ydelta = fcn(x,p)-fd.mean
                return (lsqfit_util.dot(fd.wgt,ydelta) if fd.wgt.ndim==2 
                        else fd.wgt*ydelta)
            ##
            def chivw(p,fcn=fcn,fd=fdata['y']):
                ydelta = fcn(x,p)-fd.mean
                if fd.wgt.ndim==2:
                    wgt2 = numpy.sum(numpy.outer(wj,wj) 
                            for wj in reversed(fd.wgt))
                    return lsqfit_util.dot(wgt2,ydelta)
                else:
                    return fd.wgt*fd.wgt*ydelta
            ##
            chiv.nf = len(fdata['y'].wgt)
            ##
        chiv.chivw = chivw
        return chiv
    ##
    def check_roundoff(self,rtol=0.25,atol=1e-6):
        """ Check for roundoff errors in fit.p.
            
        Compares standard deviations from fit.p and fit.palt to see if they
        agree to within relative tolerance ``rtol`` and absolute tolerance
        ``atol``. Generates an ``AssertionError`` if they do not (in which
        case an *svd* cut might be advisable).
        """
        psdev = gvar.sdev(self.p.flat)
        paltsdev = gvar.sdev(self.palt.flat)
        if not numpy.allclose(psdev,paltsdev,rtol=rtol,atol=atol):
            warnings.warn("Possible roundoff errors in fit.p; try svd cut.")
    ##
    def _getpalt(self):
        """ Alternate version of ``fit.p``; no correlation with inputs  """
        if self._palt is None:
            self._palt = nonlinear_fit._reformat(self.p0,
                                        gvar.gvar(self.pmean.flat,self.cov))
        return self._palt
    ##
    palt = property(_getpalt,doc="""Best-fit parameters using ``self.cov``.
        Faster than self.p but omits correlations with inputs.""")
    def _getp(self):
        """ Build :class:`gvar.GVar`\s for best-fit parameters. """
        if self._p is not None:
            return self._p
        ## buf = [y,prior]; D[a,i] = dp[a]/dbuf[i] ##
        pmean = self.pmean.flat
        buf = (self.y.flat if self.prior is None else
                numpy.concatenate((self.y.flat,self.prior.flat)))
        D = numpy.zeros((self.cov.shape[0],len(buf)),float)
        for i,chivw_i in enumerate(self._chivw(gvar.valder_var(pmean))):
            for a in range(D.shape[0]):
                D[a,i] = chivw_i.dotder(self.cov[a])
        ##
        ## p[a].mean=pmean[a]; p[a].der[j] = sum_i D[a,i]*buf[i].der[j] ##
        p = []
        for a in range(D.shape[0]): # der[a] = sum_i D[a,i]*buf[i].der
            p.append(gvar.gvar(pmean[a],gvar.wsum_der(D[a],buf),buf[0].cov))
        self._p = nonlinear_fit._reformat(self.p0,p)
        return self._p
        ##
    ##
    p = property(_getp,doc="Best-fit parameters with correlations.")
    fmt_partialsdev = gvar.fmt_errorbudget  # this is for legacy code
    fmt_errorbudget = gvar.fmt_errorbudget
    fmt_values = gvar.fmt_values
    def format(self,*args,**kargs): 
        """ Formats fit output details into a string for printing.
            
        The best-fit values for the fitting function are tabulated
        together with the input data if argument ``maxline>0``. 
            
        The format of the output is controlled by the following format
        strings:
            
            * ``nonlinear_fit.fmt_label`` - parameter labels
            
            * ``nonlinear_fit.fmt_parameter`` - parameters
                
            * ``nonlinear_fit.fmt_prior`` - priors
                
            * ``nonlinear_fit.fmt_table_header`` - header for data vs fit table
            
            * ``nonlinear_fit.fmt_table_line`` - line in data vs fit table 
            
            * ``nonlinear_fit.alt_fmt_table_line`` - alt. line in data vs fit table 
            
            
        :param maxline: Maximum number of data points for which fit 
            results and input data are tabulated. ``maxline<0`` implies
            that only only ``chi2``, ``Q``, ``logGBF``, and ``itns``
            are tabulated; no parameter values are included. Default
            is ``maxline=0``.
        :type maxline: integer
        :returns: String containing detailed information about last fit.
        """
        def bufnames(g):
            if g.shape is None:
                names = g.size*[""]
                for k in g:
                    gk_slice = g.slice(k)
                    if isinstance(gk_slice,slice) and gk_slice.start<gk_slice.stop:
                        names[gk_slice.start] = k
                    else:
                        names[gk_slice] = k
            else:
                names = list(numpy.ndindex(g.shape))
            return names
        ##
        ## unpack information ##
        if (args and kargs) or len(args)>1 or len(kargs)>1:
            raise ValueError("Too many arguments.")
        if args:
            maxline = args[0]
        elif kargs:
            if 'maxline' in kargs:
                maxline = kargs['maxline']
            elif 'nline' in kargs:
                maxline = kargs['nline']
            else:
                raise ValueError("Unknown keyword argument: %s"%list(kargs.keys())[0])
        else:
            maxline = 0     # default
        dof = self.dof
        if dof>0:
            chi2_dof = self.chi2/self.dof
        else:
            chi2_dof = self.chi2
        p = self.pmean.flat
        dp= self.psdev.flat
        fitfcn = self.fcn
        prior = self.prior
        if prior is None:
            prior_p0 = self.p0.flat
            prior_dp = len(prior_p0)*[float('inf')]
        else:
            prior_p0 = gvar.mean(prior.flat)
            prior_dp = gvar.sdev(prior.flat)
        pnames = bufnames(self.p0)
        try:
            Q = '%.2g' % self.Q
        except:
            Q = '?'
        try:
            logGBF = '%.5g' % self.logGBF
        except:
            logGBF = str(self.logGBF)
        ##
        ## create header ##
        table=('Least Square Fit%s:\n  chi2/dof [dof] = %.2g [%d]    Q = %s'
                '    logGBF = %s' % (self.descr,chi2_dof,dof,Q,logGBF))
        table = table+("    itns = %d\n" % self.nit)
        if maxline<0:
            return table
        ##
        ## create parameter table ##
        table = table + '\nParameters:\n'
        for i in range(len(pnames)):
            pnames[i] = str(pnames[i])+'_'
        for i in range(len(p)):
            table = (table + (nonlinear_fit.fmt_label%pnames[i]) 
                    + (nonlinear_fit.fmt_parameter % (p[i],dp[i])))
            p0,dp0 = prior_p0[i],prior_dp[i]
            table = table + '           ' + (nonlinear_fit.fmt_prior%(p0,dp0))
            table = table +'\n'
        if maxline<=0 or self.data is None:
            return table
        ##
        ## create table comparing fit results to data ## 
        data = self.data
        # x,y,ycov,ydict = self._unpack_data(data)
        x = self.x
        ydict = self.y
        y = gvar.mean(ydict.flat)
        dy = gvar.sdev(ydict.flat)
        # dy = ycov**0.5 if len(ycov.shape)==1 else ycov.diagonal()**0.5
        f = fitfcn(x,self.pmean)
        if ydict.shape is None:
            ## convert f from dict to flat numpy array using yo ##
            yo = BufferDict(ydict,buf=ydict.size*[None])
            for k in yo:
                yo[k] = f[k]
            f = yo.flat
            ##
        else:
            f = numpy.asarray(f).flat
            if len(f)==1 and len(f)!=y.size:
                f = numpy.array(y.size*[f[0]])
        ny = len(y)
        stride = 1 if maxline>=ny else (int(ny/maxline) + 1)
        try:    # include x values only if can make sense of them
            x = numpy.asarray(x).flatten()
            assert len(x)==len(y)
            "%f"%x[0]
            tabulate_x = True
            lsqfit_table_line = nonlinear_fit.fmt_table_line
        except:
            tabulate_x = False
            x = bufnames(self.y)
            lsqfit_table_line = nonlinear_fit.alt_fmt_table_line
        table = table + '\nFit:\n'
        header = (nonlinear_fit.fmt_table_header % 
                ('x_i' if tabulate_x else 'key ','y_i','f(x_i)','dy_i'))
        table = table + header + (len(header)-1)*'-' + '\n'
        for i in range(0,ny,stride):
            table = table + (lsqfit_table_line %
                             (x[i],y[i],f[i],dy[i]))
        return table
        ##
    ##
    def dump_parameters(self,filename):
        """ Dump current parameter values into file ``filename``."""
        f = open(filename,"wb")
        if self.p0.shape is not None:
            pickle.dump(numpy.array(self.pmean),f)
        else:
            pickle.dump(dict(self.pmean),f) # dump as a dict
        f.close()
    ##
    def bootstrap_iter(self,n=None,datalist=None):
        """ Iterator that returns bootstrap copies of a fit.
            
        A bootstrap analysis involves three steps: 1) make a large number of
        "bootstrap copies" of the original input data that differ from each
        other by random amounts characteristic of the underlying randomness in
        the original data; 2) repeat the entire fit analysis for each
        bootstrap copy of the data, extracting fit results from each; and 3)
        use the variation of the fit results from bootstrap copy to bootstrap
        copy to determine an approximate probability distribution (possibly
        non-gaussian) for the each result.
            
        Bootstrap copies of the data for step 2 are provided in ``datalist``.
        If ``datalist`` is ``None``, they are generated instead from the 
        means and covariance matrix of the fit data (assuming gaussian 
        statistics). The maximum number of bootstrap copies considered is 
        specified by ``n`` (``None`` implies no limit).
            
        Typical usage is::
            
            ...
            fit = lsqfit.nonlinear_fit(...)
            ...
            for bsfit in fit.bootstrap_iter(n=100,datalist=datalist):
                ... analyze fit parameters in bsfit.pmean ...
            
                        
        :param n: Maximum number of iterations if ``n`` is not ``None``;
                otherwise there is no maximum.
        :type n: integer         
        :param datalist: Collection of bootstrap ``data`` sets for fitter.
        :type datalist: sequence or iterator
        :returns: Iterator that returns an |nonlinear_fit| object containing 
                results from the fit to the next data set in ``datalist``
        """
        fargs = {}
        fargs.update(self.fitterargs)
        fargs['fcn'] = self.fcn
        prior = self.prior
        if datalist is None:
            x = self.x
            y = self.y
            if n is None:
                raise ValueError("datalist,n can't both be None.")
            if prior is None:
                for yb in gvar.bootstrap_iter(y,n):
                    fit = nonlinear_fit(data=(x,yb),prior=None,p0=self.pmean,
                                        **fargs)
                    yield fit
            else:
                g = BufferDict(y=y.flat,prior=prior.flat)
                for gb in gvar.bootstrap_iter(g,n):
                    yb = nonlinear_fit._reformat(y,buf=gb['y'])
                    priorb = nonlinear_fit._reformat(prior,buf=gb['prior'])
                    fit = nonlinear_fit(data=(x,yb),prior=priorb,p0=self.pmean,
                                        **fargs)
                    yield fit
        else:
            if prior is None:
                for datab in datalist:
                    fit = nonlinear_fit(data=datab,prior=None,p0=self.pmean,
                                        **fargs)
                    yield fit
            else:
                piter = gvar.bootstrap_iter(prior)
                for datab in datalist:
                    fit = nonlinear_fit(data=datab,prior=next(piter),
                                        p0=self.pmean,**fargs)
                    yield fit
    ##
##

def empbayes_fit(z0,fitargs,**minargs): 
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
        minargs = dict(tol=1e-3,step=math.log(1.1),maxit=30,analyzer=None)
    save = dict(lastz=None,lastp0=None)
    def minfcn(z,save=save):
        args = fitargs(z)
        if save['lastp0'] is not None:
            args['p0'] = save['lastp0']
        fit = nonlinear_fit(**args)
        if numpy.isnan(fit.logGBF):
            raise ValueError
        else:
            save['lastz'] = z 
            save['lastp0'] = fit.pmean
        return -fit.logGBF
    ##
    try:
        z = lsqfit_util.multiminex(numpy.array(z0),minfcn,**minargs).x
    except ValueError:
        print('*** empbayes_fit warning: null logGBF')
        z = save['lastz']
    args = fitargs(z)
    if save['lastp0'] is not None:
        args['p0'] = save['lastp0']
    return nonlinear_fit(**args),z
##
    
def wavg(xa,svdcut=None,svdnum=None,rescale=True,covfac=None):
    """ Weighted average of 1d-sequence of |GVar|\s or arrays of |GVar|\s.
        
    The weighted average of several |GVar|\s is what one obtains from a 
    least-squares fit of the collection of |GVar|\s to the one-parameter
    fit function ``def f(p): return p[0]``. The average is the best-fit
    value for ``p[0]``. |GVar|\s with smaller standard deviations carry
    more weight than those with larger standard deviations. The averages
    computed by ``wavg`` take account of correlations between the |GVar|\s.
        
    Typical usage is::
        
        x1 = gvar.gvar(...)
        x2 = gvar.gvar(...)
        xavg = wavg([x1,x2])    # weighted average of x1 and x2
        
    In this example, ``x1`` and ``x2`` could be replaced by arrays of 
    |GVar|\s, in which case ``xavg`` is an array as well: for example, ::
        
        x1 = [gvar.gvar(...),gvar.gvar(...)]
        x2 = [gvar.gvar(...),gvar.gvar(...)]
        xavg = wavg([x1,x2])    # xavg[i] is wgtd avg of x1[i] and x2[i]
        
        
    :param xa: The |GVar|\s to be averaged. ``xa`` is a one-dimensional
        sequence of |GVar|\s or of arrays of |GVar|\s, all of the same
        shape.
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
        
        Quality factor `Q` for fit.
        
    """
    xa = numpy.asarray(xa)
    s = None
    svdcorrection = []
    try:
        ## xa is an array of arrays ##
        shape = xa[0].shape
        xaflat = [xai.flat for xai in xa]
        ans = [wavg(xtuple) for xtuple in zip(*xaflat)]
        ans = numpy.array(ans)
        ans.shape = shape
        return ans
        ##
    except AttributeError:
        pass
    cov = gvar.evalcov(xa)
    if covfac is not None:
        cov *= covfac
    ## invert cov ## 
    if numpy.all(numpy.diag(numpy.diag(cov))==cov):
        ## cov is diagonal ## 
        invcov = 1./numpy.diag(cov)
        dof = len(xa)-1
        ans = numpy.dot(invcov,xa)/sum(invcov)
        chi2 = numpy.sum((xa-ans)**2*invcov).mean
        ##
    else:
        ## cov is not diagonal ##
        if (svdcut is None or svdcut==0) and (svdnum is None or svdnum<0):
            invcov = numpy.linalg.inv(cov)
            dof = len(xa)-1
        else:
            ## apply svdcuts; compute conditioned inverse ## 
            s = gvar.svd(cov,svdcut=svdcut,svdnum=svdnum,rescale=rescale,
                         compute_delta=True)
            invcov = numpy.sum(numpy.outer(wj,wj) for wj 
                                    in reversed(s.decomp(-1)))
            dof = len(s.val)-1
            if s.delta is not None:
                svdcorrection = sum(s.delta)
            ##
        ##
        sum_invcov = numpy.sum(invcov,axis=1)
        ans = numpy.dot(sum_invcov,xa)/sum(sum_invcov)
        chi2 = numpy.dot((xa-ans),numpy.dot(invcov,(xa-ans))).mean
    ##
    wavg.chi2 = chi2 
    wavg.dof = dof
    wavg.Q = lsqfit_util.gammaQ(dof/2.,chi2/2.)
    wavg.s = s
    wavg.svdcorrection = svdcorrection
    return ans
##

def decomp_cov(cov,svdcut=None,svdnum=None):
    """ Decompose cov into wgt.
        
    Note that logdet is the log(det(cov)) after it is modified by 
    svd cuts, if any.
    """
    cov_shape = numpy.shape(cov)
    svdcorrection = None
    if len(cov_shape)==1:
        wgt = cov**(-0.5)
        decomp_cov.logdet = numpy.sum(numpy.log(covi) for covi in cov)
    elif numpy.all(cov==numpy.diag(numpy.diag(cov))):
        cov = numpy.diag(cov)
        wgt = cov**(-0.5)
        decomp_cov.logdet = numpy.sum(numpy.log(covi) for covi in cov)
    else:
        try:
            s = gvar.svd(cov,svdcut=svdcut,svdnum=svdnum,
                    compute_delta=True,rescale=True)
            decomp_cov.logdet = -2*numpy.sum(numpy.log(di) for di in s.D)
        except numpy.linalg.LinAlgError:
            s = gvar.svd(cov,svdcut=svdcut,svdnum=svdnum,
                    compute_delta=True,rescale=False)
            decomp_cov.logdet = 0.0
        wgt = s.decomp(-1)[::-1]
        svdcorrection = s.delta
        decomp_cov.logdet += numpy.sum(numpy.log(vali) for vali in s.val)
    decomp_cov.svdcorrection = svdcorrection
    return wgt
##

## legacy definitions (obsolete) ##
BufferDict = gvar.BufferDict
CGPrior = gvar.BufferDict           
GPrior = gvar.BufferDict   
LSQFit = nonlinear_fit
##

if __name__ == '__main__':
    pass
        
        
        
        
