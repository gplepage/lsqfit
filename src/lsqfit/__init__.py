""" Introduction 
------------
This package contains tools for nonlinear least-squares curve fitting of 
data. In general a fit has four inputs:
        
    1) The dependent data ``y`` that is to be fit --- typically ``y`` 
       is a Python dictionary in an :mod:`lsqfit` analysis. Its values
       ``y[k]`` are either |GVar|\s or arrays (any shape or dimension) of
       |GVar|\s that specify the values of the dependent variables 
       and their errors.
       
    2) A collection ``x`` of independent data --- ``x`` can have any 
       structure or it can be ``None``.
       
    3) A fit function ``f(x,p)`` whose parameters ``p`` are adjusted by 
       the fit until ``f(x,p)`` equals ``y`` to within ``y``\s errors 
       --- parameters `p`` are usually specified by a dictionary whose 
       values ``p[k]`` are individual parameters or (:mod:`numpy`) 
       arrays of parameters.
       
    4) Initial estimates or *priors* for each parameter in ``p`` 
       --- priors are usually specified using a dictionary ``prior`` 
       whose values ``prior[k]`` are |GVar|\s or arrays of |GVar|\s that 
       give initial estimates (values and errors) for parameters ``p[k]``.
       
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
using, for example, ``p=fit.p``. Then the ``p[k]`` are |GVar|\s or 
arrays of |GVar|\s that give best-fit estimates and fit uncertainties 
in those estimates. The ``print(fit)`` statement prints a summary of 
the fit results.
    
The dependent variable ``y`` above could be an array instead of a 
dictionary, which is less flexible in general but possibly more 
convenient in simpler fits. Then the approximate ``y`` returned by fit
function ``f(x,p)`` must be an array with the same shape as the dependent
variable. The prior ``prior`` could also be represented by an array 
instead of a dictionary.
    
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

import collections
import warnings
import numpy
import math, pickle, time

import gvar as _gvar

## add extras and utilities to lsqfit ##
from ._extras import empbayes_fit, wavg
from ._utilities import dot as _util_dot
from ._utilities import multifit, multiminex, gammaQ
from .version import version as __version__
##

_FDATA = collections.namedtuple('_FDATA', ['mean','wgt'])
# Internal data type for nonlinear_fit.unpack_data()

class nonlinear_fit(object):
    """ Nonlinear least-squares fit.
        
    :class:`lsqfit.nonlinear_fit` fits a (nonlinear) function ``f(x,p)`` to 
    data ``y`` by varying parameters ``p``, and stores the results: 
    for example, ::
        
        fit = nonlinear_fit(data=(x,y),fcn=f,prior=prior)   # do fit
        print(fit)                               # print fit results
         
    The best-fit values for the parameters are in ``fit.p``, while the
    ``chi**2``, the number of degrees of freedom, the logarithm of Gaussian
    Bayes Factor, the number of iterations, and the cpu time for the fit
    are in ``fit.chi2``, ``fit.dof``, ``fit.logGBF``, ``fit.nit``, and
    ``fit.time``, respectively. Results for individual parameters in
    ``fit.p`` are of type |GVar|, and therefore carry information about
    errors and correlations with other parameters.
        
    :param data: Fit data consisting of ``(x,y)`` where ``x`` is the 
        independent data that is passed to the fit function, and ``y`` is a
        dictionary whose values are |GVar|\s or arrays of |GVar|\s
        specifying the means and covariance matrix for the dependent data
        (*i.e.*, the data being fit). ``y`` could instead be an array of
        |GVar|\s, rather than a dictionary. Another format for ``data`` is
        the 3-tuples ``(x,ymean,ycov)`` (or ``(x,ymean,ysdev)``) where
        ``ymean`` is an array containing the mean ``y`` values, and
        ``ycov`` is the corresponding covariance matrix (or ``ysdev`` the
        corresponding array of standard deviations, if there are no
        correlations). In this second case, ``ycov.shape`` must equal
        ``ymean.shape+ymean.shape``.
    :type data: 2-tuple or 3-tuple
    :param fcn: Fit function ``fcn(x,p)`` of the independent data ``x`` and
        the parameters ``p``. The function should return approximations to
        the ``y`` data in the same format used for ``y`` in ``data=(x,y)``
        (*i.e.*, a dictionary or array). Fit parameters are stored in
        ``p``, which is either a dictionary, where ``p[k]`` is a single
        parameter or an array of parameters (any shape), or an array of
        parameters.
    :type fcn: function
    :param prior: A dictionary (or array) containing *a priori* estimates 
        for all parameters ``p`` used by fit function ``fcn(x,p)``. Fit
        parameters ``p`` are stored in a dictionary (or array) with the
        same keys and structure (or shape) as ``prior``. The default value
        is ``None``; ``prior`` must be defined if ``p0`` is ``None``.
    :type prior: dictionary, array, or ``None``
    :param p0: Starting values for fit parameters in fit. p0 should be a
        dictionary with the same keys and structure as ``prior`` (or an
        array of the same shape if ``prior`` is an array). If ``p0`` is a
        string, it is taken as a file name and
        :class:`lsqfit.nonlinear_fit` attempts to read starting values from
        that file; best-fit parameter values are written out to the same
        file after the fit (for priming future fits). If ``p0`` is ``None``
        or the attempt to read the file fails, starting values are
        extracted from the prior. The default value is ``None``; ``p0``
        must be defined if ``prior`` is ``None``.
    :type p0: dictionary, array, string or ``None``
    :param svdcut: If positive, eigenvalues of the (rescaled) ``y`` 
        covariance matrix that are smaller than ``svdcut`` times the
        maximum eigenvalue are replaced by ``svdcut`` times the maximum
        eigenvalue. If negative, eigenmodes with eigenvalues smaller than
        ``|svdcut|`` times the largest eigenvalue are discarded. If zero or
        ``None``, the covariance matrix is left unchanged. If ``svdcut`` is
        a 2-tuple, the first entry is ``svdcut`` for the ``y`` covariance
        matrix and the second entry is ``svdcut`` for the prior's
        covariance matrix.
    :type svdcut: ``None`` or ``float`` or 2-tuple
    :param svdnum: If positive, at most ``svdnum`` eigenmodes of the 
        (rescaled) ``y`` covariance matrix are retained; the modes with the
        smallest eigenvalues are discarded. ``svdnum`` is ignored if set to
        ``None``. If ``svdnum`` is a 2-tuple, the first entry is ``svdnum``
        for the ``y`` covariance matrix and the second entry is ``svdnum``
        for the prior's covariance matrix.
    :type svdnum: ``None`` or ``int`` or 2-tuple
    :param debug: Set to ``True`` for extra debugging of the fit function
        and a check for roundoff errors. (Default is ``False``.)
    :type debug: boolean
    :param fitterargs: Dictionary of arguments passed on to 
        :class:`lsqfit.multifit`, which does the fitting.
    """
        
    fmt_parameter = '%12g +- %8.2g'
    fmt_label = "%16s"
    fmt_prior = '(%8.2g +- %8.2g)'
    fmt_table_header = '%12s%12s%12s%12s\n'
    fmt_table_line = '%12.5g%12.5g%12.5g%12.5g\n'
    alt_fmt_table_line = '%11s_%12.5g%12.5g%12.5g\n'
        
    def __init__(self, data=None, fcn=None, prior=None, p0=None, #):
                svdcut=None, svdnum=None, debug=False, **kargs): 
        ## capture arguments; initialize parameters ##
        self.fitterargs = kargs
        self.svdcut = svdcut if isinstance(svdcut, tuple) else (svdcut, None)
        self.svdnum = svdnum if isinstance(svdnum, tuple) else (svdnum, None)
        self.data = data
        self.p0file = p0 if isinstance(p0, str) else None
        self.p0 = p0 if self.p0file is None else None
        self.fcn = fcn
        self.prior = prior
        self._p = None
        self._palt = None
        self.debug = debug
        cpu_time = time.clock()
        ##
        ## unpack prior,data,fcn,p0 to reconfigure for multifit ## 
        prior = _unpack_gvars(self.prior)
        if (debug and prior is not None and
            not all(isinstance(pri,_gvar.GVar) for pri in prior.flat)):
            raise TypeError("Priors must be GVars.")
        x, y, prior, fdata = _unpack_data( #
            data=self.data, prior=prior, svdcut=self.svdcut, 
            svdnum=self.svdnum)  
        self.x = x 
        self.y = y   
        self.prior = prior  
        self.svdcorrection = (sum(fdata['svdcorrection']) 
                              if len(fdata['svdcorrection']) != 0 else [])
        if 'all' in fdata:
            self.descr = " (input data correlated with prior)" 
        elif 'prior' not in fdata:
            self.descr = " (no prior)"
        else:
            self.descr = ""
        self.p0 = _unpack_p0(p0=self.p0, p0file=self.p0file, prior=self.prior)
        p0 = self.p0.flatten()  # only need the buffer for multifit 
        flatfcn = _unpack_fcn(fcn=self.fcn, p0=self.p0, y=self.y, x=self.x)
        ##
        ## create fit function chiv for multifit ## 
        self._chiv = _build_chiv(fdata=fdata, fcn=flatfcn)
        self._chivw = self._chiv.chivw
        self.dof = self._chiv.nf - self.p0.size
        nf = self._chiv.nf
        ##
        ## trial run if debugging ##
        if self.debug:
            if self.prior is None:
                p0gvar = numpy.array([p0i*_gvar.gvar(1, 1) 
                                for p0i in p0.flat])
                nchivw = self.y.size
            else:
                p0gvar = self.prior.flatten() + p0
                nchivw = self.y.size + self.prior.size
            for p in [p0, p0gvar]:
                f = flatfcn(p)
                if len(f)!=self.y.size:
                    raise ValueError("fcn(x,p) differs in size from y: %s,%s"
                                     % (len(f), y.size))
                v = self._chiv(p)
                if nf != len(v):
                    raise RuntimeError( #
                        "Internal error -- len(chiv): (%s,%s)" % (len(v), nf))
                vw = self._chivw(p)
                if nchivw != len(vw):
                    raise RuntimeError( #
                        "Internal error -- len(chivw): (%s,%s)"
                        %(len(vw), nchivw))
        ##
        ## do the fit and save results ## 
        fit = multifit(p0, nf, self._chiv, **self.fitterargs)
        self.pmean = _reformat(self.p0, fit.x.flat)
        self.error = fit.error
        self.cov = fit.cov
        self.chi2 = numpy.sum(fit.f**2)
        self.Q = gammaQ(self.dof/2., self.chi2/2.)
        self.nit = fit.nit
        self._p = None          # lazy evaluation
        self._palt = None       # lazy evaluation
        self.psdev = _reformat(self.p0, [covii**0.5 
                               for covii in self.cov.diagonal()])
        ## compute logGBF ## 
        if 'logdet_prior' not in fdata: 
            self.logGBF = None
        else:
            logdet_cov = numpy.sum(numpy.log(        #...)
                                numpy.linalg.svd(self.cov, compute_uv=False)))
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
    def check_roundoff(self, rtol=0.25, atol=1e-6):
        """ Check for roundoff errors in fit.p.
            
        Compares standard deviations from fit.p and fit.palt to see if they
        agree to within relative tolerance ``rtol`` and absolute tolerance
        ``atol``. Generates an ``AssertionError`` if they do not (in which
        case an *svd* cut might be advisable).
        """
        psdev = _gvar.sdev(self.p.flat)
        paltsdev = _gvar.sdev(self.palt.flat)
        if not numpy.allclose(psdev, paltsdev, rtol=rtol, atol=atol):
            warnings.warn("Possible roundoff errors in fit.p; try svd cut.")
    ##
    def _getpalt(self):
        """ Alternate version of ``fit.p``; no correlation with inputs  """
        if self._palt is None:
            self._palt = _reformat(self.p0,
                                   _gvar.gvar(self.pmean.flat, self.cov))
        return self._palt
    ##
    palt = property(_getpalt, doc="""Best-fit parameters using ``self.cov``.
        Faster than self.p but omits correlations with inputs.""")
    def _getp(self):
        """ Build :class:`gvar.GVar`\s for best-fit parameters. """
        if self._p is not None:
            return self._p
        ## buf = [y,prior]; D[a,i] = dp[a]/dbuf[i] ##
        pmean = self.pmean.flat
        buf = (self.y.flat if self.prior is None else
                numpy.concatenate((self.y.flat,self.prior.flat)))
        D = numpy.zeros((self.cov.shape[0], len(buf)), float)
        for i, chivw_i in enumerate(self._chivw(_gvar.valder(pmean))):
            for a in range(D.shape[0]):
                D[a, i] = chivw_i.dotder(self.cov[a])
        ##
        ## p[a].mean=pmean[a]; p[a].der[j] = sum_i D[a,i]*buf[i].der[j] ##
        p = []
        for a in range(D.shape[0]): # der[a] = sum_i D[a,i]*buf[i].der
            p.append(_gvar.gvar(pmean[a], _gvar.wsum_der(D[a], buf), 
                     buf[0].cov))
        self._p = _reformat(self.p0, p)
        return self._p
        ##
    ##
    p = property(_getp, doc="Best-fit parameters with correlations.")
    fmt_partialsdev = _gvar.fmt_errorbudget  # this is for legacy code
    fmt_errorbudget = _gvar.fmt_errorbudget
    fmt_values = _gvar.fmt_values
    def format(self, *args, **kargs): 
        """ Formats fit output details into a string for printing.
            
        The best-fit values for the fitting function are tabulated
        together with the input data if argument ``maxline>0``. 
            
        The format of the output is controlled by the following format
        strings:
            
            * ``nonlinear_fit.fmt_label`` - parameter labels
            
            * ``nonlinear_fit.fmt_parameter`` - parameters
                
            * ``nonlinear_fit.fmt_prior`` - priors
                
            * ``nonlinear_fit.fmt_table_header`` - header for data vs fit 
            
            * ``nonlinear_fit.fmt_table_line`` - line in data vs fit  
            
            * ``nonlinear_fit.alt_fmt_table_line`` - alt line in data vs fit
            
            
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
                    if (isinstance(gk_slice, slice) and 
                        gk_slice.start<gk_slice.stop):
                        names[gk_slice.start] = k
                    else:
                        names[gk_slice] = k
            else:
                if len(g.shape) == 1:
                    names = list(range(len(g)))
                else:
                    names = list(numpy.ndindex(g.shape))
            return names
        ##
        ## unpack arguments, etc ##
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
                raise ValueError("Unknown keyword argument: %s"
                                 % list(kargs.keys())[0])
        else:
            maxline = 0     # default
        dof = self.dof
        if dof > 0:
            chi2_dof = self.chi2/self.dof
        else:
            chi2_dof = self.chi2
        p = self.pmean.flat
        dp = self.psdev.flat
        fitfcn = self.fcn
        prior = self.prior
        if prior is None:
            prior_p0 = self.p0.flat
            prior_dp = len(prior_p0)*[float('inf')]
        else:
            prior_p0 = _gvar.mean(prior.flat)
            prior_dp = _gvar.sdev(prior.flat)
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
        table = ('Least Square Fit%s:\n  chi2/dof [dof] = %.2g [%d]    Q = %s'
                 '    logGBF = %s' % (self.descr,chi2_dof,dof,Q,logGBF))
        table = table+("    itns = %d\n" % self.nit)
        if maxline < 0:
            return table
        ##
        ## create parameter table ##
        table = table + '\nParameters:\n'
        for i in range(len(pnames)):
            pnames[i] = str(pnames[i])+'_'
        for i in range(len(p)):
            table = (table + (nonlinear_fit.fmt_label%pnames[i]) 
                    + (nonlinear_fit.fmt_parameter % (p[i],dp[i])))
            p0, dp0 = prior_p0[i], prior_dp[i]
            table = table + '           ' + (nonlinear_fit.fmt_prior
                                            % (p0, dp0))
            table = table +'\n'
        if maxline <= 0 or self.data is None:
            return table
        ##
        ## create table comparing fit results to data ## 
        x = self.x
        ydict = self.y
        y = _gvar.mean(ydict.flat)
        dy = _gvar.sdev(ydict.flat)
        # dy = ycov**0.5 if len(ycov.shape)==1 else ycov.diagonal()**0.5
        f = fitfcn(self.pmean) if x is False else fitfcn(x, self.pmean)
        if ydict.shape is None:
            ## convert f from dict to flat numpy array using yo ##
            yo = BufferDict(ydict, buf=ydict.size*[None])
            for k in yo:
                yo[k] = f[k]
            f = yo.flat
            ##
        else:
            f = numpy.asarray(f).flat
            if len(f) == 1 and len(f) != y.size:
                f = numpy.array(y.size*[f[0]])
        ny = len(y)
        stride = 1 if maxline >= ny else (int(ny/maxline) + 1)
        try:    # include x values only if can make sense of them
            x = numpy.asarray(x).flatten()
            assert len(x) == len(y)
            "%f" % x[0]
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
        for i in range(0, ny, stride):
            table = table + (lsqfit_table_line %
                             (x[i], y[i], f[i], dy[i]))
        return table
        ##
    ##
    def dump_parameters(self, filename):
        """ Dump current parameter values into file ``filename``."""
        f = open(filename,"wb")
        if self.p0.shape is not None:
            pickle.dump(numpy.array(self.pmean), f)
        else:
            pickle.dump(dict(self.pmean), f) # dump as a dict
        f.close()
    ##
    def bootstrap_iter(self, n=None, datalist=None):
        """ Iterator that returns bootstrap copies of a fit.
            
        A bootstrap analysis involves three steps: 1) make a large number
        of "bootstrap copies" of the original input data that differ from
        each other by random amounts characteristic of the underlying
        randomness in the original data; 2) repeat the entire fit analysis
        for each bootstrap copy of the data, extracting fit results from
        each; and 3) use the variation of the fit results from bootstrap
        copy to bootstrap copy to determine an approximate probability
        distribution (possibly non-gaussian) for the each result.
            
        Bootstrap copies of the data for step 2 are provided in
        ``datalist``. If ``datalist`` is ``None``, they are generated
        instead from the means and covariance matrix of the fit data
        (assuming gaussian statistics). The maximum number of bootstrap
        copies considered is specified by ``n`` (``None`` implies no
        limit).
            
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
        :type datalist: sequence or iterator or ``None``
        :returns: Iterator that returns an |nonlinear_fit| object 
            containing results from the fit to the next data set in
            ``datalist``
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
                for yb in _gvar.bootstrap_iter(y, n):
                    fit = nonlinear_fit(data=(x, yb), prior=None, 
                                        p0=self.pmean, **fargs)
                    yield fit
            else:
                g = BufferDict(y=y.flat, prior=prior.flat)
                for gb in _gvar.bootstrap_iter(g, n):
                    yb = _reformat(y, buf=gb['y'])
                    priorb = _reformat(prior, buf=gb['prior'])
                    fit = nonlinear_fit(data=(x,yb), prior=priorb,
                                        p0=self.pmean, **fargs)
                    yield fit
        else:
            if prior is None:
                for datab in datalist:
                    fit = nonlinear_fit(data=datab, prior=None, p0=self.pmean,
                                        **fargs)
                    yield fit
            else:
                piter = _gvar.bootstrap_iter(prior)
                for datab in datalist:
                    fit = nonlinear_fit(data=datab, prior=next(piter),
                                        p0=self.pmean, **fargs)
                    yield fit
    ##
##

## components of nonlinear_fit ##
def _reformat(p, buf):
    """ Transfer format of ``p`` to data in 1-d array ``buf``. """
    if numpy.ndim(buf) != 1:
        raise ValueError("Buffer ``buf`` must be 1-d.")
    if hasattr(p,'keys'):
        ans = _gvar.BufferDict(p)
        if ans.size != len(buf):
            raise ValueError("p,buf size mismatch: %d,%d"%(ans.size,len(buf)))
        ans = BufferDict(ans, buf=buf)
    else:
        if numpy.size(p) != len(buf):
            raise ValueError(   #
                "p,buf size mismatch: %d,%d"%(numpy.size(p), len(buf)))
        ans = numpy.array(buf).reshape(numpy.shape(p))
    return ans
##
    
def _unpack_data(data, prior, svdcut, svdnum): 
    """ Unpack data and prior into ``(x,y,prior,fdata)``. 
        
    This routine unpacks ``data`` and ``prior`` into ``x,y,prior,fdata``
    where ``x`` is the independent data, ``y`` is the fit data, 
    ``prior`` is the collection of priors for the fit, and ``fdata``
    contains the information about the data and prior needed for the 
    fit function. Both ``y`` and ``prior`` are modified to account
    for *svd* cuts if ``svdcut>0``.
        
    Allowed layouts for ``data`` are: ``x,y,ycov``, ``x,y,ysdev``, 
    ``x,y``, and ``y``. In the last two case, ``y`` can be either an array of 
    |GVar|\s or a dictionary whose values are |GVar|\s or arrays of
    |GVar|\s. In the last case it is assumed that the fit function is a 
    function of only the parameters: ``fcn(p)`` --- no ``x``. (This is
    also assumed if ``x = False``.)
        
    Output data in ``fdata`` includes: fit decompositions of ``y`` 
    (``fdata["y"]``) and ``prior`` (``fdata["prior"]``), or of 
    both together (``fdata["all"]``) if they are correlated. 
    ``fdata["svdcorrection"]`` contains a list of all *svd* corrections
    (from ``y`` and ``prior``). ``fdata["logdet_prior"]`` contains
    the logarithm of the determinant of the prior's covariance matrix.
    """
    ## unpack data tuple ##
    data_is_3tuple = False
    if not isinstance(data,tuple):
        x = False                   # no x in fit fcn
        y = _unpack_gvars(data)
    elif len(data) == 3:
        x, ym, ycov = data
        ym = numpy.asarray(ym)
        ycov = numpy.asarray(ycov)
        y = _gvar.gvar(ym, ycov)
        data_is_3tuple = True
    elif len(data) == 2:
        x, y = data
        y = _unpack_gvars(y)
    else:
        raise ValueError("data tuple wrong length: "+str(len(data)))
    ##
    fdata = dict(svdcorrection=[])
    if prior is not None:
        ## have prior ##
        if data_is_3tuple or _gvar.orthogonal(y.flat, prior.flat):
            ## y uncorrelated with prior ##
            y = _gvar.svd(y, svdcut=svdcut[0], svdnum=svdnum[0],
                          rescale=True, compute_inv=True)
            fdata['y'] = _FDATA(mean=_gvar.mean(y.flat), 
                                wgt=_gvar.svd.inv_wgt)
            if _gvar.svd.correction is not None:
                fdata['svdcorrection'] += \
                    _gvar.svd.correction.tolist()
            prior = _gvar.svd(prior, svdcut=svdcut[0], 
                              svdnum=svdnum[0], rescale=True,
                              compute_inv=True)
            fdata['prior'] = _FDATA(mean=_gvar.mean(prior.flat),
                                    wgt=_gvar.svd.inv_wgt)
            fdata['logdet_prior'] = _gvar.svd.logdet
            if _gvar.svd.correction is not None:
                fdata['svdcorrection'] += \
                    _gvar.svd.correction.tolist()
            ##
        else:
            ## y correlated with prior ##
            yp = _gvar.svd(numpy.concatenate((y.flat, prior.flat)),
                           svdcut=svdcut[0], svdnum=svdnum[0],
                          rescale=True, compute_inv=True)
            fdata['all'] = _FDATA(mean=_gvar.mean(yp), 
                                  wgt=_gvar.svd.inv_wgt)
            if _gvar.svd.correction is not None:
                fdata['svdcorrection'] += \
                    _gvar.svd.correction.tolist()
                y.flat += _gvar.svd.correction[:y.size]
                prior.flat += _gvar.svd.correction[y.size:]
            ## log(det(cov_pr)) where cov_pr = prior part of cov ##
            invcov = numpy.sum(numpy.outer(wi, wi) 
                               for wi in _gvar.svd.inv_wgt)
            s = _gvar.SVD(invcov[y.size:, y.size:])
            fdata['logdet_prior'] = -numpy.sum(numpy.log(vi) # minus!
                                               for vi in s.val)
            ##
            ##
        ##
    else:
        ## no prior ##
        y = _gvar.svd(y, svdcut=svdcut[0], svdnum=svdnum[0],
                      rescale=True, compute_inv=True)
        fdata['y'] = _FDATA(mean=_gvar.mean(y.flat), wgt=_gvar.svd.inv_wgt)
        if _gvar.svd.correction is not None:
            fdata['svdcorrection'] += \
                _gvar.svd.correction.tolist()
        ##
    return x, y, prior, fdata
##        
    
def _unpack_gvars(g):
    """ Unpack collection of GVars to BufferDict or numpy array. """
    if g is not None:
        g = _gvar.gvar(g)
        if not hasattr(g, 'flat'):
            g = numpy.asarray(g)
    return g
##  
    
def _unpack_p0(p0, p0file, prior):
    """ Create proper p0. """
    if p0file is not None:
        ## p0 is a filename; read in values ##
        try:
            with open(p0file, "rb") as f:
                p0 = pickle.load(f)
        except (IOError, EOFError):
            if prior is None:
                raise IOError("Can't read parameters from "+p0)
            else:
                p0 = None
        ##
    if p0 is not None:
        ## repackage as BufferDict or numpy array ##
        if hasattr(p0, 'keys'):
            p0 = _gvar.BufferDict(p0)
        else:
            p0 = numpy.array(p0)
        ##
    if prior is not None:
        ## build new p0 from p0, plus the prior as needed ##
        pp = _reformat(prior, buf=[x.mean if x.mean != 0.0 
                        else x.mean+0.1*x.sdev for x in prior.flat])
        if p0 is None:
            p0 = pp
        else:
            if pp.shape is not None:
                ## pp and p0 are arrays ##
                pp_shape = pp.shape
                p0_shape = p0.shape
                if len(pp_shape)!=len(p0_shape):
                    raise ValueError(
                        "p0 and prior shapes incompatible: %s,%s"
                        % (str(p0_shape), str(pp_shape)))
                idx = []
                for npp, np0 in zip(pp_shape, p0_shape):
                    idx.append(slice(0, min(npp, np0)))
                idx = tuple(idx)    # overlapping slices in each dir
                pp[idx] = p0[idx]
                p0 = pp
                ##
            else:
                ## pp and p0 are dicts ##
                if set(pp.keys()) != set(p0.keys()):
                    ## mismatch in keys between prior and p0 ## 
                    raise ValueError("Key mismatch between prior and p0: "
                                     + ' '.join(str(k) for k in 
                                     set(prior.keys()) ^ set(p0.keys())))
                    ##   
                ## adjust p0[k] to be compatible with shape of prior[k] ## 
                for k in pp:
                    pp_shape = numpy.shape(pp[k])
                    p0_shape = numpy.shape(p0[k])
                    if len(pp_shape)!=len(p0_shape):
                        raise ValueError("p0 and prior incompatible: "
                                         +str(k))
                    if pp_shape == p0_shape:
                        pp[k] = p0[k]
                    else:
                        ## find overlap between p0 and pp ##
                        pp_shape = pp[k].shape
                        p0_shape = p0[k].shape
                        if len(pp_shape)!=len(p0_shape):
                            raise ValueError(
                                "p0 and prior incompatible: "+str(k))
                        idx = []
                        for npp, np0 in zip(pp_shape, p0_shape):
                            idx.append(slice(0, min(npp, np0)))
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
    
def _unpack_fcn(fcn, p0, y, x):
    """ reconfigure fitting fcn so inputs,outputs = flat arrays; hide x """
    if y.shape is not None:
        if p0.shape is not None:
            def nfcn(p, x=x, fcn=fcn, pshape=p0.shape):
                po = p.reshape(pshape)
                ans = fcn(po) if x is False else fcn(x, po)
                if hasattr(ans, 'flat'):
                    return ans.flat
                else:
                    return numpy.array(ans).flat
            ##
        else:
            po = BufferDict(p0, buf=numpy.zeros(p0.size, object))
            def nfcn(p, x=x, fcn=fcn, po=po):
                po.flat = p
                ans = fcn(po) if x is False else fcn(x, po)
                if hasattr(ans, 'flat'):
                    return ans.flat
                else:
                    return numpy.array(ans).flat
            ##
    else:
        yo = BufferDict(y, buf=y.size*[None])
        if p0.shape is not None:
            def nfcn(p, x=x, fcn=fcn, pshape=p0.shape, yo=yo):
                po = p.reshape(pshape)
                fxp = fcn(po) if x is False else fcn(x, po)
                for k in yo:
                    yo[k] = fxp[k]
                return yo.flat
            ##
        else:
            po = BufferDict(p0, buf=numpy.zeros(p0.size, object))
            def nfcn(p, x=x, fcn=fcn, po=po, yo=yo):
                po.flat = p
                fxp = fcn(po) if x is False else fcn(x, po)
                for k in yo:
                    yo[k] = fxp[k]
                return yo.flat
            ##
    return nfcn
##
    
def _build_chiv(fdata, fcn):
    """ Build ``chiv`` where ``chi**2=sum(chiv(p)**2)``. """
    if 'all' in fdata:
        ## y and prior correlated ##
        def chiv(p, fcn=fcn, fd=fdata['all']):
            delta = numpy.concatenate((fcn(p), p))-fd.mean
            return (_util_dot(fd.wgt, delta) if fd.wgt.ndim==2 
                    else fd.wgt*delta)
        ##
        def chivw(p, fcn=fcn, fd=fdata['all']):
            delta = numpy.concatenate((fcn(p), p))-fd.mean
            if fd.wgt.ndim == 2:
                wgt2 = numpy.sum(numpy.outer(wj, wj) 
                                for wj in reversed(fd.wgt))
                return _util_dot(wgt2, delta)
            else:
                return fd.wgt*fd.wgt*delta
        ##
        chiv.nf = len(fdata['all'].wgt)
        ##
    elif 'prior' in fdata:
        ## y and prior uncorrelated ##
        def chiv(p, fcn=fcn, yfd=fdata['y'], pfd=fdata['prior']):
            ans = []
            for d, w in [(fcn(p)-yfd.mean, yfd.wgt),
                        (p-pfd.mean, pfd.wgt)]:
                ans.append(_util_dot(w, d) if w.ndim==2 else w*d)
            return numpy.concatenate(tuple(ans))
        ##
        def chivw(p, fcn=fcn, yfd=fdata['y'], pfd=fdata['prior']):
            ans = []
            for d, w in [(fcn(p)-yfd.mean, yfd.wgt),
                        (p-pfd.mean, pfd.wgt)]:
                if w.ndim == 2:
                    w2 = numpy.sum(numpy.outer(wj, wj) 
                                   for wj in reversed(w))
                    ans.append(_util_dot(w2, d))
                else:
                    ans.append(w*w*d)
            return numpy.concatenate(tuple(ans))
        ##
        chiv.nf = len(fdata['y'].wgt)+len(fdata['prior'].wgt)
        ##
    else:
        ## no prior ##
        def chiv(p, fcn=fcn, fd=fdata['y']):
            ydelta = fcn(p)-fd.mean
            return (_util_dot(fd.wgt, ydelta) if fd.wgt.ndim==2 
                    else fd.wgt*ydelta)
        ##
        def chivw(p, fcn=fcn, fd=fdata['y']):
            ydelta = fcn(p)-fd.mean
            if fd.wgt.ndim == 2:
                wgt2 = numpy.sum(numpy.outer(wj, wj) 
                        for wj in reversed(fd.wgt))
                return _util_dot(wgt2, ydelta)
            else:
                return fd.wgt*fd.wgt*ydelta
        ##
        chiv.nf = len(fdata['y'].wgt)
        ##
    chiv.chivw = chivw
    return chiv
##
##

## legacy definitions (obsolete) ##
BufferDict = _gvar.BufferDict
CGPrior = _gvar.BufferDict           
GPrior = _gvar.BufferDict   
LSQFit = nonlinear_fit
##

       
        
        
