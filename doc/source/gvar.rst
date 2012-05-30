:mod:`gvar` - Gaussian Random Variables
==================================================

.. |GVar| replace:: :class:`gvar.GVar`

.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. module:: gvar
   :synopsis: Correlated gaussian random variables.

Introduction 
------------------
This module provides tools for representing and manipulating gaussian
random variables numerically. A gaussian variable is a random variable that
represents a *typical* random number drawn from a particular gaussian (or
normal) probability distribution; more precisely, it represents the entire
probability distribution, and not, for example, a *particular* random number
drawn from that distribution. A given gaussian variable ``x`` is therefore
completely characterized by its mean ``x.mean`` and standard deviation
``x.sdev``.
    
A mathematical function of a gaussian variable can be defined as the
probability distribution of function values obtained by evaluating the
function for random numbers drawn from the original distribution. The
distribution of function values is itself approximately gaussian provided the
standard deviation of the gaussian variable is sufficiently small. Thus we can
define a function ``f`` of a gaussian variable ``x`` to be a gaussian variable
itself, with ::
    
    f(x).mean = f(x.mean)
    f(x).sdev = x.sdev |f'(x.mean)|,
    
which follows from linearizing the ``x`` dependence of ``f(x)`` about point
``x.mean``. (This obviously fails at an extremum of ``f(x)``, where 
``f'(x)=0``.)
    
The last formula, together with its multidimensional generalization, leads
to a full calculus for gaussian random variables that assigns
gaussian-variable values to arbitrary arithmetic expressions and functions
involving gaussian variables. This calculus is useful for analyzing the
propagation of statistical and other random errors (provided the standard
deviations are small enough).
    
A multidimensional collection ``x[i]`` of gaussian variables is
characterized by the means ``x[i].mean`` for each variable, together with a
covariance matrix ``cov[i,j]``. Diagonal elements of ``cov`` specify the
standard deviations of different variables: ``x[i].sdev = cov[i,i]**0.5``.
Nonzero off-diagonal elements imply correlations between different
variables::
    
    cov[i,j] = <x[i]*x[j]>  -  <x[i]> * <x[j]>
    
where, in general, ``<y>`` is the expectation value or mean of random variable
``y``.
    
    
Creating Gaussian Variables
---------------------------
An object of type |GVar| represents a single gaussian variable. Such an
object can be created for a single variable, with mean ``xmean`` and
standard deviation ``xsdev`` (both scalars), using::
    
 	x = gvar.gvar(xmean,xsdev).
    
This function can also be used to convert strings like ``"-72.374(22)"`` or
``"511.2 +- 0.3"`` into |GVar|\s: for example, ::
    
    >>> import gvar
    >>> x = gvar.gvar(3.1415,0.0002)
    >>> print(x)
    3.1415 +- 0.0002
    >>> x = gvar.gvar("3.1415(2)")
    >>> print(x)
    3.1415 +- 0.0002
      
A |GVar| can be converted to a string of this last format using the
:meth:`GVar.fmt` method: for example, ::
    
    >>> print(x.fmt(4))
    3.1415(2)
    >>> print(x.fmt(5))
    3.14150(20)
    
where the argument is the number of decimal places retained. 
    
Function ``gvar.asgvar(x)`` returns x if it is a |GVar|; 
otherwise it returns ``gvar(x)``.
    
|GVar|\s are far more interesting when used to describe multidimensional
distributions, especially if there are correlations between different
variables. Such distributions are represented by collections of |GVar|\s in
one of two standard formats: 1) :mod:`numpy` type arrays of |GVar|\s (any
shape); or, more flexibly, 2) Python dictionaries whose values are |GVar|\s or
arrays of |GVar|\s. Most functions in :mod:`gvar` that handle multiple
|GVar|\s work with either format, and if they return multidimensional results
do so in the same format as the inputs (that is, arrays or dictionaries). Any
dictionary is converted internally into a specialized (ordered) dictionary of
type |BufferDict|, and dictionary-valued results are also |BufferDict|\s.
|BufferDict|\s are also useful for archiving |GVar|\s, since they may be
pickled using Python's :mod:`pickle` module; |GVar|\s cannot be pickled
otherwise. A pickled |BufferDict| preserves all of the correlations between
the different |GVar|\s in it.
    
To create an array of |GVar|\s with mean values specified by array
``xmean`` and covariance matrix ``xcov``, use ::
    
	x = gvar.gvar(xmean,xcov)
    
where array ``x`` has the same shape as ``xmean`` (and ``xcov.shape =
xmean.shape+xmean.shape``). Then each element ``x[i]`` of a one-dimensional
array, for example, is a |GVar| where::
    
    x[i].mean = xmean[i]        # mean of x[i]
    x[i].val  = xmean[i]        # same as x[i].mean
    x[i].sdev = xcov[i,i]**0.5  # std deviation of x[i]
    x[i].var  = xcov[i,i]       # variance of x[i]
    
|GVar|\s can be used in arithmetic expressions, just like Python
floats. These expressions result in new |GVar|\s whose means and standard
deviations are determined from the original covariance matrix. The
arithmetic expressions can include calls to standard functions including:
``exp, log, sqrt, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh,
arcsinh, arccosh, arctanh``.
    
As an example, ::
    
    >>> x,y = gvar.gvar([0.1,10.],[[0.015625,0.],[0.,4.]])
    >>> print('x =', x, '   y =', y)
    x = 0.1 +- 0.125    y = 10 +- 2
    
makes ``x`` and ``y`` |GVar|\s with standard deviations ``sigma_x=0.125`` and
``sigma_y=2``, and, in this case, no correlation between ``x`` and ``y``
(since ``cov[i,j]=0`` when ``i!=j``). If now we set, for example, ::
    
    >>> f = x+y
    >>> print('f =',f)
    f = 10.1 +- 2.0039
    
then ``f`` is a |GVar| with ::
    
    f.var = df/dx cov[0,0] df/dx + df/dx cov[0,1] df/dy + ... 
          = 2.0039**2
    
where ``cov`` is the original covariance matrix used to define ``x`` and
``y`` (in ``gvar.gvar``). Note that while ``f`` and ``y`` separately have
20% uncertainties in this example, the ratio ``f/y`` has much smaller
errors::
    
    >>> print(f/y)
    1.01 +- 0.012659
    
This happens, of course, because the errors in ``f`` and ``y`` are highly
correlated (since the error in ``f`` comes mostly from ``y``).
    
It is sometimes useful to know how much of the uncertainty in some quantity
is due to a particular input uncertainty. Continuing the example above, for
example, we might want to know how much of ``f``\s standard deviation
is due to the standard deviation of ``x`` and how much comes from ``y``. 
This is easily computed (for the example above)::
    
    >>> print(f.partialsdev(x))        # uncertainty in f due to x
    0.125
    >>> print(f.partialsdev(y))        # uncertainty in f due to y
    2.0
    >>> print(f.partialsdev(x,y))      # uncertainty in f due to x and y
    2.00390244274
    >>> print(f.sdev)                  # should be the same
    2.00390244274
    
:func:`gvar.gvar` can also be used to convert strings or tuples stored in
arrays or dictionaries into |GVar|\s: for example, ::
    
    >>> garray = gvar.gvar(["2(1)","10+-5",(99,3),gvar.gvar(0,2)])
    >>> print(garray)
    [2 +- 1 10 +- 5 99 +- 3 0 +- 2]
    >>> gdict = gvar.gvar(dict(a="2(1)",b=["10+-5",(99,3),gvar.gvar(0,2)]))
    >>> print(gdict)
    {'a': 2 +- 1, 'b': array([10 +- 5, 99 +- 3, 0 +- 2], dtype=object)}
    
If the covariance matrix in ``gvar.gvar`` is diagonal, it can be replaced
by an array of standard deviations (square roots of diagonal entries in
``cov``). The example above, therefore, is equivalent to::
    
    >>> x,y = gvar.gvar([0.1,10.],[0.125,2.])
    >>> print('x =', x, '   y =', y)
    x = 0.1 +- 0.125    y = 10 +- 2
    
    
Computing Covariance Matrices
-----------------------------   
The covariance matrix for a set of |GVar|\s, ``g0 g1`` ...,
can be computed using ::
    
    gvar.evalcov([g0,g1,..]) -> cov_g
    
where ``cov_g[i,j]`` gives the covariance between ``gi`` and ``gj``.
Instead of a list or array of ``g``\s, one can also give a dictionary ``g``
where ``g[k]`` is a |GVar|. In this case :func:`gvar.evalcov` returns a
doubly-indexed dictionary ``cov_g[k1][k2]`` where keys ``k1,k2`` are 
in ``g``.
    
Using the example from the previous section, the code
    
    >>> x,y = gvar.gvar([0.1,10.],[[0.015625,0.],[0.,4.]])
    >>> f = x+y
    >>> print(gvar.evalcov([x,y,f]))
    [[ 0.015625  0.        0.015625]
     [ 0.        4.        4.      ]
     [ 0.015625  4.        4.015625]]
    
confirms that ``x`` and ``y`` are uncorrelated with each other, but strongly
correlated with ``f``.
    
It is often convenient to group related |GVar|\s together in a dictionary
rather than an array since dictionaries are far more flexible. ``gvar.evalcov`` 
can be used to evaluate the covariance matrix for a dictionary containing
|GVar|\s and/or arbitrary arrays of |GVar|\s::
    
    >>> d = dict(x=x,y=y,g=[x+y,x-y])
    >>> cov = gvar.evalcov(d)
    >>> print(cov['x','x'])
    0.015625
    >>> print(cov['x','y'])
    0.0
    >>> print(cov['x','g'])
    [ 0.015625  0.015625]
    
    
.. _gvar-random-number-generators:
    
Random Number Generators
------------------------
|GVar|\s represent probability distributions. It is possible to use them
to generate random numbers from those distributions. For example, in
    
    >>> z = gvar.gvar(2.0,0.5)
    >>> print(z())
    2.29895701465
    >>> print(z())
    3.00633184275
    >>> print(z())
    1.92649199321
    
calls to ``z()`` generate random numbers from a gaussian random number 
generator with mean ``z.mean=2.0`` and standard deviation ``z.sdev=0.5``.
    
To obtain random arrays from an array ``g`` of |GVar|\s
use ``giter=gvar.raniter(g)`` (see :func:`gvar.raniter`) to create a
random array generator ``giter``. Each call to ``next(giter)`` generates 
a new array of random numbers. The random number arrays have the same 
shape as the array ``g`` of |GVar|\s and have the distribution implied 
by those random variables (including correlations). For example,
    
    >>> a = gvar.gvar(1.0,1.0)
    >>> da = gvar.gvar(0.0,0.1)
    >>> g = [a,a+da]
    >>> giter = gvar.raniter(f)
    >>> print(next(giter))
    [ 1.51874589  1.59987422]
    >>> print(next(giter))
    [-1.39755111 -1.24780937]
    >>> print(next(giter))
    [ 0.49840244  0.50643312]
    
Note how the two random numbers separately vary over the region ``1+-1``
(approximately), but the separation between the two is rarely more than
``0+-0.1``. This is as expected given the strong correlation between ``a``
and ``a+da``.
    
``gvar.raniter(g)`` also works when ``g`` is a dictionary (or
:class:`gvar.BufferDict`) whose entries ``g[k]`` are |GVar|\s or arrays of
|GVar|\s. In such cases the iterator returns a dictionary with the same
layout::
    
    >>> g = dict(a=gvar.gvar(0,1),b=[gvar.gvar(0,100),gvar.gvar(10,1e-3)])
    >>> print(g)
    {'a': 0 +- 1, 'b': [0 +- 100, 10 +- 0.001]}
    >>> giter = gvar.raniter(g)
    >>> print(next(giter))
    {'a': -0.88986130981173306, 'b': array([-67.02994213,   9.99973707])}
    >>> print(next(giter))
    {'a': 0.21289976681277872, 'b': array([ 29.9351328 ,  10.00008606])}
    
One use for such random number generators is dealing with situations where
the standard deviations are too large to justify the linearization 
assumed in defining functions of gaussian variables. Consider, for example,
    
    >>> x = gvar.gvar(1.,3.)
    >>> print(cos(x))
    0.540302 +- 2.52441
    
The standard deviation for ``cos(x)`` is obviously wrong since ``cos(x)``
can never be larger than one. To obtain the real mean and standard deviation,
we generate a large number of random numbers ``xi`` from ``x``, compute 
``cos(xi)`` for each, and compute the mean and standard deviation for the
resulting distribution (or any other statistical quantity, particularly if
the resulting distribution is not gaussian)::
    
    # estimate mean,sdev from 1000 random x's
    >>> ran_x = numpy.array([x() for in range(1000)]) 
    >>> ran_cos = numpy.cos(ran_x)
    >>> print('mean =',ran_cos.mean(),'  std dev =',ran_cos.std())
    mean = 0.0350548954142   std dev = 0.718647118869
    
    # check by doing more (and different) random numbers
    >>> ran_x = numpy.array([x() for in range(100000)])
    >>> ran_cos = numpy.cos(ran_x)
    >>> print('mean =',ran_cos.mean(),'  std dev =',ran_cos.std())
    mean = 0.00806276057656   std dev = 0.706357174056
    
This procedure generalizes trivially for multidimensional analyses, using 
arrays or dictionaries with :func:`gvar.raniter`.
    
Finally note that *bootstrap* copies of |GVar|\s are easily created. A
bootstrap copy of |GVar| ``x +- dx`` is another |GVar| with the same width but
where the mean value is replaced by a random number drawn from the original
distribution. Bootstrap copies of a data set, described by a collection of
|GVar|\s, can be used as new (fake) data sets having the same statistical
errors and correlations::
    
    >>> g = gvar.gvar([1.1,0.8],[[0.01,0.005],[0.005,0.01]])
    >>> print(g)
    [1.1 +- 0.1 0.8 +- 0.1]
    >>> print(gvar.evalcov(g))                  # print covariance matrix
    [[ 0.01   0.005]
     [ 0.005  0.01 ]]
    >>> gbs_iter = gvar.bootstrap_iter(g)
    >>> gbs = next(gbs_iter)                    # bootstrap copy of f
    >>> print(gbs)
    [1.13881 +- 0.1 0.896066 +- 0.1]            # different means
    >>> print(gvar.evalcov(gbs))
    [[ 0.01   0.005]                            # same covariance matrix
     [ 0.005  0.01 ]]
    
Such fake data sets are useful for analyzing non-gaussian behavior, for
example, in nonlinear fits.
    
    
Analyzing Random Samples
------------------------
:mod:`gvar` contains a several tools for collecting and analyzing random 
samples from arbitrary distributions. The random samples are represented 
by lists of numbers or arrays, where each number/array is a new sample from
the underlying distribution. For example, six samples from a one-dimensional
gaussian distribution (``1+-1``) might look like ::
    
    >>> random_numbers = [1.739, 2.682, 2.493, -0.460, 0.603, 0.800]
    
while six samples from a two-dimensional distribution (``[1+-1,2+-1]``)
might be ::
    
    >>> random_arrays = [[ 0.494, 2.734], [ 0.172, 1.400], [ 1.571, 1.304], 
    ...                  [ 1.532, 1.510], [ 0.669, 0.873], [ 1.242, 2.188]]
    
Samples from more complicated multidimensional distributions are represented
by dictionaries whose values are lists of numbers or arrays.
    
With large samples, we typically want to estimate the mean value of the 
underlying distribution. This is done using :func:`gvar.avg_data`:
for example, ::
    
    >>> print(gvar.avg_data(random_numbers))
    1.3095 +- 0.452117
    
indicates that ``1.31(45)`` is our best guess, based only upon the samples in
``random_numbers``, for the mean of the distribution from which those samples
were drawn. Similarly ::
    
    >>> print(gvar.avg_data(random_arrays))
    [0.946667 +- 0.217418 1.66817 +- 0.251002]  
      
indicates that the means for the two-dimensional distribution behind
``random_arrays`` are ``[0.95(22), 1.67(25)]``. :func:`avg_data` can also
be applied to a dictionary whose values are lists of numbers/arrays: for
example, ::
    
    >>> data = dict(n=random_numbers,a=random_arrays)
    >>> print(gvar.avg_data(data))
    {'a': array([0.946667 +- 0.217418, 1.66817 +- 0.251002], dtype=object), 
     'n': 1.3095 +- 0.452117}
    
Class :class:`gvar.Dataset` can be used to assemble dictionaries containing
random samples. For example, imagine that the random samples above were
originally written into a file, as they were generated::
    
    # file: datafile
    n 1.739
    a [ 0.494, 2.734]
    n 2.682
    a [ 0.172, 1.400]
    n 2.493
    a [ 1.571, 1.304]
    n -0.460
    a [ 1.532, 1.510]
    n 0.603
    a [ 0.669, 0.873]
    n 0.800
    a [ 1.242, 2.188]
    
Here each line is a different random sample, either from the one-dimensional
distribution (labeled ``n``) or from the two-dimensional distribution (labeled
``a``). Assuming the file is called ``datafile``, this data can be read into
a dictionary, essentially identical to the ``data`` dictionary above, using::
    
    >>> data = gvar.Dataset("datafile")
    >>> print(data['a'])
    [array([ 0.494, 2.734]), array([ 0.172, 1.400]), array([ 1.571, 1.304]) ... ]
    >>> print(avg_data(data['n']))
    1.3095 +- 0.452117
    
The brackets and commas can be omitted in the input file for one-dimensional
arrays: for example, ``datafile`` (above) could equivalently be written ::
    
    # file: datafile
    n 1.739
    a 0.494 2.734
    n 2.682
    a 0.172 1.400
    ...
   
Other data formats may also be easy to use. For example, a data file written 
using ``yaml`` would look like ::
    
    # file: datafile
    ---
    n: 1.739
    a: [ 0.494, 2.734]
    ---
    n: 2.682
    a: [ 0.172, 1.400]
    .
    .
    .
    
and could be read into a :class:`gvar.Dataset` using::
    
    import yaml
    
    data = gvar.Dataset()
    with open("datafile","r") as dfile:
        for d in yaml.load_all(dfile.read()):   # iterate over yaml records  
            data.append(d)                      # d is a dictionary
    
Finally note that data can be binned, into bins of size ``binsize``, using
:func:`gvar.bin_data`. For example, ``gvar.bin_data(data,binsize=3)`` replaces
every three samples in ``data`` by the average of those samples. This creates
a dataset that is ``1/3`` the size of the original but has the same mean.
Binning is useful for making large datasets more manageable, and also for
removing sample-to-sample correlations. Over-binning, however, erases
statistical information.
    
Class :class:`gvar.Dataset` can also be used to build a dataset sample by
sample in code: for example, ::
    
    >>> a = Dataset()
    >>> a.append(n=1.739,a=[ 0.494, 2.734])
    >>> a.append(n=2.682,a=[ 0.172, 1.400])
    ...
    
creates the same dataset as above.
    
    
Limitations
-----------
The most fundamental limitation of this module is that the calculus of
gaussian variables that it assumes is only valid when standard deviations
are small (compared to the distances over which the functions of interest
change appreciably). One way of dealing with this limitation is described
above in the section on :ref:`gvar-random-number-generators`.
    
Another potential issue is roundoff error, which can become problematic if
there is a wide range of standard deviations among correlated modes. For
example, the following code works as expected::
    
    >>> from gvar import gvar,evalcov
    >>> tiny = 1e-4
    >>> a = gvar(0.,1.)
    >>> da = gvar(tiny,tiny)
    >>> a,ada = gvar([a.mean,(a+da).mean],evalcov([a,a+da])) # = a,a+da
    >>> print(ada-a)   # should be da again
    0.0001 +- 0.0001
    
Reducing ``tiny``, however, leads to problems::
    
    >>> from gvar import gvar,evalcov
    >>> tiny = 1e-8
    >>> a = gvar(0.,1.)
    >>> da = gvar(tiny,tiny)
    >>> a,ada = gvar([a.mean,(a+da).mean],evalcov([a,a+da])) # = a,a+da
    >>> print(ada-a)   # should be da again
    1e-8 +- 0
    
Here the call to :func:`gvar.evalcov` creates a new covariance matrix for
``a`` and ``ada = a+da``, but the matrix does not have enough numerical
precision to encode the size of ``da``'s variance, which gets set, in
effect, to zero. The problem arises here for values of ``tiny`` less than
about ``2e-8`` (with 64-bit floating point numbers --- ``tiny**2`` is what
appears in the covariance matrix).
    
    
Implementation Notes; Optimizations 
-------------------------------------
There are two types of |GVar|: the underlying independent variables, created
with calls to :func:`gvar.gvar`; and variables which are obtained from
functions of the underlying variables. Each |GVar| must keep track of three
pieces of information: 1) its mean value; 2) its derivatives with respect to
the underlying variables; and 3) the covariance matrix for the underlying
variables. The derivatives and covariance matrix allow one to compute the
standard deviation of the |GVar| as well as correlations between it and any
other function of the underlying variables. A |GVar| can be constructed at a
very low level by supplying all three pieces of information --- for example, ::
    
	f = gvar.gvar(fmean,fder,cov)
    
where ``fmean`` is the mean, ``fder`` is an array where ``fder[i]`` is the
derivative of ``f`` with respect to the ``i``-th underlying variable
(numbered in the order in which they were created using :func:`gvar.gvar`),
and ``cov`` is the covariance matrix for the underlying variables (easily
obtained from an existing |GVar| ``x`` using ``x.cov``).
    
When there are lots of underlying variables, the number of derivatives can
become rather large, potentially (though not necessarily) leading to slower
calculations. One way to alleviate this problem, should it arise, is to
separate the underlying variables into groups that are never mixed in
calculations and to use different :func:`gvar.gvar`\s when generating the
variables in different groups. New versions of :func:`gvar.gvar` are
obtained using :func:`gvar.switch_gvar`: for example, ::
    
    import gvar
    ...
    x = gvar.gvar(...)
    y = gvar.gvar(...)
    z = f(x,y)
    ... other manipulations involving x and y ...
    gvar.switch_gvar()
    a = gvar(...)
    b = gvar(...)
    c = g(a,b)
    ... other manipulations involving a and b (but not x and y) ...
    
Here the :func:`gvar.gvar` used to create ``a`` and ``b`` is a different
function than the one used to create ``x`` and ``y``. A derived quantity,
like ``c``, knows about its derivatives with respect to ``a`` and ``b``,
and about their covariance matrix; but it carries no derivative information
about ``x`` and ``y``. Absent the ``switch_gvar`` line, ``c`` would have
information about its derivatives with respect to ``x`` and ``y`` (zero
derivative in both cases) and this would make calculations involving ``c``
slightly slower than with the ``switch_gvar`` line. Usually the difference
is negligible, but in cases with hundreds or thousands of underlying
variables redefining ``gvar`` can make a substantial difference. Note that
the previous :func:`gvar.gvar` can be restored using
:func:`gvar.restore_gvar`.
    
|GVar|\s are designed to work well with :mod:`numpy` arrays. They
can be combined in arithmetic expressions with arrays of numbers or of
|GVar|\s; the results in both cases are arrays of
|GVar|\s.
        
Arithmetic operators ``+ - * / ** == != <> += -= *= /=`` are all
defined. |GVar|\s are not ordered so ``> >= < <=`` are not defined.


Utilities
----------
The function used to create gaussian variable objects is:

.. autofunction:: gvar.gvar(...)

Means, standard deviations, variances, and covariance matrices can 
be extracted from arrays (or dictionaries) of |GVar|\s using:

.. autofunction:: gvar.mean(g)

.. autofunction:: gvar.sdev(g)

.. autofunction:: gvar.var(g)

.. autofunction:: gvar.evalcov(g)


The following function creates an iterator that generates random arrays
from the distribution defined by array (or dictionary) ``g`` of |GVar|\s. 
The random numbers incorporate any correlations implied by the ``g``\s.

.. autofunction:: gvar.raniter(g,n=None,svdcut=None,svdnum=None,rescale=True)

.. autofunction:: gvar.bootstrap_iter(g,n=None,svdcut=None,svdnum=None,rescale=True)

.. autofunction:: gvar.ranseed(a)

Two functions that are useful for tabulating results and for analyzing where
the errors in a |GVar| constructed from other |GVar|\s come from:

.. autofunction:: gvar.fmt_errorbudget(outputs,inputs,ndigit=2,percent=True)

.. autofunction:: gvar.fmt_values(outputs,ndigit=3)

The following functions creates new functions that generate |GVar|\s (to 
replace :func:`gvar.gvar`):

.. autofunction:: gvar.switch_gvar()

.. autofunction:: gvar.restore_gvar()

.. autofunction:: gvar.gvar_factory(cov=None)

|GVar|\s created by different functions cannot be combined in arithmetic
expressions (the error message "Incompatible GVars." results). 

Two functions are used to analyze sets of random samples from gaussian 
distributions:

.. autofunction:: gvar.avg_data(data,median=False,spread=False,bstrap=False)

.. autofunction:: gvar.bin_data(data,binsize=2)


The following function can be used to rebuild collections of |GVar|\s, 
ignoring all correlations with other variables. It can also be used to 
introduce correlations between uncorrelated variables.

.. autofunction:: gvar.rebuild(g,gvar=gvar,corr=0.0)

Finally there is a utility function-class for implementing an *svd* analysis
of a covariance or other symmetric, positive matrix:

.. autoclass:: gvar.svd(mat,svdcut=None,svdnum=None,compute_delta=False,rescale=False)
   
   .. automethod:: decomp(n=1) 

Classes
-------
The fundamental class for representing gaussian variables is:

.. autoclass:: gvar.GVar
   
   The basic attributes are:
   
   .. autoattribute:: mean
   
   .. autoattribute:: sdev
   
   .. autoattribute:: var
   
   Two methods allow one to isolate the contributions to the variance
   or standard deviation coming from other |GVar|\s:
   
   .. automethod:: partialvar(*args)
   
   .. automethod:: partialsdev(*args)
   
   There are two methods for converting ``self`` into a string, for 
   printing:
   
   .. automethod:: __str__
   
   .. automethod:: fmt(d=4,sep='')
   
   Two attributes and a method make reference to the original
   variables from which ``self`` is derived:
   
   .. autoattribute:: cov
   
   .. autoattribute:: der
   
   .. automethod:: dotder(v)

The following class is a specialized form of an ordered dictionary for holding
|GVar|\s (or other scalars) and arrays of |GVar|\s (or other scalars) that
supports Python pickling:

.. autoclass:: gvar.BufferDict

   The main attributes are:

   .. autoattribute:: size
   
   .. autoattribute:: flat
   
   .. attribute:: shape
      
      Always equal to ``None``. This attribute is included since 
      |BufferDict|\s share several attributes with :mod:`numpy` arrays to
      simplify coding that might support either type. Being dictionaries
      they do not have shapes in the sense of :mod:`numpy` arrays (hence 
      ``None``).
   
   The main methods are:
   
   .. automethod:: flatten()
   
   .. automethod:: slice(k)
   
   .. automethod:: isscalar(k)
   
   .. automethod:: add(k,v=None)
   
:class:`gvar.Dataset` is used to assemble random samples from multidimensional
distributions:

.. autoclass:: gvar.Dataset

   The main methods are:
   
   .. automethod:: append(*args,**kargs)
   
   .. automethod:: extend(*args,**kargs)
   
   .. automethod:: bootstrap_iter(n=None)
   
   .. automethod:: toarray()

   
Requirements
------------
:mod:`gvar` makes heavy use of :mod:`numpy` for array manipulations. It 
also uses the :mod:`numpy` code for implementing elementary functions
(*e.g.*, ``sin``, ``exp`` ...) in terms of member functions.
