:mod:`gvar` - Gaussian Random Variables
==================================================

.. |GVar| replace:: :class:`gvar.GVar`

.. |BufferDict| replace:: :class:`gvar.BufferDict`

.. moduleauthor:: G.P. Lepage <g.p.lepage@cornell.edu>

.. module:: gvar
   :synopsis: Correlated Gaussian random variables.

Introduction 
------------------
This module provides tools for representing and manipulating Gaussian
random variables numerically. It can deal with individual variables or
arbitrarily large sets of variables, correlated or uncorrelated. It 
also supports complicated (Python) functions of Gaussian
variables.

A Gaussian variable is a random variable that
represents a *typical* random number drawn from a particular Gaussian (or
normal) probability distribution; more precisely, it represents the entire
probability distribution. A given Gaussian variable ``x`` is therefore
completely characterized by its mean ``x.mean`` and standard deviation
``x.sdev``.
    
A mathematical function of a Gaussian variable can be defined as the
probability distribution of function values obtained by evaluating the
function for random numbers drawn from the original distribution. The
distribution of function values is itself approximately Gaussian provided the
standard deviation of the Gaussian variable is sufficiently small. Thus we can
define a function ``f`` of a Gaussian variable ``x`` to be a Gaussian variable
itself, with ::
    
    f(x).mean = f(x.mean)
    f(x).sdev = x.sdev |f'(x.mean)|,
    
which follows from linearizing the ``x`` dependence of ``f(x)`` about point
``x.mean``. (This obviously fails at an extremum of ``f(x)``, where 
``f'(x)=0``.)
    
The last formula, together with its multidimensional generalization, leads
to a full calculus for Gaussian random variables that assigns
Gaussian-variable values to arbitrary arithmetic expressions and functions
involving Gaussian variables. 
A multidimensional collection ``x[i]`` of Gaussian variables is
characterized by the means ``x[i].mean`` for each variable, together with a
covariance matrix ``cov[i, j]``. Diagonal elements of ``cov`` specify the
standard deviations of different variables: ``x[i].sdev = cov[i, i]**0.5``.
Nonzero off-diagonal elements imply correlations between different
variables::
    
    cov[i, j] = <x[i]*x[j]>  -  <x[i]> * <x[j]>
    
where ``<y>`` denotes the expectation value or mean for a random variable
``y``.

Gaussian variables are represented in :mod:`gvar` 
by objects of type |GVar|. |GVar|\s are particularly useful 
for error propagation because the errors 
(i.e., standard deviations) associated with them 
propagate through arbitrarily complex Python functions. This is true 
of errors from individual |GVar|\s,
and also of the correlations between different 
|GVar|\s. In the following sections we demonstrate how to create
|GVar|\s and how to manipulate them to solve common problems in
error propagation. They are also very useful for creating 
correlated Gaussian priors for Bayesian analyses.

.. _creating-gaussian-variables:

Creating Gaussian Variables
---------------------------
Objects of type |GVar| are of two types: 1) primary |GVar|\s 
that are created from means and covariances using 
:func:`gvar.gvar`; and 2) derived |GVar|\s that result 
from arithmetic expressions or functions involving |GVar|\s 
(primary or derived).
A single (primary) |GVar| is 
created from its mean ``xmean`` and standard deviation
``xsdev`` using::
    
 	x = gvar.gvar(xmean, xsdev).
    
This function can also be used to convert strings like ``"-72.374(22)"`` 
or ``"511.2 +- 0.3"`` into |GVar|\s: for example, ::
    
    >>> import gvar
    >>> x = gvar.gvar(3.1415, 0.0002)
    >>> print(x)
    3.14150(20)
    >>> x = gvar.gvar("3.1415(2)")
    >>> print(x)
    3.14150(20)
    >>> x = gvar.gvar("3.1415 +- 0.0002")
    >>> print(x)
    3.14150(20)

Note that ``x = gvar.gvar(x)`` is useful when you are unsure 
whether ``x`` is a |GVar| or a string representing a |GVar|.
    
|GVar|\s are usually more interesting when used to describe multidimensional
distributions, especially if there are correlations between different
variables. Such distributions are represented by collections of |GVar|\s in
one of two standard formats: 1) :mod:`numpy`  arrays of |GVar|\s (any
shape); or, more flexibly, 2) Python dictionaries whose values are |GVar|\s or
arrays of |GVar|\s. Most functions in :mod:`gvar` that handle multiple
|GVar|\s work with either format, and if they return multidimensional results
do so in the same format as the inputs (that is, arrays or dictionaries). Any
dictionary is converted internally into a specialized (ordered) dictionary of
type |BufferDict|, and dictionary-valued results are also |BufferDict|\s.
    
To create an array of |GVar|\s with mean values specified by array
``xmean`` and covariance matrix ``xcov``, use ::
    
	x = gvar.gvar(xmean, xcov)
    
where array ``x`` has the same shape as ``xmean`` (and ``xcov.shape =
xmean.shape+xmean.shape``). Then each element ``x[i]`` of a one-dimensional
array, for example, is a |GVar| where::
    
    x[i].mean = xmean[i]         # mean of x[i]
    x[i].val  = xmean[i]         # same as x[i].mean
    x[i].sdev = xcov[i, i]**0.5  # std deviation of x[i]
    x[i].var  = xcov[i, i]       # variance of x[i]
    
As an example, ::
    
    >>> x, y = gvar.gvar([0.1, 10.], [[0.015625, 0.24], [0.24, 4.]])
    >>> print('x =', x, '   y =', y)
    x = 0.10(13)    y = 10.0(2.0)
    
makes ``x`` and ``y`` |GVar|\s with standard deviations ``sigma_x=0.125`` and
``sigma_y=2``, and a fairly strong statistical correlation::

    >>> print(gvar.evalcov([x, y]))     # covariance matrix
    [[ 0.015625  0.24    ]
     [ 0.24      4.      ]]
    >>> print(gvar.evalcorr([x, y]))    # correlation matrix
    [[ 1.    0.96]
     [ 0.96  1.  ]]

Here functions :func:`gvar.evalcov` and :func:`gvar.evalcorr` compute the
covariance and correlation matrices, respectively, of the list of 
|GVar|\s in their arguments.

:func:`gvar.gvar` can also be used to convert strings or tuples stored in
arrays or dictionaries into |GVar|\s: for example, ::
    
    >>> garray = gvar.gvar(['2(1)', '10+-5', (99, 3), gvar.gvar(0, 2)])
    >>> print(garray)
    [2.0(1.0) 10.0(5.0) 99.0(3.0) 0.0(2.0)]
    >>> gdict = gvar.gvar(dict(a='2(1)', b=['10+-5', (99, 3), gvar.gvar(0, 2)]))
    >>> print(gdict)
    {'a': 2.0(1.0),'b': array([10.0(5.0), 99.0(3.0), 0.0(2.0)], dtype=object)}
    
If the covariance matrix in ``gvar.gvar`` is diagonal, it can be replaced
by an array of standard deviations (square roots of diagonal entries in
``cov``). The example above without correlations, therefore, would be::
    
    >>> x, y = gvar.gvar([0.1, 10.], [0.125, 2.])
    >>> print('x =', x, '   y =', y)
    x = 0.10(12)    y = 10.0(2.0)
    >>> print(gvar.evalcov([x, y]))     # covariance matrix
    [[ 0.015625  0.      ]
     [ 0.        4.      ]]
    >>> print(gvar.evalcorr([x, y]))    # correlation matrix
    [[ 1.  0.]
     [ 0.  1.]]
    
.. _gvar-arithmetic-and-functions:
    
|GVar| Arithmetic and Functions
-------------------------------------------
The |GVar|\s discussed in the previous section are all *primary* |GVar|\s
since they were created by specifying their means and covariances 
explicitly, using :func:`gvar.gvar`. What makes |GVar|\s particularly
useful is that they can be used in 
arithemtic expressions (and numeric pure-Python functions), just like
Python floats. Such expressions result in new, *derived* |GVar|\s 
whose means, standard deviations, and correlations 
are determined from the covariance matrix of the 
primary |GVar|\s. The
automatic propagation of correlations 
through arbitrarily complicated arithmetic is an especially useful
feature of |GVar|\s.
    
As an example, again define
    
    >>> x, y = gvar.gvar([0.1, 10.], [0.125, 2.])
    
and set
    
    >>> f = x + y
    >>> print('f =', f)
    f = 10.1(2.0)
    
Then ``f`` is a (derived) |GVar| whose variance ``f.var`` equals ::
    
    df/dx cov[0, 0] df/dx + 2 df/dx cov[0, 1] df/dy + ... = 2.0039**2
    
where ``cov`` is the original covariance matrix used to define ``x`` and
``y`` (in ``gvar.gvar``). Note that while ``f`` and ``y`` separately have
20% uncertainties in this example, the ratio ``f/y`` has much smaller
errors::
    
    >>> print(f / y)
    1.010(13)

This happens, of course, because the errors in ``f`` and ``y`` are highly
correlated --- the error in ``f`` comes mostly from ``y``. |GVar|\s 
automatically track correlations even through complicated arithmetic
expressions and functions: for example, the following 
more complicated ratio has a still
smaller error, because of stronger correlations between numerator and
denominator::

    >>> print(gvar.sqrt(f**2 + y**2) / f)
    1.4072(87)
    >>> print(gvar.evalcorr([f, y]))
    [[ 1.          0.99805258]
     [ 0.99805258  1.        ]]
    >>> print(gvar.evalcorr([gvar.sqrt(f**2 + y**2), f]))
    [[ 1.         0.9995188]
     [ 0.9995188  1.       ]]

The :mod:`gvar` module defines versions of the standard Python mathematical
functions that work with |GVar| arguments. These include:
``exp, log, sqrt, sin, cos, tan, arcsin, arccos, arctan, sinh, cosh, tanh,
arcsinh, arccosh, arctanh``. Numeric functions defined 
entirely in Python (*i.e.*, pure-Python functions) 
will likely also work with |GVar|\s.

Numeric functions implemented by modules using low-level languages like C  
will *not* work with |GVar|\s. Such functions must 
be replaced by equivalent code written 
directly in Python. In some cases it is possible to construct  
a |GVar|-capable function from low-level code for the function and its
derivative. For example, the following code defines a new version of the
standard Python error function that accepts either floats or |GVar|\s 
as its argument::

    import math
    import gvar

    def erf(x):
        if isinstance(x, gvar.GVar):
            f = math.erf(x.mean)
            dfdx = 2. * math.exp(- x.mean ** 2) / math.sqrt(math.pi)
            return gvar.gvar_function(x, f, dfdx)
        else:
            return math.erf(x)

Here function :func:`gvar.gvar_function` creates the |GVar| for a function with
mean value ``f`` and derivative ``dfdx`` at point ``x``. 

Some sample numerical analysis codes, adapted for use with |GVar|\s, are 
described in :ref:`numerical-analysis-modules-in-gvar`.

Arithmetic operators ``+ - * / ** == != <> += -= *= /=`` are all defined
for |GVar|\s. Comparison operators are also supported: ``== != > >= < <=``.
They are applied to the mean values of |GVar|\s: for example, 
``gvar.gvar(1,1) == gvar.var(1,2)`` is true, as is ``gvar.gvar(1,1) > 0``.
Logically ``x>y`` for |GVar|\s should evaluate to a boolean-valued random 
variable, but such variables are beyond the scope of this module. 
Comparison operators that act only on the mean values make it easier to implement 
pure-Python functions that work with either |GVar|\s or :class:`float`\s
as arguments. 

*Implementation Notes:* Each |GVar| keeps track of three
pieces of information: 1) its mean value; 2) its derivatives with respect to
the primary |GVar|\s (created by :func:`gvar.gvar`); 
and 3) the location of the covariance matrix for the primary |GVar|\s.
The derivatives and covariance matrix allow one to compute the
standard deviation of the |GVar|, as well as correlations between it and any
other function of the primary |GVar|\s. The derivatives for 
derived |GVar|\s are computed automatically, using *automatic 
differentiation*. 

The derivative of a |GVar| ``f`` with 
respect to a primary |GVar| ``x`` is obtained from ``f.deriv(x)``. A list
of derivatives with respect to all primary |GVar|\s is given by ``f.der``,
where the order of derivatives is the same as the order in which the primary
|GVar|\s were created.  


A |GVar| can be constructed at a
very low level by supplying all the three 
essential pieces of information --- for example, ::
    
    f = gvar.gvar(fmean, fder, cov)
    
where ``fmean`` is the mean, ``fder`` is an array where ``fder[i]`` is the
derivative of ``f`` with respect to the ``i``-th primary |GVar|
(numbered in the order in which they were created using :func:`gvar.gvar`),
and ``cov`` is the covariance matrix for the primary |GVar|\s (easily
obtained from an existing |GVar| ``x`` using ``x.cov``).

Error Budgets from |GVar|\s
------------------------------------
It is sometimes useful to know how much of the uncertainty in a derived quantity
is due to a particular input uncertainty. Continuing the example above, for
example, we might want to know how much of ``f``\s standard deviation
is due to the standard deviation of ``x`` and how much comes from ``y``. 
This is easily computed::
    
    >>> x, y = gvar.gvar([0.1, 10.], [0.125, 2.])
    >>> f = x + y
    >>> print(f.partialsdev(x))        # uncertainty in f due to x
    0.125
    >>> print(f.partialsdev(y))        # uncertainty in f due to y
    2.0
    >>> print(f.partialsdev(x, y))     # uncertainty in f due to x and y
    2.00390244274
    >>> print(f.sdev)                  # should be the same
    2.00390244274

This shows, for example, that most (2.0) of the uncertainty in ``f`` (2.0039)
is from ``y``. 

:mod:`gvar` provides a useful tool for compiling an "error budget" for 
derived |GVar|\s relative to the primary |GVar|\s from which they
were constructed: continuing the example above, ::

    >>> outputs = {'f':f, 'f/y':f/y}
    >>> inputs = {'x':x, 'y':y}
    >>> print(gvar.fmt_values(outputs))
    Values:
                    f/y: 1.010(13)           
                      f: 10.1(2.0)           

    >>> print(gvar.fmt_errorbudget(outputs=outputs, inputs=inputs))
    Partial % Errors:
                               f/y         f
    ----------------------------------------
                      y:      0.20     19.80
                      x:      1.24      1.24
    ----------------------------------------
                  total:      1.25     19.84
    
This shows ``y`` is responsible for 19.80% of the 19.84% uncertainty in ``f``,
but only 0.2% of the 1.25% uncertainty in ``f/y``. The total uncertainty in each case
is obtained by adding the ``x`` and ``y`` contributions in quadrature.


.. _storing-gvars-for-later-use:

Storing |GVar|\s for Later Use; |BufferDict|\s
--------------------------------------------------
Storing |GVar|\s in a file for later use is complicated by the need to 
capture the covariances between different |GVar|\s as well as their 
means. To pickle an array or dictionary ``g`` of |GVar|\s, for example,
we might use ::

    >>> gtuple = (gvar.mean(g), gvar.evalcov(g))
    >>> import pickle
    >>> pickle.dump(gtuple, open('outputfile.p', 'wb'))

to extract the means and covariance matrix into a tuple which then 
is saved in file ``'output.p'`` using Python's standard :mod:`pickle` 
module. To reassemble the |GVar|\s we use::

    >>> g = gvar.gvar(pickle.load('outputfile.p', 'rb'))

where :func:`pickle.load` reads ``gtuple`` back in, and :func:`gvar.gvar`
converts it back into a collection of |GVar|\s. The correlations between
different |GVar|\s  in the original array/dictionary ``g`` are preserved here,
but their correlations with other |GVar|\s are lost. So it is important to
include all |GVar|\s of  interest in a single array or dictionary before
saving them.

This recipe works for ``g``\s that are: single |GVar|\s, arrays of |GVar|\s
(any shape), or dictionaries whose values are |GVar|\s and/or arrays  of
|GVar|\s. For convenience, it is implemented in functions :func:`gvar.dump`, 
:func:`gvar.dumps`, :func:`gvar.load`, and :func:`gvar.loads`.

Pickling is simplified if the |GVar|\s that need saving are all
in a |BufferDict| since these can be serialized and saved to a file
again using Python's :mod:`pickle` module. So if ``g`` is a 
|BufferDict| containing |GVar|\s (and/or arrays of |GVar|\s), ::

    >>> import pickle
    >>> pickle.dump(g, open('outputfile.p', 'wb'))

saves the contents of ``g`` to a file named ``outputfile.p``. 
The |GVar|\s are retrieved using::

    >>> g = pickle.load(open('outputfile.p', 'rb'))

|BufferDict|\s also have methods that allow saving their contents 
using Python's :mod:`json` module rather than :mod:`pickle`.

  
.. _gvar-random-number-generators:
    
Random Number Generators
------------------------
|GVar|\s represent probability distributions. It is possible to use them
to generate random numbers from those distributions. For example, in
    
    >>> z = gvar.gvar(2.0, 0.5)
    >>> print(z())
    2.29895701465
    >>> print(z())
    3.00633184275
    >>> print(z())
    1.92649199321
    
calls to ``z()`` generate random numbers from a Gaussian random number 
generator with mean ``z.mean=2.0`` and standard deviation ``z.sdev=0.5``.
    
To obtain random arrays from an array ``g`` of |GVar|\s
use ``giter=gvar.raniter(g)`` (see :func:`gvar.raniter`) to create a
random array generator ``giter``. Each call to ``next(giter)`` generates 
a new array of random numbers. The random number arrays have the same 
shape as the array ``g`` of |GVar|\s and have the distribution implied 
by those random variables (including correlations). For example,
    
    >>> a = gvar.gvar(1.0, 1.0)
    >>> da = gvar.gvar(0.0, 0.1)
    >>> g = [a, a+da]
    >>> giter = gvar.raniter(g)
    >>> print(next(giter))
    [ 1.51874589  1.59987422]
    >>> print(next(giter))
    [-1.39755111 -1.24780937]
    >>> print(next(giter))
    [ 0.49840244  0.50643312]
    
Note how the two random numbers separately vary over the region 1±1
(approximately), but the separation between the two is rarely more than
0±0.1. This is as expected given the strong correlation between ``a``
and ``a+da``.
    
``gvar.raniter(g)`` also works when ``g`` is a dictionary (or
:class:`gvar.BufferDict`) whose entries ``g[k]`` are |GVar|\s or arrays of
|GVar|\s. In such cases the iterator returns a dictionary with the same
layout::
    
    >>> g = dict(a=gvar.gvar(0, 1), b=[gvar.gvar(0, 100), gvar.gvar(10, 1e-3)])
    >>> print(g)
    {'a': 0.0(1.0), 'b': [0(100), 10.0000(10)]}
    >>> giter = gvar.raniter(g)
    >>> print(next(giter))
    {'a': -0.88986130981173306, 'b': array([-67.02994213,   9.99973707])}
    >>> print(next(giter))
    {'a': 0.21289976681277872, 'b': array([ 29.9351328 ,  10.00008606])}
    
One use for such random number generators is dealing with situations where
the standard deviations are too large to justify the linearization 
assumed in defining functions of Gaussian variables. Consider, for example,
    
    >>> x = gvar.gvar(1., 3.)
    >>> print(cos(x))
    0.5(2.5)
    
The standard deviation for ``cos(x)`` is obviously wrong since ``cos(x)``
can never be larger than one. To obtain the real mean and standard deviation,
we generate a large number of random numbers ``xi`` from ``x``, compute 
``cos(xi)`` for each, and compute the mean and standard deviation for the
resulting distribution (or any other statistical quantity, particularly if
the resulting distribution is not Gaussian)::
    
    # estimate mean,sdev from 1000 random x's
    >>> ran_x = numpy.array([x() for in range(1000)]) 
    >>> ran_cos = numpy.cos(ran_x)
    >>> print('mean =', ran_cos.mean(), '  std dev =', ran_cos.std())
    mean = 0.0350548954142   std dev = 0.718647118869
    
    # check by doing more (and different) random numbers
    >>> ran_x = numpy.array([x() for in range(100000)])
    >>> ran_cos = numpy.cos(ran_x)
    >>> print('mean =', ran_cos.mean(), '  std dev =', ran_cos.std())
    mean = 0.00806276057656   std dev = 0.706357174056
    
This procedure generalizes trivially for multidimensional analyses, using 
arrays or dictionaries with :func:`gvar.raniter`.
    
Finally note that *bootstrap* copies of |GVar|\s are easily created. A
bootstrap copy of |GVar| ``x ± dx`` is another |GVar| with the same width but
where the mean value is replaced by a random number drawn from the original
distribution. Bootstrap copies of a data set, described by a collection of
|GVar|\s, can be used as new (fake) data sets having the same statistical
errors and correlations::
    
    >>> g = gvar.gvar([1.1, 0.8], [[0.01, 0.005], [0.005, 0.01]])
    >>> print(g)
    [1.10(10) 0.80(10)]
    >>> print(gvar.evalcov(g))                  # print covariance matrix
    [[ 0.01   0.005]
     [ 0.005  0.01 ]]
    >>> gbs_iter = gvar.bootstrap_iter(g)
    >>> gbs = next(gbs_iter)                    # bootstrap copy of f
    >>> print(gbs)
    [1.14(10) 0.90(10)]                         # different means
    >>> print(gvar.evalcov(gbs))
    [[ 0.01   0.005]                            # same covariance matrix
     [ 0.005  0.01 ]]
    
Such fake data sets are useful for analyzing non-Gaussian behavior, for
example, in nonlinear fits.
    
    
Limitations
-----------
The most fundamental limitation of this module is that the calculus of
Gaussian variables that it assumes is only valid when standard deviations
are small (compared to the distances over which the functions of interest
change appreciably). One way of dealing with this limitation is described
above in the section on :ref:`gvar-random-number-generators`.
    
Another potential issue is roundoff error, which can become problematic if
there is a wide range of standard deviations among correlated modes. For
example, the following code works as expected::
    
    >>> from gvar import gvar, evalcov
    >>> tiny = 1e-4
    >>> a = gvar(0., 1.)
    >>> da = gvar(tiny, tiny)
    >>> a, ada = gvar([a.mean, (a+da).mean], evalcov([a, a+da])) # = a,a+da
    >>> print(ada-a)   # should be da again
    0.00010(10)
    
Reducing ``tiny``, however, leads to problems::
    
    >>> from gvar import gvar, evalcov
    >>> tiny = 1e-8
    >>> a = gvar(0., 1.)
    >>> da = gvar(tiny, tiny)
    >>> a, ada = gvar([a.mean, (a+da).mean], evalcov([a, a+da])) # = a, a+da
    >>> print(ada-a)   # should be da again
    1(0)e-08
    
Here the call to :func:`gvar.evalcov` creates a new covariance matrix for
``a`` and ``ada = a+da``, but the matrix does not have enough numerical
precision to encode the size of ``da``'s variance, which gets set, in
effect, to zero. The problem arises here for values of ``tiny`` less than
about 2e-8 (with 64-bit floating point numbers --- ``tiny**2`` is what
appears in the covariance matrix).
    
    
Optimizations 
------------------------------------------------
When there are lots of primary |GVar|\s, the number of derivatives stored 
for each derived |GVar| can
become rather large, potentially (though not necessarily) leading to slower
calculations. One way to alleviate this problem, should it arise, is to
separate the primary variables into groups that are never mixed in
calculations and to use different :func:`gvar.gvar`\s when generating the
variables in different groups. New versions of :func:`gvar.gvar` are
obtained using :func:`gvar.switch_gvar`: for example, ::
    
    import gvar
    ...
    x = gvar.gvar(...)
    y = gvar.gvar(...)
    z = f(x, y)
    ... other manipulations involving x and y ...
    gvar.switch_gvar()
    a = gvar(...)
    b = gvar(...)
    c = g(a, b)
    ... other manipulations involving a and b (but not x and y) ...
    
Here the :func:`gvar.gvar` used to create ``a`` and ``b`` is a different
function than the one used to create ``x`` and ``y``. A derived quantity,
like ``c``, knows about its derivatives with respect to ``a`` and ``b``,
and about their covariance matrix; but it carries no derivative information
about ``x`` and ``y``. Absent the ``switch_gvar`` line, ``c`` would have
information about its derivatives with respect to ``x`` and ``y`` (zero
derivative in both cases) and this would make calculations involving ``c``
slightly slower than with the ``switch_gvar`` line. Usually the difference
is negligible --- it used to be more important, in earlier implementations
of |GVar| before sparse matrices were introduced to keep track of
covariances. Note that the previous :func:`gvar.gvar` can be restored using
:func:`gvar.restore_gvar`.


Utilities
----------
The function used to create Gaussian variable objects is:

.. autofunction:: gvar.gvar(...)

The following function is useful for constructing new functions that
can accept |GVar|\s as arguments:

.. autofunction:: gvar.gvar_function(x, f, dfdx)

Means, standard deviations, variances, formatted strings, covariance
matrices and correlation/comparison information can be extracted from arrays 
(or dictionaries) of |GVar|\s using:

.. autofunction:: gvar.mean(g)

.. autofunction:: gvar.sdev(g)

.. autofunction:: gvar.var(g)

.. autofunction:: gvar.fmt(g, ndecimal=None, sep='')

.. autofunction:: gvar.evalcov(g)

.. autofunction:: gvar.evalcorr(g)

.. autofunction:: gvar.uncorrelated(g1, g2)

.. autofunction:: gvar.chi2(g1, g2, svdcut=1e-15)

.. autofunction:: gvar.fmt_chi2(f)

|GVar|\s can be stored (pickled) and retrieved from files (or strings) using:

.. autofunction:: gvar.dump(g, outputfile)

.. autofunction:: gvar.dumps(g)

.. autofunction:: gvar.load(inputfile)

.. autofunction:: gvar.loads(inputstring)

|GVar|\s contain information about derivatives with respect to the *independent*
|GVar|\s from which they were constructed. This information can be extracted using:

.. autofunction:: gvar.deriv(g, x)

The following function creates an iterator that generates random arrays
from the distribution defined by array (or dictionary) ``g`` of |GVar|\s. 
The random numbers incorporate any correlations implied by the ``g``\s.

.. autofunction:: gvar.raniter(g, n=None, svdcut=None)

.. autofunction:: gvar.bootstrap_iter(g, n=None, svdcut=None)

.. autofunction:: gvar.ranseed(a)

The following two functions that are useful for tabulating results 
and for analyzing where the errors in a |GVar| constructed from 
other |GVar|\s come from:

.. autofunction:: gvar.fmt_errorbudget(outputs, inputs, ndecimal=2, percent=True, colwidth=10)

.. autofunction:: gvar.fmt_values(outputs, ndecimal=None)

The following function applies an SVD cut to the correlation matrix 
of a set of |GVar|\s:

.. autofunction:: gvar.svd

This function is useful when the correlation matrix is singular 
or almost singular, and its inverse is needed (as in curve fitting).

The following function can be used to rebuild collections of |GVar|\s, 
ignoring all correlations with other variables. It can also be used to 
introduce correlations between uncorrelated variables.

.. autofunction:: gvar.rebuild(g, gvar=gvar, corr=0.0)

   
The following functions creates new functions that generate |GVar|\s (to 
replace :func:`gvar.gvar`):

.. autofunction:: gvar.switch_gvar()

.. autofunction:: gvar.restore_gvar()

.. autofunction:: gvar.gvar_factory(cov=None)

|GVar|\s created by different functions cannot be combined in arithmetic
expressions (the error message "Incompatible GVars." results). 

   
Classes
-------
The fundamental class for representing Gaussian variables is:

.. autoclass:: gvar.GVar
   
   The basic attributes are:
   
   .. autoattribute:: mean
   
   .. autoattribute:: sdev
   
   .. autoattribute:: var
   
   Two methods allow one to isolate the contributions to the variance
   or standard deviation coming from other |GVar|\s:
   
   .. automethod:: partialvar(*args)
   
   .. automethod:: partialsdev(*args)

   Partial derivatives of the |GVar| with respect to the independent
   |GVar|\s from which it was constructed are given by:

   .. automethod:: deriv(x)
   
   There are two methods for converting ``self`` into a string, for 
   printing:
   
   .. automethod:: __str__
   
   .. automethod:: fmt(ndecimal=None, sep='')
   
   Two attributes and a method make reference to the original
   variables from which ``self`` is derived:
   
   .. attribute:: cov
      
      Underlying covariance matrix (type :class:`gvar.smat`) shared by all
      |GVar|\s.
   
   .. autoattribute:: der
   
   .. automethod:: dotder(v)

The following class is a specialized form of an ordered dictionary for
holding |GVar|\s (or other scalars) and arrays of |GVar|\s (or other
scalars) that supports Python pickling:

.. autoclass:: gvar.BufferDict

   The main attributes are:

   .. autoattribute:: size
   
   .. autoattribute:: flat

   .. autoattribute:: dtype
   
   .. attribute:: buf
   
      The (1d) buffer array. Allows direct access to the buffer: for example,
      ``self.buf[i] = new_val`` sets the value of the ``i-th`` element in
      the buffer to value ``new_val``.  Setting ``self.buf = nbuf``
      replaces the old buffer by new buffer ``nbuf``. This only works if
      ``nbuf`` is a one-dimensional :mod:`numpy` array having the same
      length as the old buffer, since ``nbuf`` itself is used as the new
      buffer (not a copy).
      
   .. attribute:: shape
      
      Always equal to ``None``. This attribute is included since 
      |BufferDict|\s share several attributes with :mod:`numpy` arrays to
      simplify coding that might support either type. Being dictionaries
      they do not have shapes in the sense of :mod:`numpy` arrays (hence 
      the shape is ``None``).
   
   The main methods are:
   
   .. automethod:: flatten()
   
   .. automethod:: slice(k)
   
   .. automethod:: isscalar(k)
   
   .. method:: update(d)
      
      Add contents of dictionary ``d`` to ``self``.
      
   .. staticmethod:: BufferDict.load(fobj, use_json=False)
   
      Load serialized |BufferDict| from file object ``fobj``.
      Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
      it uses :mod:`json` (obvioulsy).
      
   .. staticmethod:: loads(s, use_json=False)
      
      Load serialized |BufferDict| from string object ``s``.
      Uses :mod:`pickle` unless ``use_json`` is ``True``, in which case
      it uses :mod:`json` (obvioulsy).
   
   .. automethod:: dump(fobj, use_json=False)
   
   .. automethod:: dumps(use_json=False)
      
SVD analysis is handled by the following class:

.. autoclass:: gvar.SVD(mat, svdcut=None, svdnum=None, compute_delta=False, rescale=False)

   .. automethod:: decomp(n)

   
Requirements
------------
:mod:`gvar` makes heavy use of :mod:`numpy` for array manipulations. It 
also uses the :mod:`numpy` code for implementing elementary functions
(*e.g.*, ``sin``, ``exp`` ...) in terms of member functions.
