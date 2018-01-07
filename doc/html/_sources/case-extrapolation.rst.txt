.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

Case Study: Simple Extrapolation
=====================================================
In this case study, we examine a simple extrapolation problem. We show first
how *not* to solve this problem. A better solution follows, together with
a discussion of priors and Bayes factors. Finally a very simple,
alternative solution, using marginalization, is described.

The Problem
------------------
Consider a problem where we have five pieces of uncorrelated data for a
function ``y(x)``::


            x[i]       y(x[i])
            ----------------------
            0.1        0.5351 (54)
            0.3        0.6762 (67)
            0.5        0.9227 (91)
            0.7        1.3803(131)
            0.95       4.0145(399)

We know that ``y(x)`` has a Taylor expansion in ``x``::

   y(x) = sum_n=0..inf p[n] x**n

The challenge is to extract a reliable estimate for ``y(0)=p[0]`` from the
data --- that is, the challenge is to fit the data and use
the fit to extrapolate the data to ``x=0``.

A Bad Solution
-----------------
One approach that is certainly wrong is to fit the data with
a power series expansion for
``y(x)`` that is truncated after five terms (``n<=4``) ---
there are only five pieces of data and such a fit would have five
parameters.
This approach gives the following fit, where the gray band shows the 1-sigma
uncertainty in the fit function evaluated with the best-fit parameters:

.. image:: eg-appendix1a.*
   :width: 80%

This fit was generated using the following code::

   import numpy as np
   import gvar as gv
   import lsqfit

   # fit data
   y = gv.gvar([
      '0.5351(54)', '0.6762(67)', '0.9227(91)', '1.3803(131)', '4.0145(399)'
      ])
   x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])

   # fit function
   def f(x, p):
      return sum(pn * x ** n for n, pn in enumerate(p))

   p0 = np.ones(5.)              # starting value for chi**2 minimization
   fit = lsqfit.nonlinear_fit(data=(x, y), p0=p0, fcn=f)
   print(fit.format(maxline=True))

Note that here the function ``gv.gvar`` converts the strings
``'0.5351(54)'``, *etc.* into |GVar|\s. Running the code gives the
following output:

.. literalinclude:: eg-appendix1a.out

This is a "perfect" fit in that the fit function agrees exactly with
the data; the ``chi**2`` for the fit is zero. The 5-parameter fit
gives a fairly precise answer for ``p[0]``
(``0.74(4)``), but the curve looks oddly stiff. Also some of the
best-fit values for the coefficients are quite
large (*e.g.*, ``p[3]= -39(4)``), perhaps unreasonably large.

A Better Solution --- Priors
-------------------------------
The problem with a 5-parameter fit is that there is no reason to neglect
terms in the expansion of ``y(x)`` with ``n>4``. Whether or not
extra terms are important depends entirely on how large we
expect the coefficients ``p[n]`` for ``n>4`` to be. The extrapolation
problem is impossible without some idea of the size of these
parameters; we need extra information.

In this case that extra information is obviously connected to questions
of convergence of the Taylor expansion we are using to model ``y(x)``.
Let's assume we know, from previous work, that the ``p[n]`` are of
order one. Then we would need to keep at least 91 terms in the
Taylor expansion if we wanted the terms we dropped to be small compared
with the 1% data errors at ``x=0.95``. So a possible fitting function
would be::

   y(x; N) = sum_n=0..N p[n] x**n

with ``N=90``.

Fitting a 91-parameter formula to five pieces of data is also impossible.
Here, however, we have extra (*prior*) information: each coefficient is order
one, which we make specific by saying that they equal 0±1. We include
these *a priori* estimates for the parameters as extra data that must be
fit, together with our original data. So we are
actually fitting 91+5 pieces of data with 91 parameters.

The prior information is introduced into the fit as a *prior*::

   import numpy as np
   import gvar as gv
   import lsqfit

   # fit data
   y = gv.gvar([
      '0.5351(54)', '0.6762(67)', '0.9227(91)', '1.3803(131)', '4.0145(399)'
      ])
   x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])

   # fit function
   def f(x, p):
      return sum(pn * x ** n for n, pn in enumerate(p))

   # 91-parameter prior for the fit
   prior = gv.gvar(91 * ['0(1)'])

   fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
   print(fit.format(maxline=True))

Note that a starting value ``p0`` is not needed when a prior is specified.
This code also gives an excellent fit, with a ``chi**2`` per degree of
freedom of ``0.35`` (note that the data point at ``x=0.95`` is off the chart,
but agrees with the fit to within its 1% errors):

.. image:: eg-appendix1b.*
   :width: 80%

The fit code output is:

.. literalinclude:: eg-appendix1b.out

This is a much more plausible fit than than the 5-parameter fit, and gives an
extrapolated value of ``p[0]=0.489(17)``. The original data points were
created using  a Taylor expansion with random coefficients, but
with ``p[0]`` set equal to ``0.5``. So this fit to the five data points (plus
91 *a priori* values for the ``p[n]`` with ``n<91``) gives the correct
result.  Increasing the number of terms further would have no effect since the
last terms added are having no impact, and so end up equal to the prior value
---  the fit data are not sufficiently precise to add new information about
these parameters.

Bayes Factors
--------------

We can test our priors for this fit by re-doing the fit with broader and
narrower priors. Setting ``prior = gv.gvar(91 * ['0(3)'])`` gives an excellent
fit,

.. literalinclude:: eg-appendix1d.out

but with a very small ``chi2/dof`` and somewhat larger errors on the best-fit
estimates for the parameters. The logarithm of the (Gaussian) Bayes Factor,
``logGBF``, can be used to compare fits with different priors. It is the
logarithm of the probability that our data would come from parameters
generated at random using the prior. The exponential of ``logGBF`` is
more than 100 times larger with the original priors of ``0(1)`` than with
priors of ``0(3)``. This says that our data is more than 100 times more
likely to come from a world with parameters of order one than from one with
parameters of order three. Put another way it says that
the size of the fluctuations in the data
are more consistent with coefficients of order one than with coefficients of
order three --- in the latter case, there would have been larger
fluctuations in the data than are actually seen.
The ``logGBF`` values argue for the original prior.

Narrower priors, ``prior = gv.gvar(91 * ['0.0(3)'])``, give a poor fit,
and also a less optimal ``logGBF``:

.. literalinclude:: eg-appendix1e.out

Setting ``prior = gv.gvar(91 * ['0(20)'])`` gives very wide priors and
a rather strange looking fit:

.. image:: eg-appendix1d.*
   :width: 80%

Here fit errors are comparable to the data errors at the data points, as you
would expect, but balloon up in between. This is an example of
*over-fitting*: the data and priors are not sufficiently accurate to fit the
number of parameters used. Specifically the priors are too broad.
Again the Bayes Factor signals the problem: ``logGBF = -14.479`` here, which
means that our data are roughly a million times (``=exp(14)``) more
likely to to come from a world with coefficients of order one than
from one with coefficients of order twenty. That is, the broad priors suggest
much larger
variations between the leading parameters than is indicated by the data ---
again, the
data are unnaturally regular in a world described by the very broad prior.

Absent useful *a priori* information about the parameters, we can sometimes use the
data to suggest a plausible width for a set of priors. We do this by setting
the width equal to the value
that maximizes ``logGBF``. This approach suggests priors of ``0.0(6)`` for
the fit above, which gives results very similar to the fit
with priors of ``0(1)``. See :ref:`empirical-bayes` for more details.

The priors are responsible for about half of the final error in our best
estimate of ``p[0]`` (with priors of ``0(1)``); the rest comes from the
uncertainty in the data. This can be  established by creating an error budget
using the code ::

    inputs = dict(prior=prior, y=y)
    outputs = dict(p0=fit.p[0])
    print(gv.fmt_errorbudget(inputs=inputs, outputs=outputs))

which prints the following table:

.. literalinclude:: eg-appendix1g.out

The table shows that the final 3.5% error comes from a 2.7% error due
to uncertainties in ``y`` and a 2.2% error from uncertainties in the
prior (added in quadrature).

.. Bayes Factors are generally quite useful for testing priors and especially
.. the widths of the priors. The width that maximizes ``logGBF`` is the
.. one most consistent (probabilistically) with the data. Priors may be
.. narrower than this, in which case prior knowledge is more accurate than
.. the fit data.
.. Priors that are much broader
.. than the width that maximizes ``logGBF`` can lead to over-fitting,
.. as illustrated above, but have no effect if the data are mostly
.. insensitve to the corresponding parameters.

.. One can often use ``logGBF`` to determine a prior's width when there is no
.. *a priori* information.  Assuming
.. we have no *a priori* idea how big the ``p[n]``\s are in the fit above,
.. for example,
.. we might set the prior widths for all of them equal to the same value and
.. tune that same value
.. to maximize ``logGBF``, since that is the value suggested by the data. (The
.. optimal width turns out to be close to one here.)



Another Solution --- Marginalization
--------------------------------------
There is a second, equivalent way of fitting this data that illustrates the
idea of *marginalization.* We really only care about parameter ``p[0]`` in
our fit. This suggests that we remove ``n>0`` terms from the data *before*
we do the fit::

  ymod[i] = y[i] - sum_n=1...inf prior[n] * x[i] ** n

Before the fit, our best estimate for the parameters is from the priors. We
use these to create an estimate for the correction to each data point
coming from ``n>0`` terms in ``y(x)``. This new data, ``ymod[i]``,
should be fit with
a new fitting function, ``ymod(x) = p[0]`` --- that is, it should be fit
to a constant, independent of ``x[i]``. The last three lines of the code
above are easily modified to implement this idea::

   import numpy as np
   import gvar as gv
   import lsqfit

   # fit data
   y = gv.gvar([
      '0.5351(54)', '0.6762(67)', '0.9227(91)', '1.3803(131)', '4.0145(399)'
      ])
   x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])

   # fit function
   def f(x, p):
      return sum(pn * x ** n for n, pn in enumerate(p))

   # prior for the fit
   prior = gv.gvar(91 * ['0(1)'])

   # marginalize all but one parameter (p[0])
   priormod = prior[:1]                       # restrict fit to p[0]
   ymod = y - (f(x, prior) - f(x, priormod))  # correct y

   fit = lsqfit.nonlinear_fit(data=(x, ymod), prior=priormod, fcn=f)
   print(fit.format(maxline=True))

Running this code give:

.. literalinclude:: eg-appendix1c.out

Remarkably this one-parameter fit gives results for ``p[0]``
that are identical (to
machine precision) to our 91-parameter fit above. The 90 parameters for
``n>0`` are said to have been *marginalized* in this fit.
Marginalizing a parameter
in this way has no effect if the fit function is linear in that parameter.
Marginalization has almost no effect for nonlinear fits as well,
provided the fit data have small errors (in which case the parameters are
effectively linear). The fit here is:

.. image:: eg-appendix1c.*
   :width: 80%

The constant is consistent with all of the data in ``ymod[i]``,
even at ``x[i]=0.95``, because ``ymod[i]`` has much larger errors for
larger ``x[i]`` because of the correction terms.

Fitting to a constant is equivalent to doing a weighted average of the
data plus the prior, so our fit can be replaced by an average::

   lsqfit.wavg(list(ymod) + list(priormod))

This again gives ``0.489(17)`` for our final result.
Note that the central value for this average is below the central
values for every data point in ``ymod[i]``. This is a consequence of large
positive correlations introduced into ``ymod`` when we remove the
``n>0`` terms. These correlations are captured automatically in our code,
and are essential --- removing the correlations between different
``ymod``\s results in a final answer, ``0.564(97)``, which has a much
larger error.