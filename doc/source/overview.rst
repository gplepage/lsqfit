Overview and Tutorial
========================

Introduction
--------------------
The modules defined in here are designed to facilitate
least-squares fitting of noisy data by multi-dimensional, nonlinear
functions of arbitrarily many parameters. The central module is
:mod:`lsqfit` because it provides the fit functions. :mod:`lsqfit` makes
heavy use of auxiliary module :mod:`gvar`, which provides tools that
facilitate the analysis of error propagation, and also the creation of
complicated multi-dimensional gaussian distributions.

The following (complete) code illustrates basic usage of :mod:`lsqfit`::
   
   import numpy as np
   import gvar as gv
   import lsqfit
   
   y = {                      # data for the dependent variable
      "data1" : gv.gvar([1.376,2.010],[[ 0.0047,0.01],[ 0.01,0.056]]),
      "data2" : gv.gvar([1.329,1.582],[[ 0.0047,0.0067],[0.0067,0.0136]]),
      "b/a"   : gv.gvar(2.0,0.5)
   }
   x = {                      # independent variable
      "data1" : np.array([0.1,1.0]),
      "data2" : np.array([0.1,0.5])
   }
   prior = dict(a=gv.gvar(0.5,0.5),b=gv.gvar(0.5,0.5))
   
   def fcn(x,p):                # fit function of x and parameters p[k]
      ans = {}
      for k in ["data1","data2"]:
         ans[k] = gv.exp(p['a'] + x[k]*p['b'])
      ans['b/a'] = p['b']/p['a']
      return ans
      
   # do the fit   
   fit = lsqfit.nonlinear_fit(data=(x,y),prior=prior,fcn=fcn)
   print(fit.format(100))     # print standard summary of fit
   
   p = fit.p                  # best-fit values for parameters
   outputs = dict(a=p['a'],b=p['b'])
   outputs['b/a'] = p['b']/p['a']
   inputs = dict(y=y,prior=prior)
   print(gv.fmt_values(outputs))              # tabulate outputs
   print(gv.fmt_errorbudget(outputs,inputs))  # print error budget for outputs
   
   # save best-fit values in file "outputfile.p" for later use
   import pickle
   pickle.dump(fit.p,open("outputfile.p","wb"))

This code fits the function ``f(x,a,b)= exp(a+b*x)`` (see ``fcn(x,p)``) to two
sets of data, labeled ``data1`` and ``data2``, by varying parameters ``a`` and
``b`` until ``f(x["data1"],a,b)`` and ``f(x["data2"],a,b)`` equal
``y["data1"]`` and ``y["data2"]``, respectively, to within the ``y``\s'
errors. The means and covariance matrices for the ``y``\s are specified in the
``gv.gvar(...)``\s used to create them: for example, ::
   
   >>> print y['data1']
   [1.376 +- 0.0685565 2.01 +- 0.236643]

shows the means and standard deviations for the data in the first data set
(``0.0685565`` is the square root of the ``0.0047`` in the covariance matrix).
The dictionary ``prior`` gives *a priori* estimates for the two parameters,
``a`` and ``b``: each is assumed to be ``0.5 +- 0.5`` before fitting. In
addition, there is an extra piece of input data, ``y["b/a"]``, which indicates
that ``b/a`` is ``2.0 +- 0.5``. The fit function for this data is simply the
ratio ``b/a`` (represented by ``p['b']/p['a']`` in fit function ``fcn(x,p)``.)

The output from the code sample above is::

   Least Square Fit:
     chi2/dof [dof] = 0.17 [5]    Q = 0.97    logGBF = -5.2381    itns = 5

   Parameters:
                 a_    0.252798 +-    0.032           (     0.5 +-      0.5)
                 b_    0.448762 +-    0.065           (     0.5 +-      0.5)

   Fit:
           key          y_i      f(x_i)        dy_i
   ------------------------------------------------
           b/a_           2      1.7752         0.5
         data1_       1.376      1.3467    0.068557
              _        2.01      2.0169     0.23664
         data2_       1.329      1.3467    0.068557
              _       1.582      1.6115     0.11662

   Values:
                     a: 0.253(32)           
                   b/a: 1.775(298)          
                     b: 0.449(65)           

   Partial % Errors:
                                a       b/a         b
   --------------------------------------------------
                     y:     12.75     16.72     14.30
                 prior:      0.92      1.58      1.88
   --------------------------------------------------
                 total:     12.78     16.80     14.42

The best-fit values for ``a`` and ``b`` are ``0.253(32)`` and ``0.449(65)``,
respectively; and the best-fit result for ``b/a`` is ``1.775(298)``, which,
because of correlations, is slightly more accurate than might be expected from
the separate errors for ``a`` and ``b``. The error budget, at the end, for
each of these three quantities shows that the bulk of the error in each case
comes from uncertainties in the ``y`` data, with only small contributions
from uncertainties in the priors ``prior``.
   
The last section of the code uses Python's :mod:`pickle` module to save the
best-fit values of the parameters in a file for later use. They are recovered
using :mod:`pickle` again::
   
   >>> import pickle
   >>> p = pickle.load(open("outputfile.p","rb"))
   >>> print(p['a'])
   0.252798 +- 0.0323152
   >>> print(p['b'])
   0.448762 +- 0.0647224
   >>> print(p['b']/p['a'])
   1.77518 +- 0.298185
   
The recovered parameters are :class:`gvar.GVar`\s, with their full covariance
matrix intact. (:mod:`pickle` works here because the variables in ``fit.p``
are stored in a special dictionary of type :class:`gvar.BufferDict`;
:class:`gvar.GVar`\s cannot be pickled otherwise.)
   
Note that the constraint in ``y`` on ``b/a`` in this example is much tighter
than the constraints on ``a`` and ``b`` separately. This suggests a variation
on the previous code, where the tight restriction on ``b/a`` is built into the
prior rather than ``y``::

   ... as before ...
   
   y = {                      # data for the dependent variable
      "data1" : gv.gvar([1.376,2.010],[[ 0.0047,0.01],[ 0.01,0.056]]),
      "data2" : gv.gvar([1.329,1.582],[[ 0.0047,0.0067],[0.0067,0.0136]])
   }
   x = {                      # independent variable
      "data1" : np.array([0.1,1.0]),
      "data2" : np.array([0.1,0.5])
   }
   prior = dict(a=gv.gvar(0.5,0.5))
   prior['b'] = prior['a']*gv.gvar(2.0,0.5)

   def fcn(x,p):              # fit function of x and parameters p[k]
      ans = {}
      for k in ["data1","data2"]:
         ans[k] = gv.exp(p['a'] + x[k]*p['b'])
      return ans
      
   ... as before ...

Here the dependent data ``y`` no longer has an entry for ``b/a``, and neither
do results from the fit function; but the prior for ``b`` is now ``2 +-
0.5`` times the prior for ``a``, thereby introducing a correlation that
limits the ratio ``b/a`` to be ``2 +- 0.5`` in the fit. This code gives almost
identical results to the first one --- very slightly less accurate, since
there is less input data. We can often move information from the ``y`` data to
the prior or back since both are forms of input information.

What follows is a brief tutorial that demonstrates in greater detail how to
use these modules in some standard variations on the data fitting problem.
As above, code for the examples is specified completely and so can be copied
into a file, and run as is. It can also be modified, allowing for
experimentation.

.. _making-fake-data:

Making Fake Data
----------------
We need data in order to demonstrate curve fitting. The easiest route
is to make fake data. The recipe is simple: 1) choose some well defined
function ``f(x)`` of the independent variable ``x``; 2) choose values for
the ``x``\s, and therefore the "correct" values for ``y=f(x)``; and 3) add
random noise to the ``y``\s, to simulate measurement errors. Here we will work
through a simple implementation of this recipe to illustrate how the
:mod:`gvar` module can be used to build complicated gaussian distributions (in
this case for the correlated noise in the ``y``\s). A reader eager to fit
real data can skip this section on first reading.

For the function ``f`` we choose something familiar: a sum of exponentials
``sum_i=0..99 a_i exp(-E_i*x)``. We take as our exact values for the
parameters ``a_i=0.4`` and ``E_i=0.9*(i+1)``, which are easy to remember.
This is simple in Python::

   import numpy as np
   
   def f_exact(x):
       return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))
   
For ``x``\s we take ``1,2,3..10,12,14..20``, and exact ``y``\s are then given by
``f_exact(x)``::

   >>> x = array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
   >>> y_exact = f_exact(x)
   >>> print(y_exact)               # correct/exact values for y
   [  2.74047100e-01   7.92134506e-02   2.88190008e-02 ... ]

Finally we need to add random noise to the ``y_exact``\s to obtain our
fit data. We do this by forming ``y_exact*noise`` where ::

   noise = 1 + sum_n=0..99 c[n]*(x/x_max)**n,
   
Here ``x_max`` is the largest ``x`` used, and the ``c[n]`` are gaussian random 
numbers with means and standard deviations of order ``0.01``. This is easy to
implement in Python using the :mod:`gvar` module::

   import gvar as gv
   
   def make_data():                      # make x,y fit data
       x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
       cr = gv.gvar(0.0,0.01)
       c = [gv.gvar(cr(),0.01) for n in range(100)]
       x_xmax = x/max(x)
       noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
       y = f_exact(x)*noise
       return x,y

Variable ``cr`` represents a gaussian distribution with mean ``0.0`` and width
``0.01``, which we use as a random number generator: ``cr()`` is a number
drawn randomly from the distribution represented by ``cr``::

   >>> print(cr)
   0 +- 0.01
   >>> print(cr())
   0.00452180208286
   >>> print(cr())
   -0.00731564589737

We use ``cr()`` to generate mean values for the gaussian distributions
represented by the ``c[n]``\s, each of which has width ``0.01``. The resulting
``y``\s fluctuate around the corresponding values of ``f_exact(x)`` and have 
statistical errors::

   >>> print(y)
   [0.275179 +- 0.0027439 0.0795054 +- 0.000796125 ... ]
   >>> print(y-f_exact(x))
   [0.00113215 +- 0.0027439 0.000291951 +- 0.000796125 ... ]
   
Different ``y``\s are also correlated (by construction), which becomes clear
if we evaluate the covariance matrix for the ``y``\s::

   >>> print(gv.evalcov(y))
   [[  7.52900382e-06   2.18173029e-06   7.95744444e-07 ... ]
    [  2.18173029e-06   6.33815228e-07   2.31761675e-07 ... ]
    [  7.95744444e-07   2.31761675e-07   8.49651978e-08 ... ]
    ...
   ]

The diagonal elements of the covariance matrix are the variances of the
individual ``y``\s; the off-diagonal elements are a measure of the
correlations ``< (y[i]-<y[i]>) * (y[j]-<y[j]>) >``.

The gaussian deviates ``y[i]`` together with the numbers ``x[i]`` comprise our
fake data.


.. _basic-fits:

Basic Fits
----------
Now that we have fit data, ``x,y = make_data(100)``, we pretend ignorance
of the exact functional relationship between ``x`` and ``y`` (*i.e.*,
``y=f_exact(x)``). Typically we *do* know the functional form and have some
*a priori* idea about the parameter values. The point of the fit is to
improve our knowledge of the parameter values, beyond our *a priori*
impressions, by analyzing the fit data. Here we see how to do this using
the :mod:`lsqfit` module.

First we need code to represent the fit function. In this case we know
that a sum of exponentials is appropriate, so we define the following 
Python function to represent the relationship between ``x`` and ``y`` in 
our fit::

   import numpy as np
   
   def f(x,p):          # function used to fit x,y data
       a = p['a']       # array of a[i]s
       E = p['E']       # array of E[i]s
       return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

The fit parameters, ``a[i]`` and ``E[i]``, are stored in a
dictionary, using labels ``a`` and ``b`` to access them. These parameters
are varied in the fit to find the best-fit values ``p=p_fit`` for which
``f(x,p_fit)`` most closely approximates the ``y``\s in our fit data. The
number of exponentials included in the sum is specified implicitly in this
function, by the lengths of the ``p['a']`` and ``p['E']`` arrays.

Next we need to define priors that encapsulate our *a priori* knowledge 
about the parameter values. In practice we almost always have *a priori* 
knowledge about parameters; it is usually impossible to design a fit
function without some sense of the parameter sizes. Given such knowledge
it is important (usually essential) to include it in the fit. This is 
done by designing priors for the fit, which are probability distributions 
for each parameter that describe the *a priori* uncertainty in that 
parameter. As in the previous section, we use objects of type
:class:`gvar.GVar` to describe (gaussian) probability distributions.
Let's assume that before the fit we suspect that each ``a[i]`` is of order
``0.5+-0.5``, while ``E[i]`` is of order ``1+i+-0.5``. A prior
that represents this information is built using the following code::

   import lsqfit
   import gvar as gv

   def make_prior(nexp):               # make priors for fit parameters
       prior = gv.BufferDict()         # prior -- any dictionary works
       prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
       prior['E'] = [gv.gvar(i+1,0.5) for i in range(nexp)]
       return prior

where ``nexp`` is the number of exponential terms that will be used (and
therefore the number of ``a``\s and ``E``\s). With ``nexp=3``, for example,
one would then have::

   >>> print(prior['a'])
   [0.5 +- 0.5 0.5 +- 0.5 0.5 +- 0.5]
   >>> print(prior['E'])
   [1 +- 0.5 2 +- 0.5 3 +- 0.5]

We use dictionary-like class :class:`gvar.BufferDict` for the prior because it
allows us to save the prior if we wish (using Python's :mod:`pickle` module).
If saving is unnecessary, :class:`gvar.BufferDict` can be replaced by
``dict()`` or most any other Python dictionary class.

With fit data, a fit function, and a prior for the fit parameters, we are 
finally ready to do the fit, which is now easy::

  fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior)
  
So pulling together the entire code, from this section and the previous
one, our complete Python program for making fake data and fitting it is::

   import lsqfit
   import numpy as np
   import gvar as gv

   def f_exact(x):                     # exact f(x)
       return sum(0.4*np.exp(-0.9*(i+1)*x) for i in range(100))

   def f(x,p):                         # function used to fit x,y data
       a = p['a']                      # array of a[i]s
       E = p['E']                      # array of E[i]s
       return sum(ai*np.exp(-Ei*x) for ai,Ei in zip(a,E))

   def make_data():                    # make x,y fit data
       x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
       cr = gv.gvar(0.0,0.01)
       c = [gv.gvar(cr(),0.01) for n in range(100)]
       x_xmax = x/max(x)
       noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
       y = f_exact(x)*noise
       return x,y

   def make_prior(nexp):               # make priors for fit parameters
       prior = gv.BufferDict()         # prior -- any dictionary works
       prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
       prior['E'] = [gv.gvar(i+1,0.5) for i in range(nexp)]
       return prior

   def main():
       gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
       x,y = make_data()               # make fit data
       p0 = None                       # make larger fits go faster (opt.)
       for nexp in range(3,20):
           print('************************************* nexp =',nexp)
           prior = make_prior(nexp)
           fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=prior,p0=p0)
           print(fit)                  # print the fit results
           E = fit.p['E']              # best-fit parameters
           a = fit.p['a']
           print('E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0])
           print('a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0])
           print()
           if fit.chi2/fit.dof<1.:
               p0 = fit.pmean          # starting point for next fit (opt.)

   if __name__ == '__main__':
       main()

We are not sure *a priori* how many exponentials are needed to fit our
data; given that there are only fifteen ``y``\s, and these are noisy, there
may only be information in the data about the first few terms. Consequently
we wrote our code to try fitting with each of ``nexp=3,4,5..19`` terms.
(The pieces of the code involving ``p0`` are optional; they make the
more complicated fits go about 30 times faster since the output from one
fit is used as the starting point for the next fit --- see the discussion
of the ``p0`` parameter for :class:`lsqfit.nonlinear_fit`.) Running
this code produces the following output, which is reproduced here in some
detail in order to illustrate a variety of features::

   ************************************* nexp = 3
   Least Square Fit:
     chi2/dof [dof] = 6.4e+02 [15]    Q = 0    logGBF = -4876    itns = 33

   Parameters:
                 a_   0.0191246 +-  0.00089           (     0.5 +-      0.5)
                  _   0.0237325 +-   0.0011           (     0.5 +-      0.5)
                  _   0.0515777 +-   0.0024           (     0.5 +-      0.5)
                 E_     1.04066 +-   0.0024           (       1 +-      0.5)
                  _     2.06475 +-   0.0024           (       2 +-      0.5)
                  _     3.72957 +-   0.0026           (       3 +-      0.5)

   E1/E0 = 1.98408 +- 0.0024544   E2/E0 = 3.58385 +- 0.00628162
   a1/a0 = 1.24094 +- 0.000263974   a2/a0 = 2.69693 +- 0.00126443

   ************************************* nexp = 4
   Least Square Fit:
     chi2/dof [dof] = 0.57 [15]    Q = 0.9    logGBF = -74.426    itns = 291

   Parameters:
                 a_    0.401753 +-    0.004           (     0.5 +-      0.5)
                  _    0.405533 +-   0.0042           (     0.5 +-      0.5)
                  _     0.49513 +-   0.0072           (     0.5 +-      0.5)
                  _       1.124 +-    0.012           (     0.5 +-      0.5)
                 E_     0.90037 +-  0.00051           (       1 +-      0.5)
                  _     1.80235 +-   0.0012           (       2 +-      0.5)
                  _     2.77306 +-   0.0085           (       3 +-      0.5)
                  _     4.38303 +-     0.02           (       4 +-      0.5)

   E1/E0 = 2.00178 +- 0.00117831   E2/E0 = 3.07991 +- 0.00919665
   a1/a0 = 1.00941 +- 0.00287022   a2/a0 = 1.23242 +- 0.0128117

   ************************************* nexp = 5
   Least Square Fit:
     chi2/dof [dof] = 0.45 [15]    Q = 0.97    logGBF = -73.627    itns = 6

   Parameters:
                 a_    0.401829 +-    0.004           (     0.5 +-      0.5)
                  _    0.404845 +-   0.0044           (     0.5 +-      0.5)
                  _    0.477577 +-    0.026           (     0.5 +-      0.5)
                  _    0.626663 +-     0.28           (     0.5 +-      0.5)
                  _    0.617964 +-     0.35           (     0.5 +-      0.5)
                 E_    0.900363 +-  0.00051           (       1 +-      0.5)
                  _     1.80192 +-   0.0014           (       2 +-      0.5)
                  _     2.75937 +-    0.022           (       3 +-      0.5)
                  _     4.09341 +-     0.26           (       4 +-      0.5)
                  _     4.94923 +-     0.48           (       5 +-      0.5)

   E1/E0 = 2.00132 +- 0.00139785   E2/E0 = 3.06473 +- 0.0238493
   a1/a0 = 1.0075 +- 0.00413287   a2/a0 = 1.18851 +- 0.0629341

   ************************************* nexp = 6
   Least Square Fit:
     chi2/dof [dof] = 0.45 [15]    Q = 0.97    logGBF = -73.771    itns = 6

   Parameters:
                 a_    0.401835 +-    0.004           (     0.5 +-      0.5)
                  _    0.404032 +-   0.0047           (     0.5 +-      0.5)
                  _    0.460419 +-    0.041           (     0.5 +-      0.5)
                  _    0.598159 +-     0.24           (     0.5 +-      0.5)
                  _    0.471462 +-     0.37           (     0.5 +-      0.5)
                  _    0.451949 +-     0.46           (     0.5 +-      0.5)
                 E_    0.900353 +-  0.00051           (       1 +-      0.5)
                  _     1.80145 +-   0.0017           (       2 +-      0.5)
                  _     2.74537 +-    0.034           (       3 +-      0.5)
                  _     3.97765 +-     0.32           (       4 +-      0.5)
                  _     4.95873 +-     0.49           (       5 +-      0.5)
                  _     6.00919 +-      0.5           (       6 +-      0.5)

   E1/E0 = 2.00083 +- 0.00166713   E2/E0 = 3.04921 +- 0.0372569
   a1/a0 = 1.00547 +- 0.00554293   a2/a0 = 1.14579 +- 0.101026

   ************************************* nexp = 7
   Least Square Fit:
     chi2/dof [dof] = 0.45 [15]    Q = 0.96    logGBF = -73.873    itns = 6

   Parameters:
                 a_    0.401835 +-    0.004           (     0.5 +-      0.5)
                  _    0.403622 +-   0.0048           (     0.5 +-      0.5)
                  _    0.452267 +-    0.047           (     0.5 +-      0.5)
                  _    0.598425 +-     0.22           (     0.5 +-      0.5)
                  _    0.416291 +-     0.37           (     0.5 +-      0.5)
                  _    0.417308 +-     0.46           (     0.5 +-      0.5)
                  _    0.459911 +-     0.49           (     0.5 +-      0.5)
                 E_    0.900348 +-  0.00051           (       1 +-      0.5)
                  _     1.80122 +-   0.0018           (       2 +-      0.5)
                  _     2.73849 +-    0.039           (       3 +-      0.5)
                  _     3.93758 +-     0.33           (       4 +-      0.5)
                  _     4.96349 +-     0.49           (       5 +-      0.5)
                  _     6.01884 +-      0.5           (       6 +-      0.5)
                  _     7.01563 +-      0.5           (       7 +-      0.5)

   E1/E0 = 2.00058 +- 0.00179764   E2/E0 = 3.04159 +- 0.0430577
   a1/a0 = 1.00445 +- 0.00620982   a2/a0 = 1.1255 +- 0.116229
                                        .
                                        .
                                        .
                                        
    ************************************* nexp = 19
    Least Square Fit:
      chi2/dof [dof] = 0.46 [15]    Q = 0.96    logGBF = -73.951    itns = 1

    Parameters:
                  a_    0.401835 +-    0.004           (     0.5 +-      0.5)
                   _    0.403323 +-   0.0049           (     0.5 +-      0.5)
                   _    0.446511 +-    0.051           (     0.5 +-      0.5)
                   _    0.600997 +-     0.21           (     0.5 +-      0.5)
                   _    0.380338 +-     0.37           (     0.5 +-      0.5)
                   _    0.395013 +-     0.46           (     0.5 +-      0.5)
                   _    0.450063 +-     0.49           (     0.5 +-      0.5)
                   _    0.479737 +-      0.5           (     0.5 +-      0.5)
                   _     0.49226 +-      0.5           (     0.5 +-      0.5)
                   _    0.497112 +-      0.5           (     0.5 +-      0.5)
                   _    0.498932 +-      0.5           (     0.5 +-      0.5)
                   _    0.499606 +-      0.5           (     0.5 +-      0.5)
                   _    0.499855 +-      0.5           (     0.5 +-      0.5)
                   _    0.499947 +-      0.5           (     0.5 +-      0.5)
                   _     0.49998 +-      0.5           (     0.5 +-      0.5)
                   _    0.499993 +-      0.5           (     0.5 +-      0.5)
                   _    0.499997 +-      0.5           (     0.5 +-      0.5)
                   _    0.499999 +-      0.5           (     0.5 +-      0.5)
                   _         0.5 +-      0.5           (     0.5 +-      0.5)
                  E_    0.900345 +-  0.00051           (       1 +-      0.5)
                   _     1.80105 +-   0.0019           (       2 +-      0.5)
                   _     2.73354 +-    0.042           (       3 +-      0.5)
                   _     3.91278 +-     0.33           (       4 +-      0.5)
                   _     4.96687 +-     0.49           (       5 +-      0.5)
                   _     6.02418 +-      0.5           (       6 +-      0.5)
                   _     7.01928 +-      0.5           (       7 +-      0.5)
                   _     8.00922 +-      0.5           (       8 +-      0.5)
                   _     9.00374 +-      0.5           (       9 +-      0.5)
                   _     10.0014 +-      0.5           (      10 +-      0.5)
                   _     11.0005 +-      0.5           (      11 +-      0.5)
                   _     12.0002 +-      0.5           (      12 +-      0.5)
                   _     13.0001 +-      0.5           (      13 +-      0.5)
                   _          14 +-      0.5           (      14 +-      0.5)
                   _          15 +-      0.5           (      15 +-      0.5)
                   _          16 +-      0.5           (      16 +-      0.5)
                   _          17 +-      0.5           (      17 +-      0.5)
                   _          18 +-      0.5           (      18 +-      0.5)
                   _          19 +-      0.5           (      19 +-      0.5)

    E1/E0 = 2.0004 +- 0.0018858   E2/E0 = 3.0361 +- 0.0466706
    a1/a0 = 1.0037 +- 0.00663103   a2/a0 = 1.11118 +- 0.125291
   
There are several things to notice here:

   * Clearly three exponentials (``nexp=3``) is not enough. The ``chi**2`` 
     per degree of freedom (``chi2/dof``) is much larger than one. The
     ``chi**2`` improves significantly for ``nexp=4`` exponentials and by
     ``nexp=6`` the fit is as good as it is going to get --- there is
     essentially no change when further exponentials are added.
   
   * The best-fit values for each parameter are listed for each of the
     fits, together with the prior values (in parentheses, on the right).
     Values for each ``a[i]`` and ``E[i]`` are listed in order, starting at
     the points indicated.
     
     Once the fit converges, the best-fit values for the various parameters
     agree well --- that is to within their errors, approximately --- with
     the exact values, which we know since we are using fake data. For
     example, ``a`` and ``E`` for the first exponential are ``0.402(4)``
     and ``0.9003(5)``, respectively, from the fit where the exact answers
     are ``0.4`` and ``0.9``; and we get ``0.45(5)`` and ``2.73(4)`` for
     the third exponential where the exact values are ``0.4`` and ``2.7``.
     
   * Note in the ``nexp=7`` fit how the means and standard deviations for
     the parameters governing the seventh (and last) exponential are almost
     identical to the values in the corresponding priors: ``0.46(49)`` from
     the fit for ``a`` and ``7.0(5)`` for ``E``. This tells us that our fit
     data has little or no information to add to what we knew *a priori*
     about these parameters --- there isn't enough data and what we have
     isn't accurate enough. 
     
     This situation is truer still of further terms as they are added in
     the ``nexp=8`` and later fits. This is why the fit results stop
     changing once we have ``nexp=6`` exponentials. There is no point in
     including further exponentials, beyond the need to verify that the fit
     has indeed converged.
     
   * The last fit includes ``nexp=19`` exponentials and therefore has 38
     parameters. This is in a fit to 15 ``y``\s. Old-fashioned fits, without
     priors, are impossible when the number of parameters exceeds the number
     of data points. That is clearly not the case here, where the number of
     terms and parameters can be made arbitrarily large, eventually (after
     ``nexp=6`` terms) with no effect at all on the results.
     
     The reason is that the prior that we include for each new parameter
     is, in effect, a new piece of data (the mean and standard deviation of
     the *a priori* expectation for that parameter); it leads to a new term
     in the ``chi**2`` function. We are fitting both the data and our *a
     priori* expectations for the parameters. So in the ``nexp=19`` fit,
     for example, we actually have 53 pieces of data to fit: the 15 ``y``\s
     plus the 38 prior values for the 38 parameters.
     
     The effective number of degrees of freedom (``dof`` in the output
     above) is the number of pieces of data minus the number of fit
     parameters, or 53-38=15 in this last case. With priors for every
     parameter, the number of degrees of freedom is always equal to the
     number of ``y``\s, irrespective of how many fit parameters there are.
     
   * The Gaussian Bayes Factor (or *posterior probability*, whose logarithm is 
     ``logGBF`` in the output) is a measure of the likelihood that the actual
     data being fit could have come from a theory with the prior used in the
     fit. The larger this number, the more likely it is that prior and data
     could be related. Here it grows dramatically from the first fit
     (``nexp=3``) but then more-or-less stops changing around ``nexp=6``. The
     implication is that this data is much more likely to have come from a
     theory with ``nexp>=6`` than with ``nexp=3`` (which we know to be the
     actual case).
     
   * In the code, results for each fit are captured in a Python object
     ``fit``, which is of type :class:`lsqfit.nonlinear_fit`. A summary of the
     fit information is obtained by printing ``fit``. Also the best-fit
     results for each fit parameter can be accessed through ``fit.p``, as is
     done here to calculate various ratios of parameters.
     
     The errors in these last calculations automatically account for any
     correlations in the statistical errors for different parameters. This
     is obvious in the ratio ``a1/a0``, which would be ``1.004(16)`` if
     there was no statistical correlation between our estimates for ``a1``
     and ``a0``, but in fact is ``1.004(7)`` in this fit.
      
Finally we inspect the fit's quality point by point. The input data are
compared with results from the fit function, evaluated with the best-fit
parameters, in the following table (obtained in the code by printing the
output from ``fit.format(15)``\)::

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.27518     0.27521   0.0027439
              2    0.079505    0.079521  0.00079613
              3    0.028911    0.028921  0.00029149
              4    0.011266    0.011272  0.00011468
              5   0.0045023   0.0045063  4.6409e-05
              6   0.0018171   0.0018194  1.9025e-05
              7  0.00073619  0.00073746  7.8556e-06
              8  0.00029873   0.0002994  3.2608e-06
              9  0.00012129  0.00012163    1.36e-06
             10  4.9257e-05  4.9426e-05  5.7008e-07
             12  8.1264e-06  8.1636e-06    1.02e-07
             14  1.3415e-06  1.3485e-06  1.8887e-08
             16  2.2171e-07  2.2275e-07  3.7159e-09
             18  3.6605e-08  3.6794e-08   8.455e-10
             20  6.2447e-09  6.0779e-09   6.092e-10

This information is presented again in the following plot, which shows the
ratio ``y/f(x,p)``, as a function of ``x``, using the best-fit parameters
``p``. The correct result for this ratio, of course, is one. The smooth
variation in the data --- smooth compared with the size of the
statistical-error bars --- is an indication of the statistical correlations
between individual ``y``\s.

.. image:: fig1.*
   :width: 80%

This particular plot was made using the :mod:`matplotlib` module, with the 
following code added to the end of ``main()`` (outside the loop)::

      import pylab as plt   
      ratio = y/f(x,fit.pmean)
      plt.xlim(0,21)
      plt.xlabel('x')
      plt.ylabel('y/f(x,p)')
      plt.errorbar(x=x,y=gv.mean(ratio),yerr=gv.sdev(ratio),fmt='ob')
      plt.plot([0.0,21.0],[1.0,1.0])
      plt.show()


``x`` has Error Bars
--------------------
We now consider variations on our basic fit analysis (described above). The 
first variation concerns what to do when the independent variables, the 
``x``\s, have errors, as well as the ``y``\s. This is easily handled by 
turning the ``x``\s into fit parameters, and otherwise dispensing 
with independent variables.

To illustrate this, we modify the basic analysis code in the previous 
section. First we need to add errors to the ``x``\s, which we do by 
changing ``make_data`` so that each ``x`` has a random value within about 
``+-0.001%`` of its original value and an error::

   def make_data():                    # make x,y fit data
       x = np.array([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.,14.,16.,18.,20.])
       cr = gv.gvar(0.0,0.01)
       c = [gv.gvar(cr(),0.01) for n in range(100)]
       x_xmax = x/max(x)
       noise = 1+ sum(c[n]*x_xmax**n for n in range(100))
       y = f_exact(x)*noise            # noisy y[i]s
       xfac = gv.gvar(1.0,0.00001)     # gaussian distrib'n: 1 +- 0.001%
       x = np.array([xi*gv.gvar(xfac(),xfac.sdev) for xi in x]) # noisy x[i]s
       return x,y
   
Here :class:`gvar.GVar` object ``xfac`` is used as a random number
generator: each time it is called, ``xfac()`` is a different random number
from the distribution with mean ``xfac.mean`` and standard deviation
``xfac.sdev`` (that is, ``1+-0.00001``). The main program is modified so
that the (now random) ``x`` array is treated as a fit parameter. The prior
for each ``x`` is, obviously, specified by the mean and standard deviation
of that ``x``, which is read directly out of the array of ``x``\s produced 
by ``make_data()``::

   def make_prior(nexp,x):             # make priors for fit parameters
       prior = gv.BufferDict()         # prior -- any dictionary works
       prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
       prior['E'] = [gv.gvar(i+1,0.5) for i in range(nexp)]
       prior['x'] = x                  # x now an array of parameters
                                       # replace x by None in fit data
       return prior

   def main():
       gv.ranseed([2009,2010,2011,2012]) # initialize random numbers (opt.)
       x,y = make_data()               # make fit data
       p0 = None                       # make larger fits go faster (opt.)
       for nexp in range(3,20):
           print('************************************* nexp =',nexp)
           prior = make_prior(nexp,x)
           fit = lsqfit.nonlinear_fit(data=(None,y),fcn=f,prior=prior,p0=p0)
           print(fit)                  # print the fit results
           E = fit.p['E']              # best-fit parameters
           a = fit.p['a']
           print('E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0])
           print('a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0])
           print()
           if fit.chi2/fit.dof<1.:
               p0 = fit.pmean          # starting point for next fit (opt.)
   
Note that ``x`` has been replaced in the fit data by the Python null
variable ``None``. This underscores the fact that
:class:`lsqfit.nonlinear_fit` is completely uninterested in the independent
variable ``x`` in the fit data. It makes no use of it beyond passing it
through to the fit function. This means that the independent variable ``x``
in the fit data can be replaced by any collection of data, using any data
type that is desired; it is often a convenient way to send data to the
fit function that is neither a ``y`` nor a parameter.

The final code modification is to the fit function, which now ignores its
first argument (formerly ``x``), but gets ``x`` values from the parameters
``p`` instead::

   def f(xdummy,p):
       a = p['a']
       E = p['E']
       x = p['x']
       return sum(ai*exp(-Ei*x) for ai,Ei in zip(a,E))

Running the new code gives, for ``nexp=6`` terms::

   ************************************* nexp = 6
   Least Square Fit:
     chi2/dof [dof] = 0.54 [15]    Q = 0.92    logGBF = -95.553    itns = 6

   Parameters:
                 a_    0.402497 +-   0.0041           (     0.5 +-      0.5)
                  _    0.428721 +-    0.032           (     0.5 +-      0.5)
                  _    0.583018 +-     0.23           (     0.5 +-      0.5)
                  _     0.40374 +-     0.38           (     0.5 +-      0.5)
                  _    0.421848 +-     0.46           (     0.5 +-      0.5)
                  _    0.463996 +-     0.49           (     0.5 +-      0.5)
                 E_    0.900682 +-   0.0006           (       1 +-      0.5)
                  _     1.81758 +-     0.02           (       2 +-      0.5)
                  _      2.9487 +-     0.28           (       3 +-      0.5)
                  _     3.97546 +-     0.49           (       4 +-      0.5)
                  _     5.02085 +-      0.5           (       5 +-      0.5)
                  _     6.01467 +-      0.5           (       6 +-      0.5)
                 x_    0.999997 +-    1e-05           (       1 +-    1e-05)
                  _     1.99996 +-    2e-05           (       2 +-    2e-05)
                  _     3.00001 +-    3e-05           (       3 +-    3e-05)
                  _     4.00006 +-  3.6e-05           (       4 +-    4e-05)
                  _     5.00005 +-  3.4e-05           (       5 +-    5e-05)
                  _     6.00002 +-  3.9e-05           (       6 +-    6e-05)
                  _     6.99999 +-    4e-05           (       7 +-    7e-05)
                  _     7.99996 +-  4.2e-05           (       8 +-    8e-05)
                  _     8.99993 +-    5e-05           (       9 +-    9e-05)
                  _     9.99992 +-  5.9e-05           (      10 +-   0.0001)
                  _     11.9999 +-  7.9e-05           (      12 +-  0.00012)
                  _     13.9999 +-  0.00011           (      14 +-  0.00014)
                  _     15.9999 +-  0.00015           (      16 +-  0.00016)
                  _     18.0002 +-  0.00018           (      18 +-  0.00018)
                  _     20.0002 +-   0.0002           (      20 +-   0.0002)

   E1/E0 = 2.01801 +- 0.0219085   E2/E0 = 3.27385 +- 0.307128
   a1/a0 = 1.06515 +- 0.0772791   a2/a0 = 1.4485 +- 0.574717

This looks quite a bit like what we obtained before, except that now there 
are 15 more parameters, one for each ``x``, and also now all results are
a good deal less accurate. Note that one result from this analysis is new 
values for the ``x``\s. In some cases the errors on the ``x`` values have
been reduced --- by information in the fit data.


.. _correlated-parameters:

Correlated Parameters; Gaussian Bayes Factor
---------------------------------------------
:class:`gvar.GVar` objects are very useful for handling more complicated
priors, including situations where we know *a priori* of correlations 
between parameters. Returning to the :ref:`basic-fits` example above, 
imagine a situation where we still have a ``+-0.5`` uncertainty about the
value of any individual ``E[i]``, but we know *a priori* that the 
separations between adjacent ``E[i]``\s is ``0.9+-0.01``. We want to 
build the correlation between adjacent ``E[i]``\s into our prior.

We do this by introducing a :class:`gvar.GVar` object ``de[i]`` for each
separate difference ``E[i]-E[i-1]``, with ``de[0]`` being ``E[0]``::

   de = [gvar(0.9,0.01) for i in range(nexp)]
   de[0] = gvar(1,0.5)     #  different distribution for E[0]
   
Then ``de[0]`` specifies the probability distribution for ``E[0]``,
``de[0]+de[1]`` the distribution for ``E[1]``, ``de[0]+de[1]+de[2]`` the
distribution for ``E[2]``, and so on. This can be implemented (slightly 
inefficiently) in a single line of Python::

   E = [sum(de[:i+1]) for i in range(nexp)]
   
For ``nexp=3``, this implies that ::

   >>> print(E)
   [1 +- 0.5 1.9 +- 0.5001 2.8 +- 0.5002]
   >>> print(E[1]-E[0],E[2]-E[1])
   0.9 +- 0.01 0.9 +- 0.01

which shows that each ``E[i]`` separately has an uncertainty of ``+-0.5`` 
(approximately) but that differences are specified to within ``+-0.01``.

In the code, we need only change the definition of the prior in order to
introduce these correlations::

   def make_prior(nexp):               # make priors for fit parameters
       prior = gv.BufferDict()         # prior -- any dictionary works
       prior['a'] = [gv.gvar(0.5,0.5) for i in range(nexp)]
       de = [gv.gvar(0.9,0.01) for i in range(nexp)]
       de[0] = gv.gvar(1,0.5)     
       prior['E'] = [sum(de[:i+1]) for i in range(nexp)]
       return prior
   
Running the code as before, but now with the correlated prior in place, we
obtain the following fit with ``nexp=7`` terms::
   
   ************************************* nexp = 7
   Least Square Fit:
     chi2/dof [dof] = 0.44 [15]    Q = 0.97    logGBF = -66.989    itns = 3

   Parameters:
                 a_    0.401798 +-    0.004           (     0.5 +-      0.5)
                  _    0.401633 +-   0.0041           (     0.5 +-      0.5)
                  _    0.403819 +-    0.012           (     0.5 +-      0.5)
                  _    0.394153 +-    0.045           (     0.5 +-      0.5)
                  _    0.398183 +-     0.15           (     0.5 +-      0.5)
                  _    0.504394 +-     0.31           (     0.5 +-      0.5)
                  _    0.515886 +-     0.42           (     0.5 +-      0.5)
                 E_    0.900318 +-  0.00051           (       1 +-      0.5)
                  _     1.80009 +-   0.0011           (     1.9 +-      0.5)
                  _     2.70085 +-     0.01           (     2.8 +-      0.5)
                  _      3.6008 +-    0.014           (     3.7 +-      0.5)
                  _     4.50084 +-    0.017           (     4.6 +-      0.5)
                  _     5.40084 +-     0.02           (     5.5 +-      0.5)
                  _     6.30084 +-    0.022           (     6.4 +-      0.5)

   E1/E0 = 1.9994 +- 0.0010494   E2/E0 = 2.99988 +- 0.0110833
   a1/a0 = 0.999589 +- 0.00250023   a2/a0 = 1.00503 +- 0.0279927
   
The results are similar to before for the leading parameters, but
substantially more accurate for parameters describing the second and later
exponential terms, as might be expected given our enhanced knowledge about
the differences between ``E[i]``\s. The output energy differences are
particularly accurate: they range from ``E[1]-E[0] = 0.900(1)``, which is
ten times more precise than the prior, to ``E[6]-E[5] = 0.900(10)``, which
is just what was put into the fit through the prior (the fit data adds no
new information). The correlated prior allows us to merge our *a priori*
information about the energy differences with the new information carried
by the fit data ``x,y``.

Note that the Gaussian Bayes Factor (see ``logGBF`` in the output) is
significantly larger with the correlated prior (``logGBF = -67.0``) than it
was for the uncorrelated prior (``logGBF = -73.9``). If one had been
uncertain as to which prior was more appropriate, this difference says that
the data prefers the correlated prior. (More precisely, it says that we
would be significantly more likely to get this data from a theory with the
correlated prior than from one with the uncorrelated prior.) This
difference is significant despite the fact that the ``chi**2``\s in the two
cases are almost the same.


Tuning Priors and the Empirical Bayes Criterion
------------------------------------------------
Given two choices of prior for a parameter, the one that results in a larger
Gaussian Bayes Factor after fitting (see ``logGBF`` in fit output or
``fit.logGBF``) is the one preferred by the data. We can use this fact to tune
a prior or set of priors in situations where we are uncertain about the
correct *a priori* value: we vary the widths and/or central values of the
priors of interest to maximize ``logGBF``. This leads to complete nonsense if
it is applied to all the priors, but it is useful for tuning (or testing)
limited subsets of the priors when other information is unavailable. In effect
we are using the data to get a feel for what is a reasonable prior.

This method is implemented in a driver program ::
    
    fit,z = lsqfit.empbayes_fit(z0,fitargs)
    
which varies :mod:`numpy` array ``z``, starting at ``z0``, to maximize
``fit.logGBF`` where ::

    fit = lsqfit.nonlinear_fit(**fitargs(z)). 
    
Function ``fitargs(z)`` returns a dictionary containing the arguments for
:func:`nonlinear_fit`. These arguments, and the prior in particular, are
varied as some function of ``z``. The optimal fit (that is, the one for which
``fit.logGBF`` is maximum) and ``z`` are returned.
    
To illustrate, consider tuning the widths of the priors for the amplitudes,
``prior['a']``, in the example from the previous section. This is done by
adding the following code to the end of ``main()`` subroutine::

   def fitargs(z,nexp=nexp,prior=prior,f=f,data=(x,y),p0=p0):
       z = np.exp(z)
       prior['a'] = [gv.gvar(0.5,0.5*z[0]) for i in range(nexp)]
       return dict(prior=prior,data=data,fcn=f,p0=p0)
   ##
   z0 = [0.0]
   fit,z = empbayes_fit(z0,fitargs,tol=1e-3)
   print(fit)                  # print the optimized fit results
   E = fit.p['E']              # best-fit parameters
   a = fit.p['a']
   print('E1/E0 =',E[1]/E[0],'  E2/E0 =',E[2]/E[0])
   print('a1/a0 =',a[1]/a[0],'  a2/a0 =',a[2]/a[0])
   print("prior['a'] =",fit.prior['a'][0])
   print()

Function ``fitargs`` generates a dictionary containing the arguments for
:class:`lsqfit.nonlinear_fit`. These are identical to what we have been using
except that the width of the priors in ``prior['a']`` is adjusted according
to parameter ``z``. Function :func:`lsqfit.empbayes_fit` does fits for 
different values of ``z`` and selects the ``z`` that maximizes ``fit.logGBF``.
It returns the corresponding fit and the value of ``z``.

This code generates the following output when ``nexp=7``::

   Least Square Fit:
     chi2/dof [dof] = 0.77 [15]    Q = 0.71    logGBF = -60.457    itns = 1

   Parameters:
                 a_    0.402651 +-    0.004           (     0.5 +-    0.095)
                  _    0.402469 +-   0.0041           (     0.5 +-    0.095)
                  _    0.407096 +-   0.0079           (     0.5 +-    0.095)
                  _    0.385447 +-     0.02           (     0.5 +-    0.095)
                  _    0.430817 +-    0.058           (     0.5 +-    0.095)
                  _     0.47765 +-    0.074           (     0.5 +-    0.095)
                  _    0.493185 +-    0.089           (     0.5 +-    0.095)
                 E_    0.900307 +-   0.0005           (       1 +-      0.5)
                  _     1.80002 +-    0.001           (     1.9 +-      0.5)
                  _     2.70233 +-   0.0085           (     2.8 +-      0.5)
                  _     3.60274 +-    0.013           (     3.7 +-      0.5)
                  _      4.5033 +-    0.017           (     4.6 +-      0.5)
                  _     5.40351 +-    0.019           (     5.5 +-      0.5)
                  _     6.30355 +-    0.022           (     6.4 +-      0.5)

   E1/E0 = 1.99934 +- 0.00100622   E2/E0 = 3.00156 +- 0.00926136
   a1/a0 = 0.999549 +- 0.00245793   a2/a0 = 1.01104 +- 0.0165249
   prior['a'] = 0.5 +- 0.0950546

Reducing the width of the ``prior['a']``\s from ``0.5`` to ``0.1`` increased
``logGBF`` from ``-67.0`` to ``-60.5``. The error for ``a2/a0`` is 40%
smaller, but the other results are not much affected --- suggesting that the
details of ``prior['a']`` are not all that important, which is confirmed by
the error budgets generated in the next section. It is not surprising, of
course, that the optimal width is ``0.1`` since the mean values for the
``fit.p['a']``\s are clustered around ``0.4``, which is ``0.1`` below the mean
value of the priors ``prior['a']``.


Partial Errors and Error Budgets
---------------------------------
We frequently want to know how much of the uncertainty in a fit result is
due to a particular input uncertainty or subset of input uncertainties
(from the input data and/or from the priors). We refer to such errors as
"partial errors" (or partial standard deviations) since each is only part
of the total uncertainty in the fit result. The collection of such partial
errors, each associated with a different input error, is called an "error
budget" for the fit result. The partial errors from all sources of input
error reproduce the total fit error when they are added in quadrature.

Given the ``fit`` object (an :class:`lsqfit.nonlinear_fit` object) from the
example in the section on :ref:`correlated-parameters`, for example, we can
extract such information using :meth:`gvar.GVar.partialsdev` --- for example::

   >>> E = fit.p['E']
   >>> a = fit.p['a']
   >>> print(E[1]/E[0])
   1.9994 +- 0.0010494
   >>> print((E[1]/E[0]).partialsdev(fit.prior['E']))
   0.000414032342911
   >>> print((E[1]/E[0]).partialsdev(fit.prior['a']))
   0.000142408815921
   >>> print((E[1]/E[0]).partialsdev(y))
   0.000953694015457
   
This shows that the total uncertainty in ``E[1]/E[0]``, ``0.00105``, is 
the sum in quadrature of a contribution ``0.00041`` due to the priors 
specified by ``prior['E']``, ``0.00014`` due to ``prior['a']``, and 
``0.00095`` from the statistical errors in the input data ``y``.

There are two utility functions for tabulating results and error budgets.
They require dictionaries of output results and inputs, and use the 
keys from the dictionaries to label columns and rows, respectively, in
an error-budget table::

   outputs = {'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],         
            'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]}
   inputs = {'E':fit.prior['E'],'a':fit.prior['a'],'y':y}
   print(fit.fmt_values(outputs))
   print(fit.fmt_errorbudget(outputs,inputs))

This gives the following output::

   Values:
                 E2/E0: 3.000(11)           
                 E1/E0: 1.999(1)            
                 a2/a0: 1.005(28)           
                 a1/a0: 1.000(3)            

   Partial % Errors:
                            E2/E0     E1/E0     a2/a0     a1/a0
   ------------------------------------------------------------
                     a:      0.09      0.01      1.07      0.02
                     y:      0.07      0.05      0.78      0.19
                     E:      0.35      0.02      2.45      0.16
   ------------------------------------------------------------
                 total:      0.37      0.05      2.79      0.25
   
This table suggests, for example, that reducing the statistical errors in
the input ``y`` data would significantly reduce the final errors in
``E1/E0`` and ``a1/a0``, but would have only a slight impact on errors in
``E2/E0`` and ``a2/a0``. In fact a four-fold reduction in the ``y`` errors
reduces the ``E1/E0`` error to 0.02% (from 0.05%) while leaving the
``E2/E0`` error at 0.36%.


``y`` has No Error Bars
-----------------------
Occasionally there are fit problems where values for the dependent
variable ``y`` are known exactly (to machine precision). This poses a 
problem for least-squares fitting since the ``chi**2`` function is 
infinite when standard deviations are zero. How does one assign errors 
to exact ``y``\s in order to define a ``chi**2`` function that can be 
usefully minimized?

It is almost always the case in physical applications of this sort that the
fit function has in principle an infinite number of parameters. It is, of
course, impossible to extract information about infinitely many parameters
from a finite number of ``y``\s. In practice, however, we generally care about
only a few of the parameters in the fit function. (If this isn't the case,
give up.) The goal for a least-squares fit is to figure out what a finite
number of exact ``y``\s can tell us about the parameters we want to know.

The key idea here is to use priors to model the part of the fit function 
that we don't care about, and to remove that part of the function from 
the analysis by subtracting or dividing it out from the input data. To
illustrate, consider again the example described in the section on
:ref:`correlated-parameters`. Let us imagine that we know the exact values
for ``y`` for each of ``x=1, 1.2, 1.4...2.6, 2.8``. We are fitting this
data with a sum of exponentials ``a[i]*exp(-E[i]*x)`` where now we will
assume that *a priori* we know that: ``E[0]=1.0(5)``,
``E[i+1]-E[i]=0.9(2)``, and ``a[i]=0.5(5)``. Suppose that our goal is to
find good estimates for ``E[0]`` and ``a[0]``.

We know that for some set of parameters ::

   y = sum_i=0..inf  a[i]*exp(-E[i]*x)
   
for each ``x``\-\ ``y`` pair in our fit data. Given that  
``a[0]`` and ``E[0]`` are all we want to know, we might imagine defining
a new, modified dependent variable ``ymod``, equal to just
``a[0]*exp(-E[0]*x)``::

   ymod = y - sum_i=1..inf a[i]*exp(-E[i]*x)
   
We know everything on the right-hand side of this equation: we have exact
values for ``y`` and we have *a priori* estimates for the ``a[i]`` and
``E[i]`` with ``i>0``. So given means and standard deviations for every
``i>0`` parameter, and the exact ``y``, we can in principle determine a
mean and standard deviation for ``ymod``. The strategy then is to compute
the corresponding ``ymod`` for every ``y`` and ``x`` pair, and then fit
``ymod`` versus ``x`` to the *single* exponential ``a[0]*exp(-E[0]*t)``.
That fit will give values for ``a[0]`` and ``E[0]`` that reflect the
uncertainties in ``ymod``, which in turn originate in uncertainties in our
knowledge about the parameters for the ``i>0`` exponentials. 

It turns out to be quite simple to implement such a strategy using
:class:`gvar.GVar`\s. We convert our code by first modifying the main
program so that it provides prior information to a subroutine that computes
``ymod``. We will vary the number of terms ``nexp`` that are kept in the
fit, putting the rest into ``ymod`` as above (up to a maximum of ``20``
terms, which is close enough to infinity)::

   def main():
       gv.ranseed([2009,2010,2011,2012])  # initialize random numbers (opt.)
       max_prior = make_prior(20)         # maximum sized prior
       p0 = None                          # make larger fits go faster (opt.)
       for nexp in range(1,7):
           print('************************************* nexp =',nexp)
           fit_prior = gv.BufferDict()    # part of max_pior used in fit
           ymod_prior = gv.BufferDict()   # part of max_prior absorbed in ymod
           for k in max_prior:
               fit_prior[k] = max_prior[k][:nexp]
               ymod_prior[k] = max_prior[k][nexp:]
           x,y = make_data(ymod_prior)    # make fit data
           fit = lsqfit.nonlinear_fit(data=(x,y),fcn=f,prior=fit_prior,p0=p0)
           print(fit.format(10))          # print the fit results
           print()
           if fit.chi2/fit.dof<1.:
               p0 = fit.pmean             # starting point for next fit (opt.)

We put all of our *a priori* knowledge about parameters into prior
``max_prior`` and then pull out the part we need for the fit --- that is,
the first ``nexp`` terms. The remaining part of ``max_prior`` is used to
correct the exact data, which comes from a new ``make_data``::

   def make_data(ymod_prior):          # make x,y fit data
       x = np.arange(1.,10*0.2+1.,0.2)
       ymod = f_exact(x)-f(x,ymod_prior)        
       return x,ymod
   
Running the new code produces the following output, where again ``nexp`` is
the number of exponentials kept in the fit (and ``20-nexp`` is the number
pushed into the modified dependent variable ``ymod``)::

   ************************************* nexp = 1
   Least Square Fit (y correlated with prior):
     chi2/dof [dof] = 0.056 [10]    Q = 1    logGBF = -16.24    itns = 5

   Parameters:
                 a_    0.400845 +-  0.00094           (     0.5 +-      0.5)
                 E_    0.900324 +-   0.0004           (       1 +-      0.5)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.14803     0.16292     0.10692
            1.2     0.12825     0.13607    0.074202
            1.4     0.10957     0.11365    0.051975
            1.6    0.092853    0.094922    0.036625
            1.8    0.078298     0.07928     0.02591
              2    0.065813    0.066216    0.018378
            2.2      0.0552    0.055305    0.013057
            2.4    0.046231    0.046191   0.0092867
            2.6     0.03868     0.03858   0.0066089
            2.8    0.032339    0.032223   0.0047043


   ************************************* nexp = 2
   Least Square Fit (y correlated with prior):
     chi2/dof [dof] = 0.056 [10]    Q = 1    logGBF = -35.133    itns = 4

   Parameters:
                 a_    0.399968 +-  0.00079           (     0.5 +-      0.5)
                  _    0.400415 +-    0.026           (     0.5 +-      0.5)
                 E_    0.899986 +-  0.00031           (       1 +-      0.5)
                  _     1.79983 +-     0.02           (     1.9 +-     0.54)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.22281     0.22882    0.044661
            1.2     0.17939     0.18202    0.025977
            1.4     0.14454     0.14568    0.015244
            1.6     0.11677     0.11725    0.008997
            1.8    0.094655    0.094842   0.0053294
              2    0.076998    0.077061   0.0031644
            2.2    0.062849    0.062861   0.0018817
            2.4    0.051462    0.051455   0.0011199
            2.6    0.042257    0.042246  0.00066679
            2.8    0.034786    0.034776  0.00039704


   ************************************* nexp = 3
   Least Square Fit (y correlated with prior):
     chi2/dof [dof] = 0.058 [10]    Q = 1    logGBF = -50.219    itns = 4

   Parameters:
                 a_    0.399938 +-  0.00082           (     0.5 +-      0.5)
                  _    0.398106 +-    0.034           (     0.5 +-      0.5)
                  _    0.401049 +-    0.098           (     0.5 +-      0.5)
                 E_    0.899975 +-  0.00032           (       1 +-      0.5)
                  _     1.79848 +-    0.024           (     1.9 +-     0.54)
                  _     2.69343 +-      0.2           (     2.8 +-     0.57)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.25322     0.25564     0.01863
            1.2     0.19676     0.19765   0.0090783
            1.4     0.15446     0.15478   0.0044619
            1.6     0.12244     0.12255   0.0022047
            1.8    0.097892     0.09793   0.0010932
              2    0.078847    0.078859  0.00054319
            2.2    0.063905    0.063908  0.00027026
            2.4    0.052065    0.052065  0.00013456
            2.6    0.042602    0.042601   6.701e-05
            2.8    0.034983    0.034982   3.337e-05


   ************************************* nexp = 4
   Least Square Fit (input data correlated with prior):
     chi2/dof [dof] = 0.057 [10]    Q = 1    logGBF = -67.447    itns = 5

   Parameters:
                 a_    0.399937 +-  0.00077           (     0.5 +-      0.5)
                  _    0.398315 +-    0.032           (     0.5 +-      0.5)
                  _    0.401742 +-      0.1           (     0.5 +-      0.5)
                  _    0.403269 +-     0.15           (     0.5 +-      0.5)
                 E_    0.899975 +-   0.0003           (       1 +-      0.5)
                  _     1.79859 +-    0.023           (     1.9 +-     0.54)
                  _     2.69522 +-     0.19           (     2.8 +-     0.57)
                  _     3.60827 +-     0.28           (     3.7 +-     0.61)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.26558      0.2666   0.0077614
            1.2     0.20266     0.20297   0.0031677
            1.4     0.15728     0.15737   0.0013035
            1.6     0.12378     0.12381  0.00053913
            1.8    0.098532     0.09854  0.00022369
              2    0.079153    0.079155  9.2995e-05
            2.2    0.064051    0.064051  3.8703e-05
            2.4    0.052134    0.052134  1.6117e-05
            2.6    0.042635    0.042635   6.712e-06
            2.8    0.034999    0.034998  2.7948e-06

Here we use ``fit.format(10)`` to print out a table of ``x`` and 
``y`` (actually ``ymod``) values, together with the value of the 
fit function using the best-fit parameters. There are several things
to notice:

   * Were we really only interested in ``a[0]`` and ``E[0]``, a 
     single-exponential fit would have been adequate. This is because we
     are in effect doing a 20-exponential fit even in that case, by
     including all but the first term as corrections to ``y``. The answers
     given by the first fit are correct (we know the exact values since we
     are using fake data).
     
     The ability to push uninteresting parameters into a ``ymod`` can be
     highly useful in practice since it is usually much cheaper to
     incorporate those fit parameters into ``ymod`` than it is to include
     them as fit parameters --- fits with smaller numbers of parameters are
     usually a lot faster.
    
   * The ``chi**2`` and best-fit parameter means and standard deviations
     are almost unchanged by shifting terms from ``ymod`` back into the
     fit function, as ``nexp`` increases. The final results for
     ``a[0]`` and ``E[0]``, for example, are nearly identical in the
     ``nexp=1`` and ``nexp=4`` fits.
     
     In fact it is straightforward to prove that best-fit parameter means
     and standard deviations, as well as ``chi**2``, should be exactly the
     same in such situations provided the fit function is linear in all fit
     parameters. Here the fit function is approximately linear, given our
     small standard deviations, and so results are only approximately
     independent of ``nexp``.
          
   * The uncertainty in ``ymod`` for a particular ``x`` decreases as 
     ``nexp`` increases and as ``x`` increases. Also the ``nexp``
     independence of the fit results depends upon capturing all of the
     correlations in the correction to ``y``. This is why
     :class:`gvar.GVar`\s are useful since they make the implementation of
     those correlations trivial.
     
   * Although we motivated this example by the need to deal with ``y``\s
     having no errors, it is straightforward to apply the same ideas to 
     a situation where the ``y``\s have errors. Again one might want to 
     do so since fitting uninteresting fit parameters is generally more 
     costly than absorbing them into the ``y`` (which then has a modified
     mean and standard deviation).
     

SVD Cuts and Roundoff Error
-----------------------------
We did not display values for ``E1/E0``, ``a1/a0`` ... in the example in 
the previous section. Had we done so a problem would have been immediately
apparent: for example, ::

   ************************************* nexp = 4
   Least Square Fit (input data correlated with prior):
     chi2/dof [dof] = 0.057 [10]    Q = 1    logGBF = -67.447    itns = 5

   Parameters:
                 a_    0.399937 +-  0.00077           (     0.5 +-      0.5)
                  _    0.398315 +-    0.032           (     0.5 +-      0.5)
                  _    0.401742 +-      0.1           (     0.5 +-      0.5)
                  _    0.403269 +-     0.15           (     0.5 +-      0.5)
                 E_    0.899975 +-   0.0003           (       1 +-      0.5)
                  _     1.79859 +-    0.023           (     1.9 +-     0.54)
                  _     2.69522 +-     0.19           (     2.8 +-     0.57)
                  _     3.60827 +-     0.28           (     3.7 +-     0.61)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.26558      0.2666   0.0077614
            1.2     0.20266     0.20297   0.0031677
            1.4     0.15728     0.15737   0.0013035
            1.6     0.12378     0.12381  0.00053913
            1.8    0.098532     0.09854  0.00022369
              2    0.079153    0.079155  9.2995e-05
            2.2    0.064051    0.064051  3.8703e-05
            2.4    0.052134    0.052134  1.6117e-05
            2.6    0.042635    0.042635   6.712e-06
            2.8    0.034999    0.034998  2.7948e-06

   E1/E0 = 1.99849 +- 0.154988   E2/E0 = 2.99477 +- 1.65242
   a1/a0 = 0.995944 +- 0.514388   a2/a0 = 1.00451 +- 2.32754
   
The standard deviations quoted for ``E1/E0``, *etc.* are much too large
compared with the standard deviations shown for the individual parameters.
This is due to roundoff error. The standard deviations quoted for the
parameters are computed differently from the standard deviations in
``fit.p`` (which was used to calculate ``E1/E0``). The former come directly
from the curvature of the ``chi**2`` function at its minimum; the latter
are related back to the standard deviations of the input data and priors
used in the fit. The two should agree, but they will not agree if the
covariance matrix for the input ``y`` data is too ill-conditioned.

The inverse of the ``y`` covariance matrix is used in the ``chi**2``
function that is minimized by :class:`lsqfit.nonlinear_fit`. Given the
finite precision of computer hardware, it is impossible to compute this
inverse accurately if the matrix is singular or almost singular, and in
such situations the reliability of the fit results is in question. The
eigenvalues of the covariance matrix in this example (for ``nexp=6``)
indicate that this is the case: they range from ``7.2e-5`` down to
``4.2e-26``, covering 21 orders of magnitude. This is likely too large a
range to be handled with the 16--18 digits of precision available in normal
double precision computation. The smallest eigenvalues and their
eigenvectors are likely to be quite inaccurate, as is any method for
computing the inverse matrix.

The standard solution to this common problem in least-squares fitting is 
to introduce an *svd* cut, here called ``svdcut``::

   fit = nonlinear_fit(data=(x,ymod),fcn=f,prior=prior,p0=p0,svdcut=1e-12)
   
Then the inverse of the ``y`` covariance matrix is computed from its
eigenvalues and eigenvectors, but with any eigenvalue smaller than
``svdcut`` times the largest eigenvalue replaced by the cutoff (that is,
by ``svdcut`` times the largest eigenvalue). This limits the singularity of
the covariance matrix, leading to improved numerical stability. The cost is
less precision in the final results since we are in effect decreasing the
precision of the input ``y`` data (a conservative move); but numerical
stability is worth the tradeoff.

Rerunning our fit with ``svdcut=1e-12`` we obtain ::

   ************************************* nexp = 4
   Least Square Fit (input data correlated with prior):
     chi2/dof [dof] = 0.053 [10]    Q = 1    logGBF = -55.494    itns = 3

   Parameters:
                 a_    0.400162 +-   0.0013           (     0.5 +-      0.5)
                  _    0.404161 +-    0.039           (     0.5 +-      0.5)
                  _    0.404572 +-     0.11           (     0.5 +-      0.5)
                  _    0.408034 +-     0.16           (     0.5 +-      0.5)
                 E_    0.900066 +-  0.00052           (       1 +-      0.5)
                  _     1.80348 +-    0.031           (     1.9 +-     0.54)
                  _     2.71749 +-     0.21           (     2.8 +-     0.57)
                  _     3.62392 +-     0.29           (     3.7 +-     0.61)

   Fit:
            x_i         y_i      f(x_i)        dy_i
   ------------------------------------------------
              1     0.26558     0.26686   0.0077614
            1.2     0.20266     0.20309   0.0031677
            1.4     0.15728     0.15742   0.0013035
            1.6     0.12378     0.12383  0.00053913
            1.8    0.098532     0.09855  0.00022369
              2    0.079153    0.079159  9.2995e-05
            2.2    0.064051    0.064053  3.8703e-05
            2.4    0.052134    0.052135  1.6117e-05
            2.6    0.042635    0.042635   6.712e-06
            2.8    0.034999    0.034999  2.7948e-06

   E1/E0 = 2.00372 +- 0.0330005   E2/E0 = 3.01921 +- 0.234244
   a1/a0 = 1.00999 +- 0.0955902   a2/a0 = 1.01102 +- 0.269968

and consistency has been restored. Note that taking ``svdcut=-1e-12`` (with a
minus sign) causes the problematic modes to be dropped. This is a more
conventional implementation of *svd* cuts, but here it results in much less
precision than using ``svdcut=1e-12`` (for example, ``2.01972 +- 0.115874``
for ``E1/E0``, which is almost four times less precise). Dropping modes is
equivalent to setting the corresponding variances equal to infinity, which is
(obviously) much more conservative and less realistic than setting them equal
to the *svd*\-cutoff variance.

The error budget is interesting in this case. There is no contribution from
the original ``y`` data since it was exact. So all statistical uncertainty
comes from the priors in ``max_prior``, and from the *svd* cut, which
contributes since it modifies the effective variances of several eigenmodes of
the covariance matrix. The *svd* contribution can be obtained from
``fit.svdcorrection`` so the full error budget is constructed by the following
code, ::

   outputs = {'E1/E0':E[1]/E[0], 'E2/E0':E[2]/E[0],         
              'a1/a0':a[1]/a[0], 'a2/a0':a[2]/a[0]}
   inputs = {'E':max_prior['E'],'a':max_prior['a'],'svd':fit.svdcorrection}
   print(fit.fmt_values(outputs))
   print(fit.fmt_errorbudget(outputs,inputs))

which gives::

   Values:
                 E2/E0: 3.019(234)          
                 E1/E0: 2.004(33)           
                 a2/a0: 1.011(270)          
                 a1/a0: 1.010(96)           

   Partial % Errors:
                            E2/E0     E1/E0     a2/a0     a1/a0
   ------------------------------------------------------------
                     a:      2.53      0.66     10.71      3.47
                   svd:      1.30      0.49      1.81      2.46
                     E:      7.22      1.43     24.39      8.45
   ------------------------------------------------------------
                 total:      7.76      1.65     26.70      9.46
   
Here the contribution from the *svd* cut is rather modest.

The method :func:`lsqfit.nonlinear_fit.check_roundoff` can be used to check
for roundoff errors. It generates a warning if roundoff looks to be a problem.


Bootstrap Error Analysis
------------------------
Our analysis above assumes that every probability distribution relevant to
the fit is approximately gaussian. For example, we characterize the input
data for ``y`` by a mean and a covariance matrix obtained from averaging
many random samples of ``y``. For large sample sizes it is almost certainly
true that the average values follow a gaussian distribution, but in
practical applications the sample size could be too small. The *statistical
bootstrap* is an analysis tool for dealing with such situations.

The strategy is to: 1) make a large number of "bootstrap copies" of the
original input data that differ from each other by random amounts
characteristic of the underlying randomness in the original data; 2) repeat
the entire fit analysis for each bootstrap copy of the data, extracting
fit results from each; and 3) use the variation of the fit results from
bootstrap copy to bootstrap copy to determine an approximate probability
distribution (possibly non-gaussian) for the each result.
   
Consider the code from the previous section, where we might reasonably want 
another check on the error estimates for our results. That code can be
modified to include a bootstrap analysis by adding the following to the end of
the ``main()`` subroutine::
   
   Nbs = 40                                     # number of bootstrap copies
   outputs = {'E1/E0':[], 'E2/E0':[], 'a1/a0':[],'a2/a0':[]}   # results
   for bsfit in fit.bootstrap_iter(n=Nbs):
       E = bsfit.pmean['E']                     # best-fit parameter values
       a = bsfit.pmean['a']                     #   (ignore errors)
       outputs['E1/E0'].append(E[1]/E[0])       # accumulate results
       outputs['E2/E0'].append(E[2]/E[0])
       outputs['a1/a0'].append(a[1]/a[0])
       outputs['a2/a0'].append(a[2]/a[0])
   # extract means and standard deviations from the bootstrap output
   from numpy import mean,std
   for k in outputs:
       outputs[k] = gv.gvar(np.mean(outputs[k]),np.std(outputs[k]))
   print('Bootstrap results:')
   print('E1/E0 =',outputs['E1/E0'],'  E2/E1 =',outputs['E2/E0'])
   print('a1/a0 =',outputs['a1/a0'],'  a2/a0 =',outputs['a2/a0'])
   
The results are consistent with the results obtained directly from the fit
(when using ``svdcut=1e-12``)::

   Bootstrap results:
   E1/E0 = 2.00618 +- 0.027411   E2/E1 = 3.05219 +- 0.195792
   a1/a0 = 1.01777 +- 0.0755551   a2/a0 = 1.06962 +- 0.275993

In particular, the bootstrap analysis confirms our previous error estimates
(to within 10-20%, since ``Nbs=40``). When a quantity is not particularly 
gaussian, using medians instead of means might be more robust.


Troubleshooting
---------------
:class:`lsqfit.nonlinear_fit` sometimes gives unintelligible error messages 
such as::

   Traceback (most recent call last):
     File "<stdin>", line 10, in <module>
       fit = nonlinear_fit(data=(None,y),prior=prior,fcn=f)
     File "/Users/gpl/Library/Python/2.7/lib/python/site-packages/lsqfit/__init__.py", line 240, in __init__
       fit = multifit(p0, nf, self._chiv, **self.fitterargs)
     File "_utilities.pyx", line 303, in lsqfit._utilities.multifit.__init__ (src/lsqfit/_utilities.c:2668)
   RuntimeError: Python error in fit function: 33

Such messages come from inside the *gsl* routines that are actually doing
the fits and are usually due to an error in one of the inputs to the fit 
(that is, the fit data, the prior, or the fit function). Setting ``debug=True``
in the argument list of :class:`lsqfit.nonlinear_fit` might result in more 
intelligible error messages. This option also causes the fitter to check 
for significant roundoff errors in the matrix inversions of the covariance
matrices.

Occasionally :class:`lsqfit.nonlinear_fit` appears to go crazy, with gigantic
``chi**2``\s (*e.g.*, ``1e78``). This could be because there is a genuine
zero-eigenvalue mode in the covariance matrix of the data or prior. Such a
zero mode makes it impossible to invert the covariance matrix when evaluating
``chi**2``. One fix is to include *svd* cuts in the fit by setting, for
example, ``svdcut=(1e-14,1e-14)`` in the call to :class:`lsqfit.nonlinear_fit`.
These cuts will exclude exact or nearly exact zero modes, while leaving
important modes mostly unaffected.

Even if the *svd* cuts work in such a case, the question remains as to why one
of the covariance matrices has a zero mode. A common cause is if the same
:class:`gvar.GVar` was used for more than one prior. For example, one might
think that ::

   >>> import gvar as gv
   >>> z = gv.gvar(1,1)
   >>> prior = gv.BufferDict(a=z,b=z)

creates a prior ``1 +- 1`` for each of parameter ``a`` and parameter ``b``.
Indeed each parameter separately is of order ``1 +- 1``, but in a fit the two
parameters would be forced equal to each other because their priors are both
set equal to the same :class:`gvar.GVar`, ``z``::

   >>> print(prior['a'],prior['b'])
   1 +- 1 1 +- 1
   >>> print(prior['a']-prior['b'])
   0 +- 0

That is, while parameters ``a`` and ``b`` fluctuate over a range of 
``1 +- 1``, they fluctuate together, in exact lock-step. The covariance matrix
for ``a`` and ``b`` must therefore be singular, with a zero mode corresponding
to the combination ``a-b``; it is all ``1``\s in this case::

   >>> import numpy as np
   >>> cov = gv.evalcov(prior.flat)    # prior's covariance matrix
   >>> print(np.linalg.det(cov))       # determinant is zero
   0.0

This zero mode upsets :func:`nonlinear_fit`. If ``a`` and ``b`` are meant to
fluctuate together then an *svd* cut as above will give correct results (with
``a`` and ``b`` being forced equal to several decimal places, depending upon
the cut). Of course, simply replacing ``b`` by ``a`` in the fit function would
be even better. If, on the other hand, ``a`` and ``b`` were not meant to
fluctuate together, the prior should be redefined::

   >>> prior = gv.BufferDict(a=gv.gvar(1,1),b=gv.gvar(1,1))

where now each parameter has its own :class:`gvar.GVar`.   
   



