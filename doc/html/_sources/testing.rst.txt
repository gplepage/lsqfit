.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |vegas_fit| replace:: :class:`lsqfit.vegas_fit`
.. |vegas| replace:: :mod:`vegas`
.. |BufferDict| replace:: :class:`gvar.BufferDict`
.. |~| unicode:: U+00A0
   :trim:

.. highlight:: python

.. _non-gaussian-behavior:

Non-Gaussian Behavior; Testing Fits
=====================================

Introduction
--------------------
The various analyses in the Tutorial assume implicitly that every
probability distribution relevant to a fit is Gaussian. The input
data and priors are assumed Gaussian. The ``chi**2`` function is
assumed to be well approximated by a Gaussian in the vicinity of
its minimum, in order to estimate uncertainties for the best-fit
parameters. Functions of those parameters are assumed to yield
results that are described by Gaussian random variables. These assumptions
are usually pretty good for high-statistics data, when standard deviations
are small, but can lead to problems with low statistics.

Here we present three methods for testing these assumptions.
Some of these techniques, like the *statistical bootstrap* and
Bayesian integration, can also be used to analyze non-Gaussian
results.


Bayesian Integrals
-------------------
|vegas_fit| provides an alternative fitting strategy
(multi-dimensional Bayesian integrals) from that used by
|nonlinear_fit|. Both approaches assume that the fit parameters 
are described by a probability distribution whose 
probability density function (PDF) is proportional to 
:math:`\exp(-\chi^2(p)/2)`.  :math:`\chi^2(p)` 
has contributions from both the data and the prior:

.. math::

        \chi^2(p) \equiv \Delta y^T \cdot\mathrm{cov}^{-1}_y \cdot \Delta y
        \: + \: 
        \Delta p^T \cdot\mathrm{cov}^{-1}_\mathrm{prior}\cdot\Delta p,

where :math:`\Delta y_i \equiv \overline y_i - f(x_i,p)` 
and :math:`\Delta p_i\equiv \overline p_i^\mathrm{prior} - p_i`.
Both of these approaches characterize this distribution by 
specifying *best-fit* mean values and covariances for the 
fit parameters (packaged as an array or dictionary of |GVar|\s). 
|nonlinear_fit| estimates the mean values 
and covariances from the minimum of :math:`\chi^2(p)` and its
curvature at the minimum, while |vegas_fit| calculates the 
actual means and standard deviations of the parameters 
by evaluating the following integrals:

.. math::

    \overline p_i &\equiv \frac{1}{N_\mathrm{pdf}}\int d^np\,p_i\,\mathrm{e}^{-\chi^2(p)/2}\\[1.5ex]
    \mathrm{cov}(p_i,p_j) &\equiv \frac{1}{N_\mathrm{pdf}}\int d^np\,(p_i - \overline p_i)(p_j - \overline p_j)\,\mathrm{e}^{-\chi^2(p)/2}\\[1.5ex]
    N_\mathrm{pdf} &\equiv \int d^np\,\mathrm{e}^{-\chi^2(p)/2}

The integrals are evaluated numerically, using adaptive Monte Carlo integration 
(:class:`PDFIntegrator` from the :mod:`vegas`  
module). 
The best-fit results from the two approaches agree when 
:math:`\chi^2(p)` is well approximated by the quadratic expansion around its 
minimum --- that is,
insofar as
:math:`\exp(-\chi^2(p)/2)` is well approximated
by a Gaussian distribution in the parameters. 
But the results can differ significantly otherwise; the output from ``nonlinear_fit``
is the Gaussian approximation to that from ``vegas_fit``.

To compare ``vegas_fit`` with ``nonlinear_fit``, we revisit
the analysis in the section
on :ref:`correlated-parameters`. We modify the end of the ``main()`` function
in the original code to repeat the analysis using ``vegas_fit``::

    import numpy as np
    import gvar as gv
    import lsqfit
    import vegas

    def main():
        x, y = make_data()
        prior = make_prior()

        # nonlinear_fit
        fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
        print(20 * '-', 'nonlinear_fit')
        print(fit)
        print('p1/p0 =', fit.p[1] / fit.p[0], '   prod(p) =', np.prod(fit.p))
        print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(fit.p[:2])[1,0]), '\n')

        # vegas_fit
        vfit = lsqfit.vegas_fit(prior=prior, data=(x,y), fcn=fcn)
        print(20 * '-', 'vegas_fit')
        print(vfit)
        # measure p1/p0 and prod(p)
        @vegas.rbatchintegrand
        def g(p):
            return {'p1/p0':p[1] / p[0], 'prod(p)':np.prod(p, axis=0)}
        s = vfit.stats(g)
        print('p1/p0 =', s['p1/p0'], '   prod(p) =', s['prod(p)'])
        print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(vfit.p[:2])[1,0]))
    
    def make_data():
        x = np.array([
            4., 2., 1., 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625
            ])
        y = gv.gvar([
            '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
            '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
            '0.044(16)'
            ])
        return x, y

    def make_prior():
        p = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
        p[1] = 20 * p[0] + gv.gvar('0.0(1)')        # p[1] correlated with p[0]
        return p

    @vegas.rbatchintegrand
    def fcn(x, p):
        if p.ndim == 2:
            # add batch index to x if in batch mode
            x = x[:, None]
        return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

    if __name__ == '__main__':
        main()

Running this code gives the following output:

..  literalinclude:: eg3.5a.out
    :language: none

There are several things to notice about these results:

* The fit results ``vfit.p`` from |vegas_fit| are quite similar 
  to those from |nonlinear_fit| (``fit.p``), as are ``vfit.chi2`` 
  and ``fit.chi2``, and ``vfit.logBF`` and ``fit.logGBF``. This 
  suggests that the Gaussian approximation used by 
  |nonlinear_fit| is a reasonable approximation to the 
  full Bayesian analysis used by |vegas_fit|.

* ``vfit.logBF`` has an uncertainty of about 0.1\%. This comes 
  from the uncertainty in the |vegas| estimate of
  the norm of the PDF (:math:`N_\mathrm{pdf}` above). 
  :mod:`vegas` uses adaptive Monte Carlo integration 
  to estimate the values of integrals, as well as the
  uncertainties in those estimates. 
  
  The accuracy of 
  |vegas_fit|'s integrals can almost always be 
  improved by using information from the fit 
  with |nonlinear_fit|. For example, replacing 
  the ``vfit`` line in the code above with ::

    vfit = lsqfit.vegas_fit(prior=prior, data=(x,y), fcn=fcn, param=fit.p)

  reduces the error on ``vfit.logBF`` by about a factor of 
  two:

  ..  literalinclude:: eg3.5d.out 
      :language: none

  The integrals for the means and covariances are similarly improved
  (compare ``vfit.p.vegas_mean`` and ``vfit.p.vegas_cov``
  with and without ``param=fit.p``).

  The integrator re-expresses the
  fit parameter integrals in terms of new variables that are 
  optimized for integrating the (Gaussian) distribution 
  corresponding to ``param``. By default ``param=prior``,
  but ``param=fit.p`` is almost certainly a better match
  to the actual PDF used in the integrals. 
  
  A more 
  succinct way to use results from |nonlinear_fit| 
  object ``fit`` 
  is ::

    vfit = lsqfit.vegas_fit(fit=fit)

  where ``vfit``'s  ``prior``, ``data``, and ``fcn`` 
  are copied from ``fit``.

* Values for ``p[1]/p[0]`` and for the product of all 
  the ``p[i]``\s are obtained from ``vfit.stats(g)``.
  This uses ``vfit``'s (trained) integrator to evaluate 
  the means and covariances of the components of ``g(p)``. 
  While results 
  from the two fits agree well on ``p[1]/p[0]``, results
  for ``prod(p)`` do not agree so well. This suggests that
  the distribution for ``prod(p)`` in not as well 
  approximated by a Gaussian. 
  
  More information about the 
  distributions can be obtained from ``vfit.stats`` by 
  using keywords ``moments`` and ``histograms``::

    s = vfit.stats(g, moments=True, histograms=True)
    for k in s.stats:
        print('\n' + 20 * '-', k)
        print(s.stats[k])
        plot = s.stats[k].plot_histogram()
        plot.xlabel(k)
        plot.show()

  This results in the following output

  ..  literalinclude:: eg3.5b.out
      :language: none

  together with histogram plots for the distributions of 
  ``p[1]/p[0]`` and ``prod(p)``:

  .. image:: eg3.5a.png
        :width: 95%

  The distribution for ``prod(p)`` is clearly skewed. The plot 
  shows the actual distribution (gray bars) and the 
  Gaussian (blue dots) corresponding to ``s['prod(p)']``, 0.55 ± 0.41.
  It also shows fits to two two-sided Gaussian models: one that is 
  continuous (split-normal, solid green line) and another centered 
  on the median that is discontinuous (red dashes). The median 
  fit suggests that a better description of the ``prod(p)`` distribution 
  might be 0.44 plus 0.45 minus 0.24, although any of the three 
  models gives a reasonable impression
  of the range of possible values for ``prod(p)``.

* A simple way to create histograms and contour plots of the probability density 
  is from samples drawn from the underlying distribution used in the fit::

    import corner 
    import matplotlib.pyplot as plt 

    wgts,psamples = vfit.sample(nbatch=100_000)
    samples = dict()
    samples['p3'] = psamples[3]
    samples['p1/p0'] = psamples[1] / psamples[0]
    samples['prod(p)'] = np.prod(psamples, axis=0)
    corner.corner(
        data=samples, weights=wgts, range=3 * [0.99], 
        show_titles=True, quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False, fill_contours=True,
        )
    plt.show()

  Here :meth:`lsqfit.vegas_fit.sample` is used to draw approximately 100,000
  samples whose weighted density is proportional to :math:`\exp(-\chi^2(p)/2)`.
  The samples corresponding to parameter ``p[d]`` are ``psamples[d, i]`` where 
  ``i=0,1,2...100_000(approx)``; the corresponding weights are ``wgts[i]``. 
  Samples for the quantities of interest are 
  collected in dictionary ``samples``. The :mod:`corner` Python module is used 
  to create histograms of the probability density for each of the quantities in 
  ``sample``; it also creates contour plots of the joint densities for each 
  pair of quantities:

  .. image:: eg3.5b.png
        :width: 90%

  The histograms are labeled by the median value plus or minus intervals that 
  each enclose 34% of the probability (``quantiles=[0.16, 0.5, 0.84]``).

  The :mod:`corner` module (and the :mod:`arviz` module) must be installed 
  separately.

* A |vegas| integration is much faster if the integrand
  can process large batches of integration points 
  simultaneously. An example is ``fcn(x, p)`` above. When 
  called by |nonlinear_fit|, parameter ``p`` represents 
  a single point in parameter space with coordinates 
  ``p[d]`` where ``d=0...3``. When called by ``vegas_fit``
  (in rbatch mode), ``p`` represents a large number of points 
  in parameter space with coordinates ``p[d,i]`` where 
  ``d=0...3`` labels the direction in parameter space, and
  ``i``, the batch index, labels the different points in 
  parameter space. The function checks to see if it is being used 
  in batch mode, and adds a batch index to ``x`` if it 
  is. The decorator ``@vegas.rbatchintegrand`` tells 
  |vegas| that the function can be called in batch mode.
  (See the |vegas| documentation for more information.)

* |vegas| uses an iterative algorithm to adapt to the 
  PDF. By default, |vegas_fit| uses 10 iterations to 
  train the integrator to the PDF, and then 10 more,
  without further adaptation, to evaluate the integrals
  for the means :math:`\overline p_i` and covariances 
  :math:`\mathrm{cov}(p_i,p_j)` of the fit parameters,
  and the PDF's norm :math:`N_\mathrm{pdf}`  (see 
  equations above). Printing ``vfit.training.summary()`` 
  shows estimates for the norm :math:`N_\mathrm{pdf}` 
  from each of the first 10 iterations (here without 
  ``param=fit.p``):

  ..  literalinclude:: eg3.5c.out
      :language: none

  The uncertainties in the first column 
  are 25–60 times smaller after |vegas| has adapted 
  to the PDF.
  |vegas| averages results from different 
  iterations, but results from the training iterations 
  are frequently unreliable and so are discarded. 
  The final results come from 
  the final 10 iterations (see ``vfit.p.summary()``).

  The accuracy of the integrals is determined by the number 
  of iterations ``nitn`` used and, especially, by the number of integrand 
  evaluations ``neval`` allowed for each iteration. The
  defaults for these parameters are ``nitn=(10,10)`` and 
  ``neval=1000``. The following ::

    vfit = lsqfit.vegas_fit(fit=fit, nitn=(6, 10), neval=100_000)

  reduces the number of training iterations to 6 but also 
  increases the number of integrand evaluations by a factor 
  of 100. The integration errors are then about 20 times smaller,
  which is much smaller than is needed here. This particular 
  problem, however, is relatively easy for |vegas|; other problems could 
  well require hundreds of thousands or millions of integration 
  evaluations per iteration.

* At the end of :ref:`correlated-parameters`, we examined what happened
  to the |nonlinear_fit| when the correlation (``p[1]`` is approximately 
  ``20*p[0]``) was removed from the prior by setting ::

    prior = gv.gvar(['0(1)', '0(20)', '0(1)', '0(1)']).

  The fit result is completely different with the uncorrelated prior when 
  using |nonlinear_fit|. This 
  is *not* the case with |vegas_fit|, where the uncorrelated prior leads 
  to the following fit:

  ..  literalinclude:: eg3.5e.out
      :language: none

  These results are quite similar to what is obtained with the 
  correlated prior, although less accurate. This suggests 
  that the Gaussian approximation assumed by |nonlinear_fit| is 
  unreliable for the uncorrelated problem. This might have
  been anticipated since three of the four parameters have 
  means that are effectively zero (compared to their 
  standard deviations).

A central assumption when using |nonlinear_fit| or |vegas_fit| is that
the data are drawn from a Gaussian distribution. 
:ref:`outliers` shows how to use :class:`vegas.PDFIntegrator`
directly, rather than :class:`lsqfit.vegas_fit`, 
when the input data are not Gaussian. It  discusses 
two versions of a fit,  one with 5 parameters
and the other with 22 parameters.


Bootstrap Error Analysis; Non-Gaussian Output
-------------------------------------------------
The bootstrap provides another way to check on a fit's
validity, and also a method for analyzing non-Gaussian outputs.
The strategy is to:

    1.  make a large number of "bootstrap copies" of the
        original input data and prior that differ from each other
        by random amounts characteristic of the underlying randomness
        in the original data and prior (see
        the documentation for 
        
            :meth:`lsqfit.nonlinear_fit.bootstrapped_fit_iter`

        for more information);

    2.  repeat the entire fit analysis for each bootstrap copy of
        the data and prior, extracting fit results from each;

    3.  use the variation of the fit results from bootstrap copy
        to bootstrap copy to determine an approximate probability
        distribution (possibly non-Gaussian) for the each result.

To illustrate, we revisit the fit in the section
on :ref:`positive-parameters`, where
the goal is to average noisy data subject to 
the constraint that the average must be positive. 
The constraint is likely to introduce strong distortions 
in the probability density function (PDF) given that the 
fit analysis suggests a value of 0.011 |~| ± |~| 0.013. 
We will use a bootstrap analysis to investigate 
the distribution of the average. We do this 
by adding code right after the fit::

    import gvar as gv 
    import lsqfit 
    import numpy as np

    y = gv.gvar([
        '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
        '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
        '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
        '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
        ])

    # nonlinear_fit
    prior = gv.BufferDict()
    prior['f(a)'] = gv.BufferDict.uniform('f', 0, 0.04)

    def fcn(p, N=len(y)):
        return N * [p['a']]

    fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=fcn)
    print(20 * '-', 'nonlinear_fit')
    print(fit)
    print('a =', fit.p['a'])                     

    # Nbs bootstrap copies
    Nbs = 1000
    a = []
    for bsfit in fit.bootstrapped_fit_iter(Nbs):
        a.append(bsfit.p['a'].mean)
    avg_a = gv.dataset.avg_data(a, spread=True)
    print('\n' + 20 * '-', 'bootstrap')
    print('a =', avg_a)
    counts,bins = np.histogram(a, density=True)
    s = gv.PDFStatistics(histogram=(bins, counts))
    print(s)
    plot = s.plot_histogram()
    plot.xlabel('a')
    plot.show()


``fit.bootstrapped_fit_iter(Nbs)`` produces fits ``bsfit`` for each of
``Nbs=1000`` different bootstrap copies of the input data (``y`` and the prior).
We collect the mean values for parameter ``a``, ignoring the uncertainties, and
then calculate the average and standard deviation from these results using
:func:`gvar.dataset.avg_data`. We then use ``gvar.PDFStatistics`` to analyze 
the distribution of the ``a`` values and create a histogram of its PDF.

The bootstrap estimate for ``a`` agrees reasonably well with the result from |nonlinear_fit|,
but the statistical analysis shows that the distribution of ``a`` values is skewed (towards 
positive ``a`` values):

.. literalinclude:: eg3.6a.out
    :language: none

This is confirmed by the histogram: 

.. image:: eg3.6a.png
        :width: 60%

Fitting with |vegas_fit| rather than |nonlinear_fit| gives the same result as the 
bootstrap for the average value of ``a``, but is 10x faster (and more accurate):

.. literalinclude:: eg3.6b.out
    :language: none 

The histogram from |vegas_fit| is also similar to that from the bootstrap.


Testing Fits with Simulated Data
--------------------------------
Ideally we would test a fitting protocol by doing fits of data similar to
our actual fit but where we know the correct values for the fit parameters
ahead of the fit. Method

    :meth:`lsqfit.nonlinear_fit.simulated_fit_iter` 

returns an iterator that
creates any number of such simulations of the original fit.

A key assumption underlying least-squares fitting is that the fit
data ``y[i]`` are random samples from a distribution whose mean
is the fit function ``fcn(x, fitp)`` evaluated with the best-fit
values ``fitp`` for the parameters. ``simulated_fit_iter`` iterators
generate simulated data by drawing other random samples from the
same distribution, assigning them the same covariance matrix as the
original data. The simulated data are fit
using the same priors and fitter settings
as in the original fit, and the results (an :class:`lsqfit.nonlinear_fit`
object) are returned by the iterator. The fits with simulated data should
have good ``chi**2`` values, and the results from these fits
should agree, within errors, with the original fit results since the
simulated data are from the same distribution as the original data. There
is a problem with the fitting protocol if this is not the case most of the
time.

To illustrate we again examine the fits
in the section on :ref:`correlated-parameters`:
we add three fit simulations at the end of the ``main()`` function::

    import numpy as np
    import gvar as gv
    import lsqfit

    def main():
        x, y = make_data()
        prior = make_prior()
        fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
        print(40 * '*' + ' real fit')
        print(fit.format(True))

        # 3 simulated fits
        for sfit in fit.simulated_fit_iter(n=3):
            # print simulated fit details
            print(40 * '=' + ' simulation')
            print(sfit.format(True))

            # compare simulated fit results with exact values (pexact=fit.pmean)
            diff = sfit.p - sfit.pexact
            print('\nsfit.p - pexact =', diff)
            print(gv.fmt_chi2(gv.chi2(diff)))
            print

    def make_data():
        x = np.array([
            4.    ,  2.    ,  1.    ,  0.5   ,  0.25  ,  0.167 ,  0.125 ,
            0.1   ,  0.0833,  0.0714,  0.0625
            ])
        y = gv.gvar([
            '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
            '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
            '0.044(16)'
            ])
        return x, y

    def make_prior():
        p = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
        p[1] = 20 * p[0] + gv.gvar('0.0(1)')        # p[1] correlated with p[0]
        return p

    def fcn(x, p):
        return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

    if __name__ == '__main__':
        main()

This code produces the following output, showing how the input data
fluctuate from simulation to simulation:

..  literalinclude:: eg3f.out
    :language: none

The parameters ``sfit.p`` produced by the simulated fits agree well
with the original fit parameters ``pexact=fit.pmean``, with good
fits in each case. We calculate the ``chi**2`` for the difference
``sfit.p - pexact`` in
each case; good ``chi**2`` values validate the parameter values, standard
deviations, and correlations.

.. _goodness-of-fit:

Goodness of Fit
------------------
The quality of a fit is often judged by the value of
``chi**2/N``, where ``N`` is the
number of degrees of freedom.
Conventionally we expect ``chi**2/N`` to be of order ``1 ± sqrt(2/N)``
since fluctuations in the mean values of the data are of order the
uncertainties in the data. More precisely the means are
assumed to be random samples drawn from a Gaussian distribution whose
means are given by the best-fit function and whose covariance matrix
comes from
the data. There are two situations where this measure of goodness-of-fit
becomes unreliable.

The first situation is when there is a large SVD cut on the data.
As discussed in :ref:`svd-cuts-statistics`, an SVD cut increases
the uncertainties
in the data without increasing the random fluctuations in the
data means. As a result contributions
from the parts of the ``chi**2`` function affected by the SVD cut
tend to be much smaller than naively expected, artificially
pulling ``chi**2/N`` down.

The second situation that compromises ``chi**2`` is when some or all of
the priors used in a fit are broad --- that is, when a fit result for
a parameter has a much
smaller uncertainty than the corresponding prior, but a mean that
is artificially
close to the prior's mean. This often arises when the means
used in the priors are not random samples, which is
frequently the case  (unlike for the fit data). Again contributions to ``chi**2``
associated with such priors tend to be much smaller than naively expected,
pulling ``chi**2`` down.

These complications can conspire to make ``chi**2/N``
significantly less than |~| 1 when the fit is good. Of greater concern,
they can mask evidence of a bad fit:  ``chi**2/N ≈ 1`` is *not*
necessarily evidence of a good fit in such situations.

A simple way to address these situations is to redo the fit with
keyword parameter ``noise=True``. 
This causes :class:`lsqfit.nonlinear_fit` to
add extra fluctuations to the means in the prior and the data
that are characteristic of the probability distributions associated
with the priors and the SVD cut, respectively::

    prior =>  prior + (gv.sample(prior) - gv.mean(prior))
        y =>  y + gv.sample(y.correction)

These fluctuations
should leave fit results unchanged (within errors) but increase
``chi**2/N`` so it is of order one.

To add this test to the fit from :ref:`svd-cuts-statistics`, we modify the
code to include a second fit at the end::

    import numpy as np
    import gvar as gv
    import lsqfit

    def main():
        ysamples = [
            [0.0092441016, 0.0068974057, 0.0051480509, 0.0038431422, 0.0028690492], 
            [0.0092477405, 0.0069030565, 0.0051531383, 0.0038455855, 0.0028700587], 
            [0.0092558569, 0.0069102437, 0.0051596569, 0.0038514537, 0.0028749153], 
            [0.0092294581, 0.0068865156, 0.0051395262, 0.003835656, 0.0028630454], 
            [0.009240534, 0.0068961523, 0.0051480046, 0.0038424661, 0.0028675632],
            ]
        y = gv.dataset.avg_data(ysamples)
        x = np.array([15., 16., 17., 18., 19.])
        def fcn(p):
            return p['a'] * gv.exp(- p['b'] * x)
        prior = gv.gvar(dict(a='0.75(5)', b='0.30(3)'))
        fit = lsqfit.nonlinear_fit(data=y, prior=prior, fcn=fcn, svdcut=0.0028)
        print(fit.format(True))

        print('\n================ Add noise to prior, SVD')
        noisyfit = lsqfit.nonlinear_fit(
            data=y, prior=prior, fcn=fcn, svdcut=0.0028, noise=True,
            )
        print(noisyfit.format(True))

    if __name__ == '__main__':
        main()

Running this code gives the following output:

..  literalinclude:: eg10e.out
    :language: none

The fit with extra noise has a larger ``chi**2``, as expected,
but is still a good fit. It also
gives fit parameters that agree within errors
with those from the
original fit. In general, there is probably something wrong with
the original fit (e.g., ``svdcut``
too small, or priors inconsistent with the fit data)
if adding noise makes ``chi**2/N`` signficantly larger than one,
or changes the best-fit values of the parameters significantly.

.. _fit-residuals:

Fit Residuals and Q-Q Plots
---------------------------
It can be useful to examine the normalized residuals from a fit (in array
``fit.residuals``). These are the differences between
the data and the corresponding values from the fit function using the best-fit
values for the fit parameters. The differences are projected onto the 
eigenvectors of the correlation matrix and normalized by dividing by the 
square root of the corresponding eigenvalues. 
The statistical assumptions underlying |nonlinear_fit| imply that the 
normalized fit residuals should be uncorrelated and distributed 
randomly about zero in 
a Gaussian distribution. 

One way to test whether residuals from a fit have a Gaussian distribution
is to make a *Q-Q plot.* Plots for the two fits from the previous section
(one without extra noise on the left, and the other with noise) are:

.. image:: eg10e1.png
    :width: 49%

.. image:: eg10e2.png
    :width: 49%

These plots were made using 

    :meth:`lsqfit.nonlinear_fit.qqplot_residuals`, 

by adding the following lines at the end 
of the ``main()`` method::

    fit.qqplot_residuals().show()
    noisyfit.qqplot_residuals().show()

In each case the residuals are first ordered, from smallest to largest. 
They are then plotted against the value
expected for a residual at that position in the list if 
the list elements were drawn at 
random from a Gaussian distribution of unit width and zero mean. (For 
example, given 100 samples from the Gaussian distribution, the sample 
in position 16 of the ordered list should have a value around -1 since 
16% of the values should be more than one standard deviation below the 
mean.) The residuals are consistent with a Gaussian distribution if they
fall on or near a straight line in such a plot.

The plots show fits of the residuals to straight lines (red solid lines).
The residuals in the left plot (without additional noise) are reasonably
consistent with a straight line (and, therefore, a Gaussian distribution), 
but the slope |~| (0.55) is much less than 
one. This is because ``chi2/dof = 0.38`` for this fit 
is much smaller than one. Typically
we expect the slope to be roughly the square root of |~| ``chi2/dof``
(since ``chi2`` equals the sum of the residuals squared).

The residuals in the right plot are also quite linear in the Q-Q plot. 
In this case the fit residuals include extra noise associated with 
the priors and with the SVD cut, as discussed in the previous section.
As a result ``chi2/dof = 0.81`` is much closer to one, as is the 
resulting slope |~| (0.90) in the Q-Q plot. The fit line through 
the residuals is much closer here to the dashed line in the plot,
which is what would result if the residuals had unit standard 
deviation and zero mean.

The fits pictured above have relatively few residuals. 
Q-Q plots become increasingly 
compelling as the number of residuals increases. 
The following plots, from fits without (left) 
and with (right) prior/SVD noise,
come from a lattice QCD analysis with 383 |~| residuals:

.. image:: bmixing1.png 
    :width: 49%

.. image:: bmixing2.png 
    :width: 49%



