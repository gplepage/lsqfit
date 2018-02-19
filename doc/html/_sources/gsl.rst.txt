GSL Routines
===============================================

.. |GVar| replace:: :class:`gvar.GVar`
.. |nonlinear_fit| replace:: :class:`lsqfit.nonlinear_fit`
.. |BufferDict| replace:: :class:`gvar.BufferDict`

Fitters
--------
:mod:`lsqfit` uses routines from the GSL C-library provided it is
installed; GSL is the open-source Gnu Scientific Library. There are two
fitters that are available for use by |nonlinear_fit|.

.. class:: lsqfit.gsl_multifit(n, f, tol=(1e-5, 0.0, 0.0), maxit=1000, alg='lm', solver='qr', scaler='more', factor_up=3.0, factor_down=2.0, avmax=0.75)

    GSL fitter for nonlinear least-squares multidimensional fits.

    :class:`gsl_multifit` is a function-class whose constructor does a
    least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
    vector ``x``.

    :class:`gsl_multifit` is a wrapper for the ``multifit`` GSL routine.

    :parameters:

        * **x0** (*array of floats*): Starting point for minimization.

        * **n** (*positive int*): Length of vector returned by the fit function ``f(x)``.

        * **f** (*array-valued function*): ``sum_i f_i(x)**2`` is minimized
            by varying parameters ``x``. The parameters are a 1-d
            :class:`numpy` array of either numbers or :class:`gvar.GVar`\s.

        * **tol** (*float or tuple*): Assigning ``tol=(xtol, gtol, ftol)`` causes the
            fit to stop searching for a minimum when any of

                ``xtol >=`` relative change in parameters between iterations

                ``gtol >=`` relative size of gradient of ``chi**2``

                ``ftol >=`` relative change in ``chi**2`` between iterations

            is statisfied. See the GSL documentation for detailed
            definitions of the stopping conditions. Typically one sets
            ``xtol=1/10**d`` where ``d`` is the number of digits of precision
            desired in the result, while ``gtol<<1`` and ``ftol<<1``. Setting
            ``tol=eps`` where ``eps`` is a number is equivalent to setting
            ``tol=(eps,1e-10,1e-10)``. Setting ``tol=(eps1,eps2)`` is
            equivlent to setting ``tol=(eps1,eps2,1e-10)``. Default is
            ``tol=1e-5``. (Note: ``ftol`` option is disabled in some versions
            of  the GSL library.)

        * **maxit** (*int*): Maximum number of iterations in search for minimum;
                default is 1000.

        * **alg** (*str*): GSL algorithm to use for minimization. The following
            options are supported (see GSL documentation for more
            information):

                ``'lm'``
                    Levenberg-Marquardt algorithm (default).

                ``'lmaccel'``
                    Levenberg-Marquardt algorithm with geodesic
                    acceleration. Can be faster than ``'lm'`` but
                    less stable. Stability is controlled by damping
                    parameter ``avmax``; setting it to zero turns
                    acceleration off.

                ``'subspace2D'``
                    2D generalization of dogleg algorithm. This
                    can be substantially faster than the two ``'lm'``
                    algorithms.

                ``'dogleg'``
                    dogleg algorithm.

                ``'ddogleg'``
                    double dogleg algorithm.

        * **scaler** (*str*): Scaling method used in minimization. The following
            options are supported (see GSL documentation for more
            information):

                ``'more'``
                    More rescaling, which makes the problem scale
                    invariant. Default.

                ``'levenberg'``
                    Levenberg rescaling, which is not scale
                    invariant but may be more efficient in certain problems.

                ``'marquardt'``
                    Marquardt rescaling. Probably not as good as
                    the other two options.

        * **solver** (*str*): Method use to solve the linear equations for the
            solution from a given step. The following options
            are supported (see GSL documentation for more information):

                ``'qr'``
                    QR decomposition of the Jacobian. Default.

                ``'cholesky'``
                    Cholesky decomposition of the Jacobian. Can
                    be substantially faster than ``'qr'`` but not as reliable
                    for singular Jacobians.

                ``'svd'``
                    SVD decomposition. The most robust for singular
                    situations, but also the slowest.

        * **factor_up** (*float*): Factor by which search region is increased
            when a search step is accepted. Values that are too large
            destablize the search; values that are too small slow down
            the search. Default is ``factor_up=3``.

        * **factor_down** (*float*): Factor by which search region is decreased
            when a search step is rejected. Values that are too large
            destablize the search; values that are too small slow down
            the search. Default is ``factor_up=2``.

        * **avmax** (*float*): Damping parameter for geodesic acceleration. It
            is the maximum allowed value for the acceleration divided
            by the velocity. Smaller values imply less acceleration.
            Default is ``avmax=0.75``.

    **Attributes:**

        **x** (*array*):
            Location of the most recently computed (best) fit point.

        **cov** (*array*):
            Covariance matrix at the minimum point.

        **description** (*str*):
            Short description of internal fitter settings.

        **f** (*array*):
            Fit function value ``f(x)`` at the minimum in
            the most recent fit.

        **J** (*array*):
            Gradient ``J_ij = df_i/dx[j]`` for most recent fit.

        **nit** (*int*):
            Number of function evaluations used in last fit to find
            the minimum.

        **stopping_criterion** (*int*):
            Criterion used to
            stop fit:

                0. didn't converge

                1. ``xtol >=`` relative change in parameters between iterations

                2. ``gtol >=`` relative size of gradient of ``chi**2``

                3. ``ftol >=`` relative change in ``chi**2`` between iterations

        **error** (*str or None*):
            ``None`` if fit successful; an error
            message otherwise.

.. class:: lsqfit.gsl_v1_multifit(x0, n, f, tol=1e-5, maxit=1000, alg='lmsder',analyzer=None)

    Fitter for nonlinear least-squares multidimensional fits. (GSL v1.)

    :class:`gsl_v1_ multifit` is a function-class whose constructor does a
    least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
    vector ``x``.

    :class:`gsl_v1_multifit` is a wrapper for the (older, v1) ``multifit``
    GSL routine (see ``nlin.h``). This package was used in earlier
    versions of :mod:`lsqfit` (<9.0) and is typically not as
    effective as :class:`gsl_multifit`. It is included for legacy
    code.

    :parameters:

        * **x0** (*array of floats*): Starting point for minimization.

        * **n** (*positive int*): Length of vector returned by the fit function ``f(x)``.

        * **f** (*array-valued function*): ``sum_i f_i(x)**2`` is minimized
            by varying parameters ``x``. The parameters are a 1-d
            :class:`numpy` array of either numbers or :class:`gvar.GVar`\s.

        * **tol** (*float or tuple*): Assigning ``tol=(xtol, gtol, ftol)`` causes the
            fit to stop searching for a minimum when any of

                ``xtol >=`` relative change in parameters between iterations

                ``gtol >=`` relative size of gradient of ``chi**2``

                ``ftol >=`` relative change in ``chi**2`` between iterations

            is statisfied. See the GSL documentation for detailed
            definitions of the stopping conditions. Typically one sets
            ``xtol=1/10**d`` where ``d`` is the number of digits of precision
            desired in the result, while ``gtol<<1`` and ``ftol<<1``. Setting
            ``tol=eps`` where ``eps`` is a number is equivalent to setting
            ``tol=(eps,1e-10,1e-10)``. Setting ``tol=(eps1,eps2)`` is
            equivlent to setting ``tol=(eps1,eps2,1e-10)``. Default is
            ``tol=1e-5``. (Note: the ``ftol`` option is disabled in some
            versions of  the GSL library.)

        * **maxit** (*int*): Maximum number of iterations in search for minimum;
                default is 1000.

        * **alg** (*str*): GSL algorithm to use for minimization. Two options are
            currently available: ``"lmsder"``, the scaled LMDER algorithm
            (default); and ``"lmder"``, the unscaled LMDER algorithm.
            With version 2 of the GSL library, another option is ``"lmniel"``,
            which can be useful when there is much more data than parameters.

        * **analyzer** (*callable*): Optional function of ``x,`` ``[...f_i(x)...],``
            ``[[..df_ij(x)..]]``
            which is called after each iteration. This can be used to inspect
            intermediate steps in the minimization, if needed.

    **Attributes:**

        **x** (*array*):
            Location of the most recently computed (best) fit point.

        **cov** (*array*):
            Covariance matrix at the minimum point.

        **f** (*callable*):
            Fit function value ``f(x)`` at the minimum in
            the most recent fit.

        **J** (*array*):
            Gradient ``J_ij = df_i/dx[j]`` for most recent fit.

        **nit** (*int*):
            Number of function evaluations used in last fit to find
            the minimum.

        **stopping_criterion** (*int*):
            Criterion used to
            stop fit:

                0. didn't converge

                1. ``xtol >=`` relative change in parameters between iterations

                2. ``gtol >=`` relative size of gradient of ``chi**2``

                3. ``ftol >=`` relative change in ``chi**2`` between iterations

        **error** (*str or None*):
            ``None`` if fit successful; an error
            message otherwise.

Minimizer
----------
The :func:`lsqfit.empbayes_fit` uses a minimizer from the GSL library
to minimize ``logGBF``.

.. class:: lsqfit.gsl_multiminex(x0, f, tol=1e-4, maxit=1000, step=1, alg='nmsimplex2', analyzer=None)

    Minimizer for multidimensional functions.

    :class:`multiminex` is a function-class whose constructor minimizes a
    multidimensional function ``f(x)`` by varying vector ``x``. This routine
    does *not* use user-supplied information about the gradient of ``f(x)``.

    :class:`multiminex` is a wrapper for the ``multimin`` GSL routine.

    :parameters:

        * **x0** (*array*): Starting point for minimization search.

        * **f** (*callable*): Function ``f(x)`` to be minimized by varying vector ``x``.

        * **tol** (*float*): Minimization stops when ``x`` has converged to with
            tolerance ``tol``; default is ``1e-4``.

        * **maxit** (*int*): Maximum number of iterations in search for minimum;
            default is 1000.

        * **step** (*float*): Initial step size to use in varying components of ``x``;
            default is 1.

        * **alg** (*str*): GSL algorithm to use for minimization. Three options are
            currently available: ``"nmsimplex"``, Nelder Mead Simplex
            algorithm; ``"nmsimplex2"``, an improved version of
            ``"nmsimplex"`` (default); and ``"nmsimplex2rand"``, a version
            of ``"nmsimplex2"`` with random shifts in the start position.

        * **analyzer** (*callable*): Optional function of ``x``, which is called after
            each iteration. This can be used to inspect intermediate steps in
            the minimization, if needed.

    **Attributes:**

        **x** (*array*):
            Location of the minimum.

        **f** (*float*):
            Value of function ``f(x)`` at the minimum.

        **nit** (*int*):
            Number of iterations required to find the minimum.

        **error** (*None or str*):
            ``None`` if minimization successful; an error
            message otherwise.
