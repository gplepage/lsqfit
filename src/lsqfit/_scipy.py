# Copyright (c) 2016 G. Peter Lepage.
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

import scipy.optimize
import scipy.linalg
import scipy.special
import numpy
import gvar as _gvar

gammaQ = scipy.special.gammaincc

class scipy_least_squares(object):
    """ :mod:`scipy` fitter for nonlinear least-squares multidimensional fits.

    :class:`scipy_leastq` is a function-class whose constructor does a
    least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
    vector ``x``.

    :class:`scipy_leastsq` is a wrapper for the ``scipy.optimize.leastsq``.

    Args:
        x0 (array of floats): Starting point for minimization.

        n (positive int): Length of vector returned by the fit function ``f(x)``.

        f (array-valued function): ``sum_i f_i(x)**2`` is minimized
            by varying parameters ``x``. The parameters are a 1-d
            :class:`numpy` array of either numbers or :class:`gvar.GVar`\s.

        tol (float or tuple): Assigning ``tol=(xtol, gtol, ftol)`` causes the
            fit to stop searching for a minimum when any of

                ``xtol >=`` relative change in parameters between iterations

                ``gtol >=`` relative size of gradient of ``chi**2``

                ``ftol >=`` relative change in ``chi**2`` between iterations

            is statisfied. See the ``scipy.optimize.least_squares``
            documentation detailed definitions of the stopping conditions.
            Typically one sets ``xtol=1/10**d`` where ``d`` is the number of
            digits of precision desired in the result, while ``gtol<<1`` and
            ``ftol<<1``. Setting ``tol=eps`` where ``eps`` is a number is
            equivalent to setting ``tol=(eps,1e-10,1e-10)``. Setting
            ``tol=(eps1,eps2)`` is equivlent to setting
            ``tol=(eps1,eps2,1e-10)``. Default is ``tol=1e-5``.

        method (str or None): Minimization algorithm. Options
            include:

                ``'trf'``
                    Trusted Region Reflective algorithm (default). Best
                    choice with bounded parameters.

                ``'dogbox'``
                    dogleg algorithm adapted for bounded parameters.

                ``'lm'``
                    Levenberg-Marquardt algorithm as implemented in MINPACK.
                    Best for smaller problems. Does not work with bounded
                    parameters (bounds are ignored).

            Setting ``method=None`` implies the default ``'trf'``.

        maxit (int): Maximum number of function evaluations in search
            for minimum; default is 1000.

    Other arguments include: ``x_jac``, ``loss``, ``tr_solver``,
    ``f_scale``, ``tr_options``, ``bounds``. See the documentation for
    ``scipy.optimize.least_squares`` for information about these and other
    options.

    :class:`lsqfit.scipy_least_squares` objects have the following
    attributes.

    Attributes:
        x (array): Location of the most recently computed (best) fit point.

        cov (array): Covariance matrix at the minimum point.

        description (str): Short description of internal fitter settings.

        f (array): Fit function value ``f(x)`` at the minimum in
            the most recent fit.

        J (array): Gradient ``J_ij = df_i/dx[j]`` for most recent fit.

        nit (int): Number of function evaluations used in last fit to find
            the minimum.

        stopping_criterion (int): Criterion used to
            stop fit:

                0. didn't converge

                1. ``xtol >=`` relative change in parameters between iterations

                2. ``gtol >=`` relative size of gradient of ``chi**2``

                3. ``ftol >=`` relative change in ``chi**2`` between iterations

        error (str or None): ``None`` if fit successful; an error
            message otherwise.

        results (dict): Results returned by ``scipy.optimize.least_squares``.
    """
    def __init__(
        self, x0, n, f, tol=(1e-8, 1e-8, 1e-8),
        maxit=1000, **extra_args
        ):
        super(scipy_least_squares, self).__init__()

        # standardize tol
        if numpy.shape(tol) == ():
            tol = (tol, 1e-10, 1e-10)
        elif numpy.shape(tol) == (1,):
            tol = (tol[0], 1e-10, 1e-10)
        elif numpy.shape(tol) == (2,):
            tol = (tol[0], tol[1], 1e-10)
        elif numpy.shape(tol) != (3,):
            raise ValueError("tol must be number or a 1-, 2-, or 3-tuple")
        self.tol = tol

        self.description = 'method = {}'.format(
            'trf'
            if 'method' not in extra_args or extra_args['method'] is None else
            extra_args['method']
            )
        self.maxit = maxit
        self.n = n
        self.error = None
        self.x0 = x0
        p = len(x0)
        _valder = _gvar.valder(p * [0.0])

        def func(x):
            return numpy.asarray(f(x), float)

        def Dfun(x):
            fx = f(_valder + x)
            ans = numpy.empty(numpy.shape(fx) + numpy.shape(x), float)
            for i in range(len(fx)):
                ans[i, :] = fx[i].der
            return ans

        fit = scipy.optimize.least_squares(
            fun=func, jac=Dfun, x0=x0,
            xtol=tol[0], gtol=tol[1], ftol=tol[2],
            max_nfev=maxit,
            **extra_args
            )
        if fit.status > 4:
            raise RuntimeError('fit crashed -- ' + fit.message)
        self.x = fit.x
        self.f = f(self.x)
        self.J = Dfun(self.x)
        self.nit = fit.nfev
        self.results = fit

        # compute covariance (from scipy's curve_fit)
        _, _s, _VT = scipy.linalg.svd(fit.jac, full_matrices=False)
        _threshold = numpy.finfo(float).eps * max(fit.jac.shape) * _s[0]
        _s = _s[_s > _threshold]
        _VT = _VT[:_s.size]
        self.cov = numpy.dot(_VT.T / _s**2, _VT)

        self.error = None
        if fit.status < 0:
            self.stopping_criterion = 0
        else:
            self.stopping_criterion = {0:0, 1:2, 2:3, 3:1, 4:1}[fit.status]


class scipy_multiminex(object):
    """ :mod:`scipy` minimizer for multidimensional functions.

    :class:`scipy_multiminex` is a function-class whose constructor minimizes a
    multidimensional function ``f(x)`` by varying vector ``x``. This routine
    does *not* use user-supplied information about the gradient of ``f(x)``.

    :class:`scipy_multiminex` is a wrapper for the ``minimize``
    :mod:`scipy` function. It gives access to only part of that
    function.

    Args:
        x0 (array of floats): Starting point for minimization search.
        f: Function ``f(x)`` to be minimized by varying vector ``x``.
        tol (float): Minimization stops when ``x`` has converged to with
            tolerance ``tol``; default is ``1e-4``.
        maxit (positive int): Maximum number of iterations in search for minimum;
            default is 1000.
        analyzer (function): Optional function of the current ``x``.
            This can be used to inspect intermediate steps in the
            minimization, if needed.

    :class:`lsqfit.scipy_multiminex` objects have the following attributes.

    Attributes:
        x (array): Location of the minimum.
        f (float): Value of function ``f(x)`` at the minimum.
        nit (int): Number of iterations required to find the minimum.
        error (Noe or str): ``None`` if fit successful; an error
            message otherwise.
    """
    def __init__(self, x0, f, tol=1e-4, maxit=1000, analyzer=None):
        super(scipy_multiminex, self).__init__()
        # preserve inputs
        self.x0 = x0
        self.tol = tol
        self.maxit = maxit
        res = scipy.optimize.minimize(
            fun=f, x0=x0, tol=tol, options=dict(maxiter=maxit),
            callback=analyzer, method='Nelder-Mead',
            )
        self.x = res.x
        self.f = f(self.x)
        self.error = None if res.success else res.message
        self.nit = res.nit





# following doesn't work sometimes --- not as good as least_squares
# class scipy_leastsq(object):
#     """ :mod:`scipy` fitter for nonlinear least-squares multidimensional fits.

#     Args:
#         x0 (array of floats): Starting point for minimization.
#         n (positive int): Length of vector returned by the fit function ``f(x)``.
#         f (array-valued function): ``sum_i f_i(x)**2`` is minimized
#             by varying parameters ``x``. The parameters are a 1-d
#             :class:`numpy` array of either numbers or :class:`gvar.GVar`\s.
#         tol (float or tuple): Setting ``tol=(xtol, gtol, ftol)`` causes
#             the fit to stop when any of the following criteria is satisfied:

#                 1) step size small: ``|dx_i| <= xtol * (xtol + |x_i|)``;

#                 2) gradient small: ``||g . x||_inf <= gtol * ||f||^2``;

#                 3) residuals small: ``||f(x+dx) - f(x)|| <= ftol * max(||f(x)||, 1)``.

#             Typically one sets ``xtol=1/10**d`` where ``d`` is the number
#             of digits of precision desired in the result, while ``gtol<<1``
#             and ``ftol<<1``. Setting ``tol=eps`` where ``eps`` is a number
#             is equivalent to setting ``tol=(eps,0,0)``. Setting
#             ``tol=(eps1,eps2)`` is equivlent to setting ``tol=(eps1,eps2,0)``.
#             Default is ``tol=1e-5``.
#         maxit (int): Maximum number of function evaluations in search
#             for minimum; default is 1000.

#     Additional arguments include: ``diag``. See the documentation for
#     ``scipy.optimize.leastsq`` for additional information.

#     :class:`scipy_leastq` is a function-class whose constructor does a
#     least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
#     vector ``x``. The following attributes are available:

#     .. attribute:: x

#         Location of the most recently computed (best) fit point.

#     .. attribute:: cov

#         Covariance matrix at the minimum point.

#     .. attribute:: f

#         The fit function ``f(x)`` at the minimum in the most recent fit.

#     .. attribute:: J

#         Gradient ``J_ij = df_i/dx[j]`` for most recent fit.

#     .. attribute:: nit

#         Number of function evaluations used in last fit to find the minimum.

#     .. attribute:: stopping_criterion

#         Criterion used to stop fit:

#             0 => didn't converge

#             1 => step size small

#             2 => gradient small

#             3 => residuals small

#     .. attribute:: error

#         ``None`` if fit successful; an error message otherwise.

#     .. attribute:: results

#         Results returned by ``scipy.optimize.leastsq``.

#     :class:`scipy_leastsq` is a wrapper for the ``scipy.optimize.leastsq``.
#     """
#     def __init__(
#         self, x0, n, f, tol=(1e-8, 1e-8, 1e-8), maxit=1000, **extra_args
#         ):
#         super(scipy_leastsq, self).__init__()

#         # standardize tol
#         if numpy.shape(tol) == ():
#             tol = (tol, 1e-10, 1e-10)
#         elif numpy.shape(tol) == (1,):
#             tol = (tol[0], 1e-10, 1e-10)
#         elif numpy.shape(tol) == (2,):
#             tol = (tol[0], tol[1], 1e-10)
#         elif numpy.shape(tol) != (3,):
#             raise ValueError("tol must be number or a 1-, 2-, or 3-tuple")
#         self.tol = tol

#         # capture parameters
#         self.maxit = maxit
#         self.n = n
#         self.error = None
#         self.x0 = x0
#         p = len(x0)
#         _valder = _gvar.valder(p * [0.0])

#         def func(x):
#             return numpy.asarray(f(x), float)

#         def Dfun(x):
#             fx = f(_valder + x)
#             ans = numpy.empty(numpy.shape(fx) + numpy.shape(x), float)
#             for i in range(len(fx)):
#                 ans[i, :] = fx[i].der
#             return ans

#         fit = scipy.optimize.leastsq(
#             func=func, Dfun=Dfun, x0=x0,
#             xtol=tol[0], gtol=tol[1], ftol=tol[2],
#             full_output=True, maxfev=maxit, **extra_args
#             )
#         status = fit[4]
#         if status > 4 or status < 0:
#             self.error = fit[3]
#             status = 0
#         else:
#             self.error = None
#         self.x = fit[0]
#         self.cov = fit[1]
#         self.f = f(self.x)
#         self.J = Dfun(self.x)
#         self.nit = fit[2]['nfev']
#         self.stopping_criterion = {0:0, 1:2, 2:3, 3:1, 4:1}[status]
#         self.results = fit

#         if self.cov is None and self.nit == self.maxit:
#             # compute covariance (from scipy's curve_fit)
#             # needed when exhaust iterations: leastsq doesn't fill in fit[1]
#             _, _s, _VT = scipy.linalg.svd(self.J, full_matrices=False)
#             _threshold = numpy.finfo(float).eps * max(self.J.shape) * _s[0]
#             _s = _s[_s > _threshold]
#             _VT = _VT[:_s.size]
#             self.cov = numpy.dot(_VT.T / _s**2, _VT)
