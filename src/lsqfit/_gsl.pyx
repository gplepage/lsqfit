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

cimport numpy
cimport cython
cimport gvar

import gvar
import numpy
import sys

from numpy cimport npy_intp as INTP_TYPE
# index type for numpy (signed) -- same as numpy.intp_t and Py_ssize_t

# gsl interface
cdef extern from "gsl/gsl_sf.h":
    struct gsl_sf_result_struct:
        double val
        double err
    int gsl_sf_gamma_inc_Q_e (double a, double x, gsl_sf_result_struct* res)

cdef extern from "gsl/gsl_errno.h":
    void* gsl_set_error_handler_off()
    char* gsl_strerror(int errno)
    int GSL_SUCCESS
    int GSL_CONTINUE
    int GSL_EFAILED
    int GSL_EBADFUNC

cdef extern from "gsl/gsl_nan.h":
    double GSL_NAN

gsl_set_error_handler_off()

cdef extern from "gsl/gsl_sf.h":
    struct gsl_sf_result_struct:
        double val
        double err
    int gsl_sf_gamma_inc_Q_e (double a, double x, gsl_sf_result_struct* res)

cdef extern from "gsl/gsl_matrix_double.h":
    ctypedef struct gsl_matrix:
        int size1
        int size2
        int tda
        double * data
        void * block
        int owner
    gsl_matrix *gsl_matrix_alloc (int n1, int n2)
    void gsl_matrix_set_zero (gsl_matrix * m)
    void gsl_matrix_free (gsl_matrix * m)
    int gsl_matrix_memcpy(gsl_matrix * dest, const gsl_matrix * src);

cdef inline double gsl_matrix_get(gsl_matrix *m, int i, int j):
    return m.data[i*m.tda+j]

cdef inline void gsl_matrix_set(gsl_matrix *m, int i, int j, double x):
    m.data[i*m.tda+j] = x

# cdef gsl_matrix* array2matrix(numpy.ndarray[numpy.float_t,ndim=2] a):
#     cdef Py_ssize_t i1,i2
#     cdef gsl_matrix* m
#     cdef Py_ssize_t n1 = a.shape[0]
#     cdef Py_ssize_t n2 = a.shape[1]
#     m = gsl_matrix_alloc(n1,n2)
#     for i1 from 0<=i1<n1:
#         for i2 from 0<=i2<n2:
#             gsl_matrix_set(m,i1,i2,a[i1,i2])
#     return m

cdef numpy.ndarray[numpy.float_t, ndim=2] matrix2array(gsl_matrix* m):
    """ copies contents of m into numeric array """
    cdef Py_ssize_t i1, i2
    cdef numpy.ndarray[numpy.float_t, ndim=2] ans
    ans = numpy.zeros((m.size1, m.size2), numpy.float_)
    for i1 in range(m.size1):
        for i2 in range(m.size2):
            ans[i1, i2] = gsl_matrix_get(m, i1, i2)
    return ans

cdef extern from "gsl/gsl_vector.h":
    ctypedef struct gsl_vector:
        int size
        int stride
        double *data
        void *block
        int owner
    gsl_vector* gsl_vector_alloc(int N)
    void gsl_vector_free(gsl_vector *v)

cdef inline void gsl_vector_set(gsl_vector *v, int i, double x):
    v.data[i*v.stride] = x

cdef inline double gsl_vector_get(gsl_vector *v, int i):
    return v.data[i*v.stride]

cdef  gsl_vector* array2vector(numpy.ndarray[numpy.float_t, ndim=1] a):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = a.shape[0]
    cdef gsl_vector* v = gsl_vector_alloc(n)
    for i from 0<=i<n:   # in range(n):
        gsl_vector_set(v, i, a[i])
    return v

cdef numpy.ndarray[numpy.float_t, ndim=1] vector2array(gsl_vector* v):
    """ copies contents of v into numeric array """
    cdef Py_ssize_t i
    cdef numpy.ndarray[numpy.float_t, ndim=1] ans
    ans = numpy.zeros(v.size, numpy.float_)
    for i in range(v.size):
        ans[i] = gsl_vector_get(v, i)
    return ans

cdef extern from "gsl/gsl_multifit_nlinear.h":
    ctypedef enum gsl_multifit_nlinear_fdtype:
        GSL_MULTIFIT_NLINEAR_FWDIFF
        GSL_MULTIFIT_NLINEAR_CTRDIFF

    ctypedef struct gsl_multifit_nlinear_fdf:
        int (* f) (const gsl_vector * x, void * params, gsl_vector * f)
        int (* df) (const gsl_vector * x, void * params, gsl_matrix * df)
        int (* fvv) (const gsl_vector * x, const gsl_vector * v, void * params,
                   gsl_vector * fvv)
        size_t n        # number of functions
        size_t p        # number of independent variables
        void * params   # user parameters
        size_t nevalf   # number of function evaluations
        size_t nevaldf  # number of Jacobian evaluations
        size_t nevalfvv # number of fvv evaluations

    ctypedef struct gsl_multifit_nlinear_trs:
        pass

    ctypedef struct gsl_multifit_nlinear_scale:
        pass

    ctypedef struct gsl_multifit_nlinear_solver:
        pass

    ctypedef struct gsl_multifit_nlinear_parameters:
        const gsl_multifit_nlinear_trs *trs        # trust region subproblem method
        const gsl_multifit_nlinear_scale *scale    # scaling method
        const gsl_multifit_nlinear_solver *solver  # solver method
        gsl_multifit_nlinear_fdtype fdtype         # finite difference method
        double factor_up                           # factor for increasing trust radius
        double factor_down                         # factor for decreasing trust radius
        double avmax                               # max allowed |a|/|v|
        double h_df                                # step size for finite difference Jacobian
        double h_fvv                               # step size for finite difference fvv

    ctypedef struct gsl_multifit_nlinear_type:
        pass

    ctypedef struct gsl_multifit_nlinear_workspace:
        const gsl_multifit_nlinear_type * type
        gsl_multifit_nlinear_fdf * fdf
        gsl_vector * x             # parameter values x
        gsl_vector * f             # residual vector f(x)
        gsl_vector * dx            # step dx
        gsl_vector * g             # gradient J^T f
        gsl_matrix * J             # Jacobian J(x)
        gsl_vector * sqrt_wts_work # sqrt(W)
        gsl_vector * sqrt_wts      # ptr to sqrt_wts_work, or NULL if not using weights
        size_t niter               # number of iterations performed
        gsl_multifit_nlinear_parameters params
        void *state

    gsl_multifit_nlinear_workspace * gsl_multifit_nlinear_alloc (
                                const gsl_multifit_nlinear_type * T,
                                const gsl_multifit_nlinear_parameters * params,
                                size_t n, size_t p)

    void gsl_multifit_nlinear_free (gsl_multifit_nlinear_workspace * w)

    gsl_multifit_nlinear_parameters gsl_multifit_nlinear_default_parameters()

    int gsl_multifit_nlinear_init (const gsl_vector * x,
                               gsl_multifit_nlinear_fdf * fdf,
                               gsl_multifit_nlinear_workspace * w)

    int gsl_multifit_nlinear_winit (const gsl_vector * x,
                                    const gsl_vector * wts,
                                    gsl_multifit_nlinear_fdf * fdf,
                                    gsl_multifit_nlinear_workspace * w)

    int gsl_multifit_nlinear_iterate (gsl_multifit_nlinear_workspace * w)

    double gsl_multifit_nlinear_avratio (const gsl_multifit_nlinear_workspace * w)

    int gsl_multifit_nlinear_driver (
        const size_t maxiter,
        const double xtol,
        const double gtol,
        const double ftol,
        void (*callback)(
            const size_t iter, void *params,
            const gsl_multifit_nlinear_workspace *w
            ),
        void *callback_params,
        int *info,
        gsl_multifit_nlinear_workspace * w
        )

    gsl_matrix * gsl_multifit_nlinear_jac (
        const gsl_multifit_nlinear_workspace * w
        )

    const char * gsl_multifit_nlinear_name (
        const gsl_multifit_nlinear_workspace * w
        )

    gsl_vector * gsl_multifit_nlinear_position (
        const gsl_multifit_nlinear_workspace * w
        )

    gsl_vector * gsl_multifit_nlinear_residual (
        const gsl_multifit_nlinear_workspace * w
        )

    size_t gsl_multifit_nlinear_niter (
        const gsl_multifit_nlinear_workspace * w
        )

    int gsl_multifit_nlinear_rcond (
        double *rcond, const gsl_multifit_nlinear_workspace * w
        )

    const char * gsl_multifit_nlinear_trs_name (
        const gsl_multifit_nlinear_workspace * w
        )

    int gsl_multifit_nlinear_eval_f(
        gsl_multifit_nlinear_fdf *fdf,
        const gsl_vector *x,
        const gsl_vector *swts,
        gsl_vector *y
        )

    int gsl_multifit_nlinear_eval_df(
        const gsl_vector *x,
        const gsl_vector *f,
        const gsl_vector *swts,
        const double h,
        const gsl_multifit_nlinear_fdtype fdtype,
        gsl_multifit_nlinear_fdf *fdf,
        gsl_matrix *df, gsl_vector *work
        )

    int gsl_multifit_nlinear_eval_fvv(
        const double h,
        const gsl_vector *x,
        const gsl_vector *v,
        const gsl_vector *f,
        const gsl_matrix *J,
        const gsl_vector *swts,
        gsl_multifit_nlinear_fdf *fdf,
        gsl_vector *yvv, gsl_vector *work
        )

    # /* covar.c */
    int gsl_multifit_nlinear_covar (
        const gsl_matrix * J, const double epsrel,
        gsl_matrix * covar
        )

    # /* convergence.c */
    int gsl_multifit_nlinear_test (
        const double xtol, const double gtol,
        const double ftol, int *info,
        const gsl_multifit_nlinear_workspace * w
        )

    # /* top-level algorithms */
    const gsl_multifit_nlinear_type * gsl_multifit_nlinear_trust

    # /* trust region subproblem methods */
    const gsl_multifit_nlinear_trs * gsl_multifit_nlinear_trs_lm
    const gsl_multifit_nlinear_trs * gsl_multifit_nlinear_trs_lmaccel
    const gsl_multifit_nlinear_trs * gsl_multifit_nlinear_trs_dogleg
    const gsl_multifit_nlinear_trs * gsl_multifit_nlinear_trs_ddogleg
    const gsl_multifit_nlinear_trs * gsl_multifit_nlinear_trs_subspace2D

    # /* scaling matrix strategies */
    const gsl_multifit_nlinear_scale * gsl_multifit_nlinear_scale_levenberg
    const gsl_multifit_nlinear_scale * gsl_multifit_nlinear_scale_marquardt
    const gsl_multifit_nlinear_scale * gsl_multifit_nlinear_scale_more

    # /* linear solvers */
    const gsl_multifit_nlinear_solver * gsl_multifit_nlinear_solver_cholesky
    const gsl_multifit_nlinear_solver * gsl_multifit_nlinear_solver_qr
    const gsl_multifit_nlinear_solver * gsl_multifit_nlinear_solver_svd

cdef extern from "gsl/gsl_multifit_nlin.h":

    ctypedef struct gsl_multifit_function_fdf:
        int (* f) (gsl_vector * x, void *params, gsl_vector *f)
        int (* df) (gsl_vector *x, void *params, gsl_matrix *df)
        int (* fdf) (gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *df)
        int n  # number of functions
        int p  # number of independent variables
        void * params
    ctypedef struct gsl_multifit_fdfsolver:
        gsl_multifit_function_fdf* fdf
        gsl_vector *x
        gsl_vector *f
        gsl_vector *dx
        void *state
    ctypedef struct gsl_multifit_fdfsolver_type:
        int dummy
    gsl_multifit_fdfsolver* gsl_multifit_fdfsolver_alloc(
        gsl_multifit_fdfsolver_type * T, int n, int p
        )

    int gsl_multifit_fdfsolver_set(
        gsl_multifit_fdfsolver *s,
        gsl_multifit_function_fdf *fdf,
        gsl_vector *x
        )
    int gsl_multifit_fdfsolver_iterate(gsl_multifit_fdfsolver *s)

    void gsl_multifit_fdfsolver_free(gsl_multifit_fdfsolver *s)
    int gsl_multifit_test_delta(gsl_vector * dx, gsl_vector * x,
                                double epsabs, double epsrel)
    int gsl_multifit_test_gradient (gsl_vector * g, double epsabs)
    int gsl_multifit_covar (gsl_matrix * J, double epsrel, gsl_matrix * covar)
    int gsl_multifit_fdfsolver_jac(gsl_multifit_fdfsolver * s, gsl_matrix * J)
    int gsl_multifit_fdfsolver_test (
        gsl_multifit_fdfsolver * s,
        double xtol, double gtol, double ftol, int *info
        )

    # solvers
    gsl_multifit_fdfsolver_type *gsl_multifit_fdfsolver_lmder
    gsl_multifit_fdfsolver_type *gsl_multifit_fdfsolver_lmsder
    gsl_multifit_fdfsolver_type *gsl_multifit_fdfsolver_lmniel

# for v1 of gsl multifit
cdef extern from "gsl/gsl_multimin.h":
    ctypedef struct gsl_multimin_function:
        double (*f) (gsl_vector* x, void* p)
        int n
        void * params

    ctypedef struct gsl_multimin_fminimizer_type:
        char *name
        int size
        int (*alloc) (void *state, int n)
        int (*set) (void *state, gsl_multimin_function * f,
                    gsl_vector * x, double * size,
                    gsl_vector * step_size)
        int (*iterate) (void *state, gsl_multimin_function * f,
                        gsl_vector * x, double * size, double* fval)
        void (*free) (void *state)

    gsl_multimin_fminimizer_type* gsl_multimin_fminimizer_nmsimplex
    gsl_multimin_fminimizer_type* gsl_multimin_fminimizer_nmsimplex2
    gsl_multimin_fminimizer_type* gsl_multimin_fminimizer_nmsimplex2rand

    ctypedef struct gsl_multimin_fminimizer:
        gsl_multimin_fminimizer_type *type
        gsl_multimin_function *f
        double fval
        gsl_vector * x
        double xize
        void *state

    gsl_multimin_fminimizer * gsl_multimin_fminimizer_alloc(
        gsl_multimin_fminimizer_type *T, int n
        )

    int gsl_multimin_fminimizer_set (
        gsl_multimin_fminimizer * s,
        gsl_multimin_function * f,
        gsl_vector * x,
        gsl_vector * step_size
        )

    void gsl_multimin_fminimizer_free(gsl_multimin_fminimizer *s)

    char * gsl_multimin_fminimizer_name (gsl_multimin_fminimizer * s)
    int gsl_multimin_fminimizer_iterate(gsl_multimin_fminimizer *s)
    gsl_vector * gsl_multimin_fminimizer_x (gsl_multimin_fminimizer * s)
    double gsl_multimin_fminimizer_size (gsl_multimin_fminimizer * s)
    double gsl_multimin_fminimizer_minimum (gsl_multimin_fminimizer * s)
    int gsl_multimin_test_size(double size , double epsabs)

# multifit
_valder = None          # ValDer workspace
_p_f = None             # Python fit function
_pyerr = None           # Python exception generated by fit function

# _pyerr is needed because we need to propagate the Python error from
# the fit function through the gsl minimization routines. Normal cython
# exception handling can't be used because because the fit function is
# called by non-cython code. So exceptions that occur inside a fit function
# are stashed in _pyerr and then finally raised just after control returns
# to the cython wrapper from the gsl fitter. The original traceback is
# used so the error message is just what would have come from the fit
# function had cython not been in the way.
#
# Note that gsl-generated errors do not generate exceptions. Rather they
# are stored in multifit's "error" attribute and so can be examined and
# reacted to at run time.

class gsl_multifit(object):
    """ GSL fitter for nonlinear least-squares multidimensional fits.

    :class:`gsl_multifit` is a function-class whose constructor does a
    least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
    vector ``x``.

    :class:`gsl_multifit` is a wrapper for the ``multifit`` GSL routine.

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

            is statisfied. See the GSL documentation for detailed
            definitions of the stopping conditions. Typically one sets
            ``xtol=1/10**d`` where ``d`` is the number of digits of precision
            desired in the result, while ``gtol<<1`` and ``ftol<<1``. Setting
            ``tol=eps`` where ``eps`` is a number is equivalent to setting
            ``tol=(eps,1e-10,1e-10)``. Setting ``tol=(eps1,eps2)`` is
            equivlent to setting ``tol=(eps1,eps2,1e-10)``. Default is
            ``tol=1e-5``. (Note: ``ftol`` option is disabled in some versions
            of  the GSL library.)

        maxit (int): Maximum number of iterations in search for minimum;
                default is 1000.

        alg (str): GSL algorithm to use for minimization. The following
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

        scaler (str): Scaling method used in minimization. The following
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

        solver (str): Method use to solve the linear equations for the
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

        factor_up (float): Factor by which search region is increased
            when a search step is accepted. Values that are too large
            destablize the search; values that are too small slow down
            the search. Default is ``factor_up=3``.

        factor_down (float): Factor by which search region is decreased
            when a search step is rejected. Values that are too large
            destablize the search; values that are too small slow down
            the search. Default is ``factor_up=2``.

        avmax (float): Damping parameter for geodesic acceleration. It
            is the maximum allowed value for the acceleration divided
            by the velocity. Smaller values imply less acceleration.
            Default is ``avmax=0.75``.

    :class:`lsqfit.gsl_multifit` objects have the following attributes.

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
    """
    def __init__(
        self,
        numpy.ndarray[numpy.float_t, ndim=1] x0,
        size_t n,
        object f,
        object tol=(1e-5, 0.0, 0.0),
        unsigned int maxit=1000,
        object alg='lm',
        object solver='qr',
        object scaler='more',
        double factor_up=3.0,
        double factor_down=2.0,
        double avmax=0.75,
        ):
        global _valder, _p_f, _pyerr
        cdef gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust
        cdef gsl_multifit_nlinear_workspace *w
        cdef gsl_multifit_nlinear_fdf fdf
        cdef gsl_multifit_nlinear_parameters fdf_params = \
            gsl_multifit_nlinear_default_parameters()
        cdef gsl_matrix *covar
        cdef gsl_matrix *J
        cdef gsl_vector *x0v
        cdef gsl_multifit_nlinear_trs * _alg
        cdef gsl_multifit_nlinear_scale * _scaler
        cdef gsl_multifit_nlinear_solver * _solver
        cdef size_t i, it, p
        cdef int status, info

        super(gsl_multifit, self).__init__()

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
        self.maxit = maxit
        self.alg = alg
        self.solver = solver
        self.scaler = scaler
        self.factor_up = factor_up
        self.factor_down = factor_down
        self.avmax = avmax
        self.x0 = x0
        self.n = n
        self.error =  None
        self.description = "methods = {}/{}/{}".format(
            self.alg, self.scaler, self.solver
            )
        if self.alg == 'lmaccel':
            self.description += "    avmax = {}".format(self.avmax)
        p = len(x0)

        # choose algorithms fdf_params
        if self.alg == 'lm':
            fdf_params.trs = gsl_multifit_nlinear_trs_lm
        elif self.alg == 'lmaccel':
            fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel
        elif self.alg == 'dogleg':
            fdf_params.trs = gsl_multifit_nlinear_trs_dogleg
        elif self.alg == 'ddogleg':
            fdf_params.trs = gsl_multifit_nlinear_trs_ddogleg
        elif self.alg == 'subspace2D':
            fdf_params.trs = gsl_multifit_nlinear_trs_subspace2D
        else:
            raise ValueError('unkown algorithm ' + str(alg))

        if scaler == 'more':
            fdf_params.scale = gsl_multifit_nlinear_scale_more
        elif scaler == 'levenberg':
            fdf_params.scale = gsl_multifit_nlinear_scale_levenberg
        elif scaler == 'marquardt':
            fdf_params.scale = gsl_multifit_nlinear_scale_marquardt
        else:
            raise ValueError('unkown scaler ' + str(scaler))

        if solver == 'qr':
            fdf_params.solver = gsl_multifit_nlinear_solver_qr
        elif solver == 'cholesky':
            fdf_params.solver = gsl_multifit_nlinear_solver_cholesky
        elif solver == 'svd':
            fdf_params.solver = gsl_multifit_nlinear_solver_svd
        else:
            raise ValueError('unkown solver ' + str(solver))

        # set other parameters
        fdf_params.factor_up = self.factor_up
        fdf_params.factor_down = self.factor_down
        fdf_params.avmax = self.avmax

        # set up fit function
        fdf.f = &_c_f
        fdf.df = &_c_df
        fdf.fvv = NULL
        fdf.p = p
        fdf.n = n
        fdf.params = NULL
        old_p_f = _p_f
        _p_f = f

        # allocate parameter and fitter workspaces
        _valder = gvar.valder(p * [0.0])
        w = gsl_multifit_nlinear_alloc (T, &fdf_params, n, p)

        # initialize and run fit
        x0v = array2vector(x0)
        gsl_multifit_nlinear_init (x0v, &fdf, w)
        status = gsl_multifit_nlinear_driver(maxit, tol[0], tol[1], tol[2],
                                       NULL, NULL, &info, w)
        # check for Python errors; record other errors
        if _pyerr is not None:
            tmp, _pyerr = _pyerr, None
            if hasattr(tmp[1],'with_traceback'):
                raise tmp[1].with_traceback(tmp[2]) # python3
            else:
                raise tmp[0], tmp[1].args, tmp[2]   # python2
        elif status:
            self.error = (status, str(gsl_strerror(status)))

        # identify stopping criterion
        if info >= 0 and info <= 3:
            self.stopping_criterion = info
        elif info == 30:
            self.stopping_criterion = 1
        elif info == 31:
            self.stopping_criterion = 2
        elif info == 29:
            self.stopping_criterion = 3
        else:
            self.stopping_criterion = 0

        # calculate covariance matrix for fit output
        J = gsl_multifit_nlinear_jac(w)
        covar = gsl_matrix_alloc(p, p)
        gsl_multifit_nlinear_covar (J, 0.0, covar)

        # convert results to Python
        self.cov = matrix2array(covar)
        self.x = vector2array(gsl_multifit_nlinear_position(w))
        self.f = vector2array(gsl_multifit_nlinear_residual(w))
        self.J = matrix2array(gsl_multifit_nlinear_jac(w))
        self.nit = gsl_multifit_nlinear_niter(w)
        if status == 11 and self.nit < self.maxit:
            self.error = "gsl_multifit can't improve on starting value; may have converged already."
        if info == 0 and self.error is None:
            self.error = "gsl_multifit didn't converge in {} iterations".format(maxit)
        self.results = None
        # deallocate work areas
        gsl_multifit_nlinear_free(w);
        gsl_matrix_free(covar)
        gsl_vector_free(x0v)
        _p_f = old_p_f


# wrappers for multifit's python function #
cdef int _c_f(gsl_vector* vx, void* params, gsl_vector* vf):
    global _p_f, _pyerr
    cdef numpy.ndarray f
    # can't do numpy.ndarray[object,ndim=1] because might be numbers
    cdef Py_ssize_t i
    cdef Py_ssize_t n = vf.size
    try:
        f = _p_f(vector2array(vx))
        for i in range(n):
            gsl_vector_set(vf, i, f[i])
        return GSL_SUCCESS
    except:
        _pyerr = sys.exc_info()
        return GSL_EBADFUNC

cdef int _c_df(gsl_vector* vx, void* params, gsl_matrix* mJ):
    global _p_f, _pyerr, _valder
    cdef gvar.GVar fi
    cdef gvar.svec fi_d
    cdef numpy.ndarray[object, ndim=1] f
    try:
        f = _p_f(_valder+vector2array(vx))
        gsl_matrix_set_zero(mJ)
        assert len(f[0].cov) == mJ.size2, \
            'covariance matrix mismatch: '+str((len(f[0].cov), mJ.size2))
        for i in range(mJ.size1):
            fi = f[i]
            fi_d = fi.d
            for j in range(fi_d.size):
                gsl_matrix_set(mJ, i, fi_d.v[j].i, fi_d.v[j].v)
        return GSL_SUCCESS
    except:
        _pyerr = sys.exc_info()
        return GSL_EBADFUNC

cdef int _c_fdf(gsl_vector* vx, void* params, gsl_vector* vf, gsl_matrix* mJ):
    global _p_f, _pyerr, _valder
    cdef gvar.GVar fi
    cdef gvar.svec f_i_d
    cdef numpy.ndarray[object, ndim=1] f
    try:
        f = _p_f(_valder+vector2array(vx))
        gsl_matrix_set_zero(mJ)
        assert len(f[0].cov) == mJ.size2, \
            'covariance matrix mismatch: '+str((len(f[0].cov), mJ.size2))
        for i in range(mJ.size1):
            fi = f[i]
            fi_d = fi.d
            gsl_vector_set(vf, i, fi.v)
            for j in range(fi_d.size):
                gsl_matrix_set(mJ, i, fi_d.v[j].i, fi_d.v[j].v)
        return GSL_SUCCESS
    except:
        _pyerr = sys.exc_info()
        return GSL_EBADFUNC

class gsl_v1_multifit(object):
    """ Fitter for nonlinear least-squares multidimensional fits. (GSL v1.)

    :class:`gsl_v1_ multifit` is a function-class whose constructor does a
    least-squares fit by minimizing ``sum_i f_i(x)**2`` as a function of
    vector ``x``.

    :class:`gsl_v1_multifit` is a wrapper for the (older, v1) ``multifit``
    GSL routine (see ``nlin.h``). This package was used in earlier
    versions of :mod:`lsqfit` (<9.0).

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

            is statisfied. See the GSL documentation for detailed
            definitions of the stopping conditions. Typically one sets
            ``xtol=1/10**d`` where ``d`` is the number of digits of precision
            desired in the result, while ``gtol<<1`` and ``ftol<<1``. Setting
            ``tol=eps`` where ``eps`` is a number is equivalent to setting
            ``tol=(eps,1e-10,1e-10)``. Setting ``tol=(eps1,eps2)`` is
            equivlent to setting ``tol=(eps1,eps2,1e-10)``. Default is
            ``tol=1e-5``. (Note: the ``ftol`` option is disabled in some
            versions of  the GSL library.)

        maxit (int): Maximum number of iterations in search for minimum;
                default is 1000.

        alg (str): GSL algorithm to use for minimization. Two options are
            currently available: ``"lmsder"``, the scaled LMDER algorithm
            (default); and ``"lmder"``, the unscaled LMDER algorithm.
            With version 2 of the GSL library, another option is ``"lmniel"``,
            which can be useful when there is much more data than parameters.

        analyzer: Optional function of ``x, [...f_i(x)...], [[..df_ij(x)..]]``
            which is called after each iteration. This can be used to inspect
            intermediate steps in the minimization, if needed.

    :class:`lsqfit.gsl_v1_multifit` objects have the following attributes.

    Attributes:
        x (array): Location of the most recently computed (best) fit point.

        cov (array): Covariance matrix at the minimum point.

        f (callable): Fit function value ``f(x)`` at the minimum in
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
    """

    def __init__(
        self, numpy.ndarray[numpy.float_t, ndim=1] x0, int n,
        object f,
        object tol=1e-5,
        object reltol=None,
        object abstol=None,
        unsigned int maxit=1000,
        object alg='lmsder',
        object analyzer=None
        ):
        global _valder, _p_f, _pyerr
        cdef gsl_multifit_fdfsolver_type *T
        cdef gsl_multifit_fdfsolver *s
        cdef int status, rval, criterion
        cdef Py_ssize_t i, it, p
        cdef gsl_matrix *covar
        cdef gsl_matrix *J
        cdef gsl_vector* x0v
        # cdef numpy.ndarray[numpy.float_t, ndim=1] ans
        super(gsl_v1_multifit, self).__init__()
        # hold onto inputs
        # reltol and abstol are deprecated but still work (for legacy code)
        if reltol is not None:
            tol = (reltol, 1e-10, 1e-10)
        elif abstol is not None:
            tol = (abstol, 1e-10, 1e-10)
        elif type(tol) in [list, tuple]:
            if len(tol) == 1:
                tol = (tol[0], 1e-10, 1e-10)
            elif len(tol) == 2:
                tol = (tol[0], tol[1], 1e-10)
            elif len(tol) == 3:
                pass
            else:
                raise ValueError('bad format for tol: ' + str(tol))
        else:
            tol = (tol, 1e-10, 1e-10)
        self.tol = tol
        self.maxit = maxit
        self.alg = alg
        self.x0 = x0
        self.n = n
        self.error =  None
        p = len(x0)
        covar = gsl_matrix_alloc(p, p)
        if alg=="lmsder":
            T = gsl_multifit_fdfsolver_lmsder
        elif alg=="lmder":
            T = gsl_multifit_fdfsolver_lmder
        elif alg=="lmniel":
            T = gsl_multifit_fdfsolver_lmniel
        else:
            raise ValueError("Unknown algorithm: "+alg)
        self.description = "alg = {}".format(alg)
        cdef gsl_multifit_function_fdf gf
        gf.f = &_c_f
        gf.df = &_c_df
        gf.fdf = &_c_fdf
        gf.p = p
        gf.n = n
        gf.params = NULL
        old_p_f = _p_f
        _p_f = f
        _valder = gvar.valder(p*[0.0])  # workspace
        s = gsl_multifit_fdfsolver_alloc(T, n, p)
        x0v = array2vector(x0)
        gsl_multifit_fdfsolver_set(s, &gf, x0v)
        J = gsl_matrix_alloc(n, p)
        for it in range(1, maxit+1):
            status = gsl_multifit_fdfsolver_iterate(s)
            if _pyerr is not None:
                tmp, _pyerr = _pyerr, None
                if hasattr(tmp[1],'with_traceback'):
                    raise tmp[1].with_traceback(tmp[2]) # python3
                else:
                    raise tmp[0], tmp[1].args, tmp[2]   # python2
            elif status:
                self.error = (status, str(gsl_strerror(status)))
                criterion = 0
                break
            if analyzer is not None:
                gsl_multifit_fdfsolver_jac(s, J)
                analyzer(vector2array(s.x), vector2array(s.f),
                        matrix2array(J))
            if len(tol) == 2:
                rval = gsl_multifit_test_delta(s.dx, s.x, tol[1], tol[0])
                criterion = 1 if rval != GSL_CONTINUE else 0
            else:
                rval = gsl_multifit_fdfsolver_test(s, tol[0], tol[1], tol[2], &criterion)
            if rval != GSL_CONTINUE:
                break

        gsl_multifit_fdfsolver_jac(s, J)
        gsl_multifit_covar(J, 0.0, covar)
        self.cov = matrix2array(covar)
        self.x = vector2array(s.x)
        self.f = vector2array(s.f)
        self.J = matrix2array(J)
        self.nit = it
        self.stopping_criterion = criterion
        if it>=maxit and rval==GSL_CONTINUE:
            self.error ="multifit didn't convernge in %d iterations" % maxit
        self.results = None
        gsl_multifit_fdfsolver_free(s)
        gsl_matrix_free(covar)
        gsl_matrix_free(J)
        gsl_vector_free(x0v)
        _p_f = old_p_f

# multiminex
_p_fs = None                # Python function to be minimized

# Also uses _pyerr for exceptions --- see comment above (multifit)

class gsl_multiminex(object):
    """ Minimizer for multidimensional functions.

    :class:`multiminex` is a function-class whose constructor minimizes a
    multidimensional function ``f(x)`` by varying vector ``x``. This routine
    does *not* use user-supplied information about the gradient of ``f(x)``.

    :class:`multiminex` is a wrapper for the ``multimin`` GSL routine.

    Args:
        x0 (array): Starting point for minimization search.
        f (callable): Function ``f(x)`` to be minimized by varying vector ``x``.
        tol (float): Minimization stops when ``x`` has converged to with
            tolerance ``tol``; default is ``1e-4``.
        maxit (int): Maximum number of iterations in search for minimum;
            default is 1000.
        step (float): Initial step size to use in varying components of ``x``;
            default is 1.
        alg (str): GSL algorithm to use for minimization. Three options are
            currently available: ``"nmsimplex"``, Nelder Mead Simplex
            algorithm; ``"nmsimplex2"``, an improved version of
            ``"nmsimplex"`` (default); and ``"nmsimplex2rand"``, a version
            of ``"nmsimplex2"`` with random shifts in the start position.
        analyzer: Optional function of ``x``, which is called after
            each iteration. This can be used to inspect intermediate steps in
            the minimization, if needed.

    :class:`lsqfit.gsl_multiminex` objects have the following attributes.

    Attributes:
        x (array): Location of the minimum.

        f (float): Value of function ``f(x)`` at the minimum.

        nit (int):  Number of iterations required to find the minimum.

        error (None or str): ``None`` if minimization successful; an error
            message otherwise.
    """
    def __init__(self, numpy.ndarray[numpy.float_t, ndim=1] x0, object f, #):
                 double tol=1e-4, int maxit=1000, step=1.0, alg="nmsimplex2",
                 analyzer=None):
        global _p_fs, _pyerr
        cdef gsl_vector* vx0 = array2vector(x0)
        cdef int dim = vx0.size
        cdef gsl_vector* ss = array2vector(numpy.array(dim*[step]))
        cdef gsl_multimin_function fcn
        cdef int i, status, rval
        cdef Py_ssize_t it
        cdef gsl_multimin_fminimizer* s
        cdef numpy.ndarray[numpy.float_t, ndim=1] x
        cdef double fx

        super(gsl_multiminex, self).__init__()
        old_p_fs = _p_fs
        _p_fs = f
        # preserve inputs
        self.x0 = x0
        self.tol = tol
        self.maxit = maxit
        self.step = step
        self.alg = alg

        fcn.f = &_c_fs
        fcn.n = dim
        fcn.params = NULL
        if alg=="nmsimplex":
            s = gsl_multimin_fminimizer_alloc(
                gsl_multimin_fminimizer_nmsimplex, dim
                )
        elif alg=="nmsimplex2rand":
            s = gsl_multimin_fminimizer_alloc(
                gsl_multimin_fminimizer_nmsimplex2rand, dim
                )
        else:
            s = gsl_multimin_fminimizer_alloc(
                gsl_multimin_fminimizer_nmsimplex2, dim
                )
        gsl_multimin_fminimizer_set(s, &fcn, vx0, ss)

        for it in range(1, maxit+1):
            status = gsl_multimin_fminimizer_iterate(s)
            if _pyerr is not None:
                tmp, _pyerr = _pyerr, None
                if hasattr(tmp[1],'with_traceback'):
                    raise tmp[1].with_traceback(tmp[2]) # python3
                else:
                    raise tmp[0], tmp[1].args, tmp[2]   # python2
            if status:
                self.error = (status, str(gsl_strerror(status)))
                break
            if analyzer is not None:
                x = vector2array(gsl_multimin_fminimizer_x(s))
                fx = gsl_multimin_fminimizer_minimum(s)
                analyzer(x)
            rval = gsl_multimin_test_size(gsl_multimin_fminimizer_size(s), tol)
            if rval!=GSL_CONTINUE:
                break
        self.x = vector2array(gsl_multimin_fminimizer_x(s))
        self.f = gsl_multimin_fminimizer_minimum(s)
        self.nit = it
        self.error = None
        if it>=maxit and rval==GSL_CONTINUE:
            self.error = (
                "MultiMinimizer failed to converge in %d iterations" % maxit
                )
        gsl_vector_free(vx0)
        gsl_vector_free(ss)
        gsl_multimin_fminimizer_free(s)
        _p_fs = old_p_fs

    def __getitem__(self, i):
        return self.x[i]

    def __str__(self):
        return str(self.x)

# wrapper for multiminex's python function #
cdef double _c_fs(gsl_vector* vx, void* p):
    global _p_fs, _pyerr
    if _pyerr is not None:
        return GSL_NAN
    try:
        return _p_fs(vector2array(vx))
    except:
        _pyerr = sys.exc_info()
        return GSL_NAN

# miscellaneous functions
def gammaQ(double a, double x):
    """ Return the normalized incomplete gamma function ``Q(a,x) = 1-P(a,x)``.

    ``Q(a, x) = 1/Gamma(a) * \int_x^\infty dt exp(-t) t ** (a-1) = 1 - P(a, x)``

    Note that ``gammaQ(ndof/2., chi2/2.)`` is the probabilty that one could
    get a ``chi**2`` larger than ``chi2`` with ``ndof`` degrees
    of freedom even if the model used to construct ``chi2`` is correct.
    """
    cdef gsl_sf_result_struct res
    cdef int status
    status = gsl_sf_gamma_inc_Q_e(a, x, &res)
    assert status==GSL_SUCCESS, status
    return res.val
