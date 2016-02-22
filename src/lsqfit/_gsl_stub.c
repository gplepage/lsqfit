#include "_gsl_stub.h"

#if GSL_MAJOR_VERSION < 2
int gsl_multifit_fdfsolver_jac(gsl_multifit_fdfsolver * s, gsl_matrix * J)
    {
        return gsl_matrix_memcpy(J, s->J);
    }

// dummies, for compilation:

int gsl_multifit_fdfsolver_test (
        gsl_multifit_fdfsolver * s,
        double xtol, double gtol, double ftol, int *info
        )
    {
        return 0;
    }
gsl_multifit_fdfsolver_type *gsl_multifit_fdfsolver_lmniel = 0;
#endif