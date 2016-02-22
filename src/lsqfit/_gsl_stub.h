#include <gsl/gsl_version.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_matrix_double.h>

#if GSL_MAJOR_VERSION < 2
int gsl_multifit_fdfsolver_jac(gsl_multifit_fdfsolver * s, gsl_matrix * J);
extern gsl_multifit_fdfsolver_type *gsl_multifit_fdfsolver_lmniel;
#endif
