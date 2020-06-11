from __future__ import print_function
import tee
import sys
STDOUT = sys.stdout

# NB: Need to run cases (True, False), (False, False) and (False, True)
LSQFIT_ONLY = False
MULTI_W = True

import matplotlib.pyplot as plt
import numpy as np

import gvar as gv
import lsqfit

def main():
    ### 1) least-squares fit to the data
    x = np.array([
        0.2, 0.4, 0.6, 0.8, 1.,
        1.2, 1.4, 1.6, 1.8, 2.,
        2.2, 2.4, 2.6, 2.8, 3.,
        3.2, 3.4, 3.6, 3.8
        ])
    y = gv.gvar([
        '0.38(20)', '2.89(20)', '0.85(20)', '0.59(20)', '2.88(20)',
        '1.44(20)', '0.73(20)', '1.23(20)', '1.68(20)', '1.36(20)',
        '1.51(20)', '1.73(20)', '2.16(20)', '1.85(20)', '2.00(20)',
        '2.11(20)', '2.75(20)', '0.86(20)', '2.73(20)'
        ])
    prior = make_prior()
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=fitfcn)
    if LSQFIT_ONLY:
        sys.stdout = tee.tee(STDOUT, open('case-outliers-lsq.out', 'w'))
    elif not MULTI_W:
        sys.stdout = tee.tee(STDOUT, open('case-outliers.out', 'w'))
    print(fit)

    # plot data
    plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='o', c='b')

    # plot fit function
    xline = np.linspace(x[0], x[-1], 100)
    yline = fitfcn(xline, fit.p)
    plt.plot(xline, gv.mean(yline), 'k:')
    yp = gv.mean(yline) + gv.sdev(yline)
    ym = gv.mean(yline) - gv.sdev(yline)
    plt.fill_between(xline, yp, ym, color='0.8')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('case-outliers1.png', bbox_inches='tight')
    if LSQFIT_ONLY:
        return

    ### 2) Bayesian integral with modified PDF
    pdf = ModifiedPDF(data=(x, y), fcn=fitfcn, prior=prior)

    # integrator for expectation values with modified PDF
    expval = lsqfit.BayesIntegrator(fit, pdf=pdf)

    # adapt integrator to pdf
    expval(neval=1000, nitn=15)

    # evaluate expectation value of g(p)
    def g(p):
        w = p['w']
        c = p['c']
        return dict(w=[w, w**2], mean=c, outer=np.outer(c,c))

    results = expval(g, neval=1000, nitn=15, adapt=False)
    print(results.summary())
    # expval.map.show_grid(15)

    if MULTI_W:
        sys.stdout = tee.tee(STDOUT, open('case-outliers-multi.out', 'w'))

    # parameters c[i]
    mean = results['mean']
    cov = results['outer'] - np.outer(mean, mean)
    c = mean + gv.gvar(np.zeros(mean.shape), gv.mean(cov))
    print('c =', c)
    print(
        'corr(c) =',
        np.array2string(gv.evalcorr(c), prefix=10 * ' '),
        '\n',
        )

    # parameter w
    wmean, w2mean = results['w']
    wsdev = gv.mean(w2mean - wmean ** 2) ** 0.5
    w = wmean + gv.gvar(np.zeros(np.shape(wmean)), wsdev)
    print('w =', w, '\n')

    # Bayes Factor
    print('logBF =', np.log(results.norm))
    sys.stdout = STDOUT

    if MULTI_W:
        return

    # add new fit to plot
    yline = fitfcn(xline, dict(c=c))
    plt.plot(xline, gv.mean(yline), 'r--')
    yp = gv.mean(yline) + gv.sdev(yline)
    ym = gv.mean(yline) - gv.sdev(yline)
    plt.fill_between(xline, yp, ym, color='r', alpha=0.2)
    plt.savefig('case-outliers2.png', bbox_inches='tight')
    # plt.show()

class ModifiedPDF:
    """ Modified PDF to account for measurement failure. """

    def __init__(self, data, fcn, prior):
        self.x, self.y = data
        self.fcn = fcn
        self.prior = prior

    def __call__(self, p):
        w = p['w']
        y_fx = self.y - self.fcn(self.x, p)
        data_pdf1 = self.gaussian_pdf(y_fx, 1.)
        data_pdf2 = self.gaussian_pdf(y_fx, 10.)
        prior_pdf = self.gaussian_pdf(
            p.buf[:len(self.prior.buf)] - self.prior.buf
            )
        return np.prod((1. - w) * data_pdf1 + w * data_pdf2) * np.prod(prior_pdf)

    @staticmethod
    def gaussian_pdf(x, f=1.):
        xmean = gv.mean(x)
        xvar = gv.var(x) * f ** 2
        return gv.exp(-xmean ** 2 / 2. /xvar) / gv.sqrt(2 * np.pi * xvar)


def fitfcn(x, p):
    c = p['c']
    return c[0] + c[1] * x #** c[2]

def make_prior():
    prior = gv.BufferDict(c=gv.gvar(['0(5)', '0(5)']))
    if LSQFIT_ONLY:
        return prior
    if MULTI_W:
        prior['unif(w)'] = gv.BufferDict.uniform('unif', 0., 1., shape=19)
    else:
        prior['unif(w)'] = gv.BufferDict.uniform('unif', 0., 1.)
    return prior


if __name__ == '__main__':
    gv.ranseed([12345])
    main()