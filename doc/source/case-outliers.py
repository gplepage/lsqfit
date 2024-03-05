from __future__ import print_function
from outputsplitter import log_stdout
import sys
STDOUT = sys.stdout

# NB: Need to run cases (True, False), (False, False) and (False, True)
LSQFIT_ONLY = False
MULTI_W = True

import matplotlib.pyplot as plt
import numpy as np

import gvar as gv
import lsqfit
import vegas

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
        log_stdout('case-outliers-lsq.out')
    elif not MULTI_W:
        log_stdout('case-outliers.out')
        print(20 * '-', 'nonlinear_fit')
    else:
        print(20 * '-', 'nonlinear_fit')
    print(fit)

    # plot data
    plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='o', c='b')

    # plot fit function
    xline = np.linspace(x[0], x[-1], 100)
    yline = fitfcn(xline, fit.p)
    plt.plot(xline, gv.mean(yline), 'k:')
    # yp = gv.mean(yline) + gv.sdev(yline)
    # ym = gv.mean(yline) - gv.sdev(yline)
    # plt.fill_between(xline, yp, ym, color='0.8')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('case-outliers1.png', bbox_inches='tight')
    if LSQFIT_ONLY:
        return

    ### 2) Bayesian integral with modified PDF
    print(20 * '-', 'Bayesian integral fit')
    modpdf = ModifiedPDF(data=(x, y), fitfcn=fitfcn, prior=prior)

    # integrator for expectation values with modified PDF
    modpdf_ev = vegas.PDFIntegrator(fit.p, pdf=modpdf)

    # adapt integrator to pdf
    modpdf_ev(neval=4000, nitn=10)

    # evaluate means and covariances of g(p)
    @vegas.rbatchintegrand
    def g(p):
        return {k:p[k] for k in ['c', 'b', 'w']}

    s = modpdf_ev.stats(g)
    print(s.summary())
    # modpdf_ev.map.show_grid(15)

    if MULTI_W:
        log_stdout('case-outliers-multi.out')

    print('c =', s['c'])
    print('corr(c) =', str(gv.evalcorr(s['c'])).replace('\n', '\n' + 10*' '), '\n')
    print('b =', s['b'])
    print('w =', s['w'], '\n')
    print('logBF =', np.log(s.pdfnorm))
    sys.stdout = STDOUT

    if MULTI_W:
        return

    # add new fit to plot
    yline = fitfcn(xline, dict(c=s['c']))
    plt.plot(xline, gv.mean(yline), 'r--')
    yp = gv.mean(yline) + gv.sdev(yline)
    ym = gv.mean(yline) - gv.sdev(yline)
    plt.fill_between(xline, yp, ym, color='r', alpha=0.2)
    plt.savefig('case-outliers2.png', bbox_inches='tight')
    # plt.show()

@vegas.rbatchintegrand
class ModifiedPDF:
    """ Modified PDF to account for measurement failure. """
    def __init__(self, data, fitfcn, prior):
        x, y = data
        self.fitfcn = fitfcn
        self.prior_pdf = gv.PDF(prior, mode='rbatch')
        # add rbatch index
        self.x = x[:, None]
        self.ymean = gv.mean(y)[:, None]
        self.yvar = gv.var(y)[:, None]

    def __call__(self, p):
        w = p['w']
        b = p['b']

        # modified PDF for data
        fxp = self.fitfcn(self.x, p)
        chi2 = (self.ymean - fxp) ** 2 / self.yvar
        norm = np.sqrt(2 * np.pi * self.yvar)
        y_pdf = np.exp(-chi2 / 2) / norm
        yb_pdf = np.exp(-chi2 / (2 * b**2)) / (b * norm)
        # product over PDFs for each y[i]
        data_pdf = np.prod((1 - w) * y_pdf + w * yb_pdf, axis=0)

        # multiply by prior PDF
        return data_pdf * self.prior_pdf(p)

def fitfcn(x, p):
    c = p['c']
    return c[0] + c[1] * x #** c[2]

def make_prior():
    prior = gv.BufferDict(c=gv.gvar(['0(5)', '0(5)']))
    if LSQFIT_ONLY:
        return prior
    prior['gb(b)'] = gv.BufferDict.uniform('gb', 5., 20.)
    if MULTI_W:
        prior['gw(w)'] = gv.BufferDict.uniform('gw', 0., 1., shape=19)
    else:
        prior['gw(w)'] = gv.BufferDict.uniform('gw', 0., 1.)
    return prior


if __name__ == '__main__':
    gv.ranseed([12345])
    main()