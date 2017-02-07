from __future__ import print_function

import numpy as np
import gvar as gv
import lsqfit

def main():
    x, y = make_data()
    prior = make_prior(100)             # 100 exponential terms in all
    p0 = None
    for nexp in range(1, 6):
        # marginalize the last 100 - nexp terms (in ymod_prior)
        fit_prior = gv.BufferDict()     # part of prior used in fit
        ymod_prior = gv.BufferDict()    # part of prior absorbed in ymod
        for k in prior:
            fit_prior[k] = prior[k][:nexp]
            ymod_prior[k] = prior[k][nexp:]
        ymod = y - fcn(x, ymod_prior)   # remove temrs in ymod_prior

        # fit modified data with just nexp terms (in fit_prior)
        fit = lsqfit.nonlinear_fit(
            data=(x, ymod), prior=fit_prior, fcn=fcn, p0=p0, tol=1e-15, svdcut=1e-4,
            )

        # print fit information
        print('************************************* nexp =',nexp)
        print(fit.format(True))
        p0 = fit.pmean

    # print summary information and error budget
    E = fit.p['E']                      # best-fit parameters
    a = fit.p['a']
    # outputs = {
    #     'E1/E0':E[1] / E[0], 'E2/E0':E[2] / E[0],
    #     'a1/a0':a[1] / a[0], 'a2/a0':a[2] / a[0]
    #     }
    # inputs = {
    #     'E prior':prior['E'], 'a prior':prior['a'],
    #     'svd cut':fit.svdcorrection,
    #     }
    outputs = gv.BufferDict()
    outputs['E2/E0'] = E[2] / E[0]
    outputs['E1/E0'] = E[1] / E[0]
    outputs['a2/a0'] = a[2] / a[0]
    outputs['a1/a0'] = a[1] / a[0]
    inputs = gv.BufferDict()
    inputs['E prior'] = prior['E']
    inputs['svd cut'] = fit.svdcorrection
    inputs['a prior'] = prior['a']
    print(fit.fmt_values(outputs))
    print(fit.fmt_errorbudget(outputs, inputs))

def fcn(x,p):
    a = p['a']       # array of a[i]s
    E = p['E']       # array of E[i]s
    return np.sum(ai * np.exp(-Ei*x) for ai, Ei in zip(a, E))

def make_prior(nexp):
    prior = gv.BufferDict()
    prior['a'] = gv.gvar(nexp * ['0.5(5)'])
    dE = gv.gvar(nexp * ['1.0(1)'])
    prior['E'] = np.cumsum(dE)
    return prior

def make_data():
    x = np.array([ 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2, 2.4, 2.6])
    y = np.array([
        0.2740471001620033,  0.2056894154005132,  0.158389402324004 ,
        0.1241967645280511,  0.0986901274726867,  0.0792134506060024,
        0.0640743982173861,  0.052143504367789 ,  0.0426383022456816,
        ])
    return x, y

if __name__ == '__main__':
    main()
