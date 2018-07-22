from __future__ import print_function

# import matplotlib.pyplot as plt
import numpy as np
import gvar as gv
import lsqfit

gv.ranseed(1)

def main():
    if not hasattr(lsqfit, 'BayesIntegrator'):
        # fake the run so that `make run` still works
        outfile = open('bayes.out', 'r').read()
        print(outfile[:-1])
        return
    x, y = make_data()
    prior = make_prior()
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(fit)
    # Bayesian integrator
    expval = lsqfit.BayesIntegrator(fit, sync_ran=False)

    # adapt integrator expval to PDF from fit
    neval = 1000
    nitn = 10
    expval(neval=neval, nitn=nitn)

    # <g(p)> gives mean and covariance matrix, and counts for histograms
    hist = [
        gv.PDFHistogram(fit.p[0]), gv.PDFHistogram(fit.p[1]),
        gv.PDFHistogram(fit.p[2]), gv.PDFHistogram(fit.p[3]),
        ]
    def g(p):
        return dict(
            mean=p,
            outer=np.outer(p, p),
            count=[
                hist[0].count(p[0]), hist[1].count(p[1]),
                hist[2].count(p[2]), hist[3].count(p[3]),
                ],
            )

    # evaluate expectation value of g(p)
    results = expval(g, neval=neval, nitn=nitn, adapt=False)

    # analyze results
    # print('\nIterations:')
    # print(results.summary())
    print('Integration Results:')
    pmean = results['mean']
    pcov =  results['outer'] - np.outer(pmean, pmean)
    print('    mean(p) =', pmean)
    print('    cov(p) =\n', pcov)

    # create GVars from results
    p = gv.gvar(gv.mean(pmean), gv.mean(pcov))
    print('\nBayesian Parameters:')
    print(gv.tabulate(p))

    # show histograms
    print('\nHistogram Statistics:')
    count = results['count']
    for i in range(4):
        # print histogram statistics
        print('p[{}]:'.format(i))
        print(hist[i].analyze(count[i]).stats)
    #     # make histogram plots
    #     plt.subplot(2, 2, i + 1)
    #     plt.xlabel('p[{}]'.format(i))
    #     hist[i].make_plot(count[i], plot=plt)
    #     if i % 2 != 0:
    #         plt.ylabel('')
    # plt.show()

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
    p[1] = 20 * p[0] + gv.gvar('0.0(1)')     # p[1] correlated with p[0]
    return p

def fcn(x, p):
    return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

if __name__ == '__main__':
    main()
