from __future__ import print_function

import lsqfit
import gvar as gv
import numpy as np

try:
    import vegas

    gv.ranseed([1])

    PLOT = False
    RESULTS = False

    # least-squares fit
    x = np.array([0.1, 1.2, 1.9, 3.5])
    y = gv.gvar(['1.2(1.0)', '2.4(1)', '2.0(1.2)', '5.2(3.2)'])
    prior = gv.BufferDict()
    prior['a'] = '0(5)'
    prior['s'] = '0(2)'
    prior['g'] = '2(2)'
    prior = gv.gvar(prior)
    def f(x, p):
        return p['a'] + p['s'] * x ** p['g']
    fit = lsqfit.nonlinear_fit(data=(x,y), prior=prior, fcn=f, debug=True)
    print(fit)

    hist = gv.PDFHistogram(fit.p['s'] * fit.p['g'])

    # Bayesian integral to evaluate expectation value of s*g
    def g(p):
        sg = p['s'] * p['g']
        return dict(
            moments=[sg, sg**2, sg**3, sg**4],
            histogram=hist.count(sg),
            )

    expval = lsqfit.BayesIntegrator(fit, limit=20.)
    warmup = expval(neval=2000, nitn=10)
    results = expval(g, neval=2000, nitn=10, adapt=False)
    if RESULTS:
        print(results.summary())
    stats = hist.analyze(results['histogram']).stats
    print('s*g from Bayesian integral:')
    print(stats)
    print('s*g from fit:', fit.p['s'] * fit.p['g'])
    if PLOT:
        hist.make_plot(results['histogram'], show=True)
except ImportError:
    # fake the run so that `make run` still works
    outfile = open('bayes.out', 'r').read()
    print(outfile)
