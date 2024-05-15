import numpy as np
import gvar as gv
import lsqfit
from vegas import rbatchintegrand
import corner 
import matplotlib.pyplot as plt 
gv.ranseed(123454)
from outputsplitter import log_stdout, unlog_stdout

def main():
    log_stdout('eg3.5a.out')
    x, y = make_data()
    prior = make_prior()

    # nonlinear_fit
    fit = lsqfit.nonlinear_fit(prior=prior, data=(x,y), fcn=fcn)
    print(20 * '-', 'nonlinear_fit')
    print(fit)
    print('p1/p0 =', fit.p[1] / fit.p[0], '   prod(p) =', np.prod(fit.p))
    print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(fit.p[:2])[1,0]))

    # vegas_fit
    neval = 1000
    vfit = lsqfit.vegas_fit(prior=prior, data=(x,y), fcn=fcn, neval=neval) # param=fit.p)
    print('\n' + 20 * '-', 'vegas_fit')
    print(vfit)
    # measure p1/p0 and prod(p)
    @rbatchintegrand
    def g(p):
        return {'p1/p0':p[1] / p[0], 'prod(p)':np.prod(p, axis=0)}
    s = vfit.stats(g, moments=True, histograms=True)
    print('p1/p0 =', s['p1/p0'], '   prod(p) =', s['prod(p)'])
    print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(vfit.p[:2])[1,0]))
    unlog_stdout()
    log_stdout('eg3.5c.out')

    print(vfit.training.summary())
    unlog_stdout()
    print(vfit.p.vegas_mean)
    print()

    log_stdout('eg3.5d.out')
    vfit = lsqfit.vegas_fit(prior=prior, data=(x,y), fcn=fcn, param=fit.p, neval=neval)
    print(20 * '-', 'vegas_fit   (param=fit.p)')
    print(vfit)
    s = vfit.stats(g, moments=True, histograms=True)
    print('p1/p0 =', s['p1/p0'], '   prod(p) =', s['prod(p)'])
    print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(vfit.p[:2])[1,0]))
    unlog_stdout()

    log_stdout('eg3.5b.out')
    # s = vfit.stats(g, moments=True, histograms=True)
    # print(s.summary(True))
    import matplotlib.pyplot as plot
    plot.rcParams['figure.figsize'] = [9.4, 3.4]
    for i, k in enumerate(s.stats):
        if i > 0:
            print()
        print(20 * '-', k)
        print(s.stats[k])
        plot.subplot(1, 2, i + 1)
        plot = s.stats[k].plot_histogram(plot=plot)
        plot.xlabel(k)
    plot.savefig('eg3.5a.png', bbox_inches='tight')
    # plot.show()
    unlog_stdout()
    print(vfit.p.vegas_mean)

    # samples
    wgts, psamples = vfit.sample(nbatch=100_000)
    samples = dict()
    samples['p3'] = psamples[3]
    samples['p1/p0'] = psamples[1] / psamples[0]
    samples['prod(p)'] = np.prod(psamples, axis=0)
    corner.corner(
        data=samples, weights=wgts, range=3 * [0.99], 
        show_titles=True, quantiles=[0.16, 0.5, 0.84],
        plot_datapoints=False, fill_contours=True,
        )
    plt.savefig('eg3.5b.png', bbox_inches='tight')
    plt.show()

    # log_stdout('eg3.5c.out')
    print(vfit.training.summary())
    # unlog_stdout()
    # print(vfit.p.summary())
    old = vfit.logBF
    vfit = lsqfit.vegas_fit(fit=fit, nitn=(6,10), neval=100_000)
    print(old.sdev/vfit.logBF.sdev)
    print(vfit.training.summary())
    unlog_stdout()

    log_stdout('eg3.5e.out')
    prior[1] = gv.gvar('0(20)')
    vfit = lsqfit.vegas_fit(prior=prior, data=(x,y), fcn=fcn, param=fit.p, neval=neval)
    print(20 * '-', 'vegas_fit   (uncorrelated prior)')
    print(vfit)
    s = vfit.stats(g, moments=True, histograms=True)
    print('p1/p0 =', s['p1/p0'], '   prod(p) =', s['prod(p)'])
    print('corr(p0,p1) = {:.2f}'.format(gv.evalcorr(vfit.p[:2])[1,0]))
    unlog_stdout()

def make_data():
    x = np.array([
        4., 2., 1., 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0625
        ])
    y = gv.gvar([
        '0.198(14)', '0.216(15)', '0.184(23)', '0.156(44)', '0.099(49)',
        '0.142(40)', '0.108(32)', '0.065(26)', '0.044(22)', '0.041(19)',
        '0.044(16)'
        ])
    return x, y

def make_prior():
    p = gv.gvar(['0(1)', '0(1)', '0(1)', '0(1)'])
    p[1] = 20 * p[0] + gv.gvar('0.0(1)')        # p[1] correlated with p[0]
    return p

@rbatchintegrand
def fcn(x, p):
    if p.ndim == 2:
        x = x[:, None]
    return (p[0] * (x**2 + p[1] * x)) / (x**2 + x * p[2] + p[3])

if __name__ == '__main__':
    main()