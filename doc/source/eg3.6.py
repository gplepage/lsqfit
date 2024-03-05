import gvar as gv
import lsqfit
import numpy as np
from outputsplitter import log_stdout, unlog_stdout

gv.ranseed(123)

y = gv.gvar([
    '-0.17(20)', '-0.03(20)', '-0.39(20)', '0.10(20)', '-0.03(20)',
    '0.06(20)', '-0.23(20)', '-0.23(20)', '-0.15(20)', '-0.01(20)',
    '-0.12(20)', '0.05(20)', '-0.09(20)', '-0.36(20)', '0.09(20)',
    '-0.07(20)', '-0.31(20)', '0.12(20)', '0.11(20)', '0.13(20)'
    ])

log_stdout('eg3.6a.out')

# nonlinear_fit
prior = gv.BufferDict()
prior['f(a)'] = gv.BufferDict.uniform('f', 0, 0.04)

def fcn(p, N=len(y)):
    return N * [p['a']]

fit = lsqfit.nonlinear_fit(prior=prior, data=y, fcn=fcn)
print(20 * '-', 'nonlinear_fit')
print(fit)
print('a =', fit.p['a'])

# Nbs bootstrap copies
Nbs = 1000
a = []
for bsfit in fit.bootstrapped_fit_iter(Nbs):
    a.append(bsfit.p['a'].mean)
avg_a = gv.dataset.avg_data(a, spread=True)
print('\n' + 20 * '-', 'bootstrap')
print('a =', avg_a)
counts,bins = np.histogram(a, density=True)
s = gv.PDFStatistics(histogram=(bins, counts))
print(s)
plot = s.plot_histogram()
plot.xlabel('a')
plot.savefig('eg3.6a.png', bbox_inches='tight')
plot.show()

unlog_stdout()
print()
log_stdout('eg3.6b.out')

# vegas_fit
print(20 * '-', 'vegas_fit')
print(lsqfit.vegas_fit(fit=fit))

unlog_stdout()

