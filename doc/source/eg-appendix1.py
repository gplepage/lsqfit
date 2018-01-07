import numpy as np
import gvar as gv
import lsqfit
import sys
import tee
import pylab as plt
import collections

# editing info:
# 1b.out - cut 16-87
# 1d.out - cut 4-end
# 1e.out - cut 4-end

MAKE_PLOTS = True # False
N = 91
STDOUT = sys.stdout
OUTDIR = ''
# OUTDIR = 'tmp/'

def main():
	bad_analysis()
	good_analysis()
	marginalized_analysis()
	prior_analysis()
	test_fit()

def f(x, p):                  # fit function
  return sum(pn * (x) ** n for n, pn in enumerate(p))

def make_data():
	y = gv.gvar([
	  '0.5351(54)', '0.6762(67)', '0.9227(91)', '1.3803(131)', '4.0145(399)'
	  ])
	x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])
	return x, y

def bad_analysis():
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1a.out', 'w'))
	x, y = make_data()
	p0 = np.ones(5, float)              # starting value for chi**2 minimization
	fit = lsqfit.nonlinear_fit(data=(x, y), p0=p0, fcn=f)
	print fit.format(maxline=True)
	make_plot(x, y, fit, name='eg-appendix1a')
	return fit

def good_analysis(plot=MAKE_PLOTS):
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1b.out', 'w'))
	x, y = make_data()
	prior = gv.gvar(N * ['0(1)'])   # prior for the fit
	fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
	print fit.format(maxline=True)
	if plot:
		make_plot(x, y, fit, name='eg-appendix1b')
	inputs = gv.BufferDict(prior=prior)
	for xi, yi in zip(x, y):
		inputs['y(%.2f)' % xi] = yi
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1g.out', 'w'))
	inputs = dict(prior=prior, y=y)
	outputs = dict(p0=fit.p[0])
	print gv.fmt_errorbudget(inputs=inputs, outputs=outputs)
	return fit

def marginalized_analysis():
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1c.out', 'w'))
	x, y = make_data()
	prior = gv.gvar(91 * ['0(1)'])   # prior for the fit
	ymod = y - (f(x, prior) - f(x, prior[:1]))
	priormod = prior[:1]
	fit = lsqfit.nonlinear_fit(data=(x, ymod), prior=priormod, fcn=f)
	print fit.format(maxline=True)
	sys.stdout = STDOUT
	print lsqfit.wavg(list(ymod) + list(priormod))
	make_plot(x, ymod, fit, 'ymod(x)', name='eg-appendix1c')
	inputs = dict(prior=prior, y0=y[0], y1=y[1], y2=y[2], y3=y[3], y4=y[4])
	outputs = dict(p0=fit.p[0])
	print gv.fmt_errorbudget(inputs=inputs, outputs=outputs)
	return fit

def prior_analysis():
	x, y = make_data()
	# loose prior
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1d.out', 'w'))
	prior = gv.gvar(91 * ['0(3)'])   # prior for the fit
	fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
	print fit.format(maxline=True)
	# really loose prior
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1h.out', 'w'))
	prior = gv.gvar(91 * ['0(20)'])   # prior for the fit
	fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
	print fit.format(maxline=True)
	make_plot(x, y, fit, xmax=0.96, name='eg-appendix1d')
	# tight prior
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1e.out', 'w'))
	prior = gv.gvar(91 * ['0.0(3)'])   # prior for the fit
	fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
	print fit.format(maxline=True)

def test_fit():
	sys.stdout = STDOUT
	fit = good_analysis(plot=False)
	sp0 = []
	q = []
	sys.stdout = tee.tee(STDOUT, open(OUTDIR+'eg-appendix1f.out', 'w'))
	for sfit in fit.simulated_fit_iter(20):
		# print sfit.format(maxline=-1), sfit.p[0] - sfit.pexact[0]
		sp0.append(sfit.pmean[0])
		q.append(sfit.Q)
	print(np.average(sp0), np.std(sp0), np.average(q))

NPLT = 0
def make_plot(x, y, fit, ylabel='y(x)', xmax=1.0, name='appendix1'):
	global NPLT
	if not MAKE_PLOTS:
		return
	plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='bo')
	x = np.arange(0., xmax, 0.01)
	yfit = f(x, fit.p)
	plt.plot(x, gv.mean(yfit), 'k--')
	yplus = gv.mean(yfit) + gv.sdev(yfit)
	yminus = gv.mean(yfit) - gv.sdev(yfit)
	plt.fill_between(x, yminus, yplus, color='0.8')
	plt.xlim(0,1)
	plt.ylim(0.3,1.9)
	plt.xlabel('x')
	NPLT += 1
	plt.ylabel(ylabel)
	plt.savefig(name + '.png', bbox_inches='tight')
	plt.show()


if __name__ == '__main__':
	main()