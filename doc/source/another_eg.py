import numpy as np
import gvar as gv
import lsqfit
import sys
import tee
import pylab as plt
import collections

def f(x, p):                  # fit function
  return sum(pn * (x) ** n for n, pn in enumerate(p))

def main():
	sys_stdout = sys.stdout
	# fit data
	y = gv.gvar([
	  '0.5351(54)', '0.6762(67)', '0.9227(91)', '1.3803(131)', '4.0145(399)'
	  ])
	x = np.array([0.1, 0.3, 0.5, 0.7, 0.95])

	if False:
		p0 = np.ones(5.)              # starting value for chi**2 minimization
		fit = lsqfit.nonlinear_fit(data=(x, y), p0=p0, fcn=f)
		print fit.format(maxline=5)
		make_plot(x, y, fit)
	if True:
		N = 91
		prior = gv.gvar(N * ['0(1)'])   # prior for the fit
		fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
		print fit.format(maxline=10)
		make_plot(x, y, fit)
		inputs = gv.BufferDict(prior=prior)
		for xi, yi in zip(x, y):
			inputs['y(%.2f)' % xi] = yi
		inputs = dict(prior=prior, y=y)
		outputs = dict(p0=fit.p[0])
		print gv.fmt_errorbudget(inputs=inputs, outputs=outputs)
	if False:
		prior = gv.gvar(91 * ['0(1)'])   # prior for the fit
		ymod = y - (f(x, prior) - f(x, prior[:1]))
		priormod = prior[:1]
		fit = lsqfit.nonlinear_fit(data=(x, ymod), prior=priormod, fcn=f)
		print fit.format(maxline=5)
		print lsqfit.wavg(list(ymod) + list(priormod))
		make_plot(x, ymod, fit, 'ymod(x)')
		inputs = dict(prior=prior, y0=y[0], y1=y[1], y2=y[2], y3=y[3], y4=y[4])
		outputs = dict(p0=fit.p[0])
		print gv.fmt_errorbudget(inputs=inputs, outputs=outputs)
	if False:
		prior = gv.gvar(91 * ['0(3)'])   # prior for the fit
		fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
		print fit.format(maxline=5)
		prior = gv.gvar(91 * ['0.0(3)'])   # prior for the fit
		fit = lsqfit.nonlinear_fit(data=(x, y), prior=prior, fcn=f)
		print fit.format(maxline=5)
	if False:
		sp0 = []
		q = []
		for sfit in fit.simulated_fit_iter(20000):
			# print sfit.format(maxline=-1), sfit.p[0] - sfit.pexact[0]
			sp0.append(sfit.pmean[0])
			q.append(sfit.Q)
		print(np.average(sp0), np.std(sp0), np.average(q))

def make_plot(x, y, fit, ylabel='y(x)'):
	plt.errorbar(x, gv.mean(y), gv.sdev(y), fmt='bo')
	x = np.arange(0., 1., 0.01)
	yfit = f(x, fit.p)
	plt.plot(x, gv.mean(yfit), 'k--')
	yplus = gv.mean(yfit) + gv.sdev(yfit)
	yminus = gv.mean(yfit) - gv.sdev(yfit)
	plt.fill_between(x, yminus, yplus, color='0.8')
	plt.xlim(0,1)
	plt.ylim(0.3,1.9)
	plt.xlabel('x')
	plt.ylabel(ylabel)
	plt.show()



if __name__ == '__main__':
	main()