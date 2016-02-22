"""
pendulum.py --- illustrates a more complex fitting problem.

This codes fits data for the location of a pendulum at different times
to determine the pendulum's parameters (especially the ratio g/l where
g is the acceleration due to gravity and l is the pendulum's length).
The fit function uses gvar.ode.Integrator to integrate the
equation of motion (Newton's Law) for a pendulum.
"""

# Created by G. Peter Lepage (Cornell University) on 2014-04-28.
# Copyright (c) 2014 G. Peter Lepage.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

from __future__ import print_function   # makes this work for python2 and 3

MAKE_PLOT = False

import math

import numpy as np
try:
    import pylab as plt
except:
    MAKE_PLOT = False

import gvar as gv
import lsqfit

def main():
    # pendulum data exhibits experimental error in ability to measure theta
    t = [ 0.1, 0.2, 0.3, 0.4,  0.5, 0.6,  0.7,  0.8,  0.9, 1.]
    theta = gv.gvar([
        '1.477(79)', '0.791(79)', '-0.046(79)', '-0.852(79)',
        '-1.523(79)', '-1.647(79)', '-1.216(79)', '-0.810(79)',
        '0.185(79)', '0.832(79)'
        ])

    # prior: assume experimental error in ability to specify theta(0)
    prior = gv.BufferDict()
    prior['g/l'] = (2 * math.pi) ** 2 * gv.gvar(1, 0.1)
    prior['theta(0)'] = gv.gvar(math.pi / 2., 0.05)

    # fit function: use class Pendulum object to integrate pendulum motion
    def fitfcn(p, t=t):
        pendulum = Pendulum(p['g/l'])
        return pendulum(p['theta(0)'], t)

    # do the fit and print results
    fit = lsqfit.nonlinear_fit(data=theta, prior=prior, fcn=fitfcn)
    print(fit.format(maxline=True))
    print('fit/exact for (g/l) =', fit.p['g/l'] / (2*math.pi) ** 2)
    print('fit/exact for theta(0) =', fit.p['theta(0)'] / (math.pi / 2.))

    if MAKE_PLOT:
        # make figure (saved to file pendulum.pdf)
        plt.figure(figsize=(4,3))
        # start plot with data
        plt.errorbar(
            x=t, y=gv.mean(theta), yerr=gv.sdev(theta),
            fmt='k.',
            )
        # use best-fit function to add smooth curve for 100 points
        t = np.linspace(0., 1.1, 100)
        th = fitfcn(fit.p, t)
        show_plot(t, th)

class Pendulum(object):
    """ Integrator for pendulum motion.

    Input parameters are:
        g/l .... where g is acceleration due to gravity and l the length
        tol .... precision of numerical integration of ODE
    """
    def __init__(self, g_l, tol=1e-4):
        self.g_l = g_l
        self.odeint = gv.ode.Integrator(deriv=self.deriv, tol=tol)

    def __call__(self, theta0, t_array):
        """ Calculate pendulum angle theta for every t in t_array.

        Assumes that the pendulum is released at time t=0
        from angle theta0 with no initial velocity. Returns
        an array containing theta(t) for every t in t_array.
        """
        # initial values
        t0 = 0
        y0 = [theta0, 0.0]              # theta and dtheta/dt

        # solution
        y = self.odeint.solution(t0, y0)
        return [y(t)[0] for t in t_array]

    def deriv(self, t, y, data=None):
        " Calculate [dtheta/dt, d2theta/dt2] from [theta, dtheta/dt]."
        theta, dtheta_dt = y
        return np.array([dtheta_dt, - self.g_l * gv.sin(theta)])


# def make_pendulum_data():
#     """ Make pendulum data (t, theta) for fitting code. """
#     # make exact data
#     g_l = (2 * math.pi) ** 2
#     theta0 = math.pi / 2.
#     pendulum = Pendulum(g_l)
#     t_array = np.linspace(0., 1., 11)
#     th_array = pendulum(theta0, t_array)

#     # add noise
#     ran = gv.gvar(0, math.pi/40.)
#     th_data = gv.mean(th_array) + [gv.gvar(ran(), ran.sdev) for i in range(len(th_array))]

#     # print( t_array)
#     # print([str(th) for th in th_data])
#     # show_plot(t_array, th_array)
#     return t_array, th_data

def show_plot(t_array, th_array):
    """ Display theta vs t plot. """
    th_mean = gv.mean(th_array)
    th_sdev = gv.sdev(th_array)
    thp = th_mean + th_sdev
    thm = th_mean - th_sdev
    plt.fill_between(t_array, thp, thm, color='0.8')
    plt.plot(t_array, th_mean, linewidth=0.5)
    plt.xlabel('$t$')
    plt.ylabel(r'$\theta(t)$')
    plt.savefig('pendulum.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()