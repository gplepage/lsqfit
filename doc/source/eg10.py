"""
svd and inadequate statistics

"""
# Copyright (c) 2019 G. Peter Lepage.
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

import numpy as np
import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
from collections import OrderedDict
import tee
import sys

SEED = (1573116538, 1842119200, 1626997435)
NSAMPLE = 5

def main():
    gv.ranseed(SEED)
    y = exact(NSAMPLE)
    ysamples = [yi for yi in gv.raniter(y, n=NSAMPLE)]
    # above code (don't comment it out) generates the following
    ysamples = [
        [0.0092441016, 0.0068974057, 0.0051480509, 0.0038431422, 0.0028690492], 
        [0.0092477405, 0.0069030565, 0.0051531383, 0.0038455855, 0.0028700587], 
        [0.0092558569, 0.0069102437, 0.0051596569, 0.0038514537, 0.0028749153], 
        [0.0092294581, 0.0068865156, 0.0051395262, 0.003835656, 0.0028630454], 
        [0.009240534, 0.0068961523, 0.0051480046, 0.0038424661, 0.0028675632],
        ]
    dstr = '['
    for yi in ysamples:
        dstr += ('[' + len(yi) * '{:10.8g},' + '],').format(*yi)
    dstr += ']'
    ysamples = eval(dstr)
    print(np.array(ysamples).tolist())
    s = gv.dataset.svd_diagnosis(ysamples)
    # s.plot_ratio(show=True)
    y = s.avgdata
    x = np.array([15., 16., 17., 18., 19.])
    def f(p):
        return p['a'] * gv.exp(- p['b'] * x)
    prior = gv.gvar(dict(a='0.75(5)', b='0.30(3)'))
    sys_stdout = sys.stdout

    sys.stdout = tee.tee(sys_stdout, open('eg10a.out', 'w'))
    fit = lsqfit.nonlinear_fit(data=y, fcn=f, prior=prior, svdcut=0.0)
    print(fit)

    sys.stdout = tee.tee(sys_stdout, open('eg10b.out', 'w'))
    fit = lsqfit.nonlinear_fit(data=y, fcn=f, prior=prior, svdcut=s.svdcut)
    print(fit)

    sys.stdout = tee.tee(sys_stdout, open('eg10c.out', 'w'))
    fit = lsqfit.nonlinear_fit(data=y, fcn=f, prior=prior, svdcut=s.svdcut, add_svdnoise=True)
    print(fit)

    sys.stdout = tee.tee(sys_stdout, open('eg10d.out', 'w'))
    yex = gv.gvar(gv.mean(y), gv.evalcov(exact(1.)))
    fit = lsqfit.nonlinear_fit(data=yex, fcn=f, prior=prior, svdcut=0)
    print(fit)
    # fit.plot_residuals().show()

    sys.stdout = tee.tee(sys_stdout, open('eg10e.out', 'w'))
    fit = lsqfit.nonlinear_fit(data=y, fcn=f, prior=prior, svdcut=s.svdcut)
    print(fit)
    print('\n================ Add noise to prior, SVD')
    noisyfit = lsqfit.nonlinear_fit(
        data=y, prior=prior, fcn=f, svdcut=s.svdcut,
        add_svdnoise=True, add_priornoise=True
        )
    print(noisyfit.format(True))
    # save figures
    fit.qqplot_residuals(plot=plt).savefig('eg10e1.png', bbox_inches='tight')
    plt.cla()
    noisyfit.qqplot_residuals(plot=plt).savefig('eg10e2.png', bbox_inches='tight')

def exact(nsample):
    y = gv.gvar(
        [0.009258421671615743, 0.006910286297586342, 0.005157944135538242, 0.003850506172983167, 0.0028740686573919486],
        nsample ** 0.5 * np.array(
            [6.113763211010068e-06, 4.769788955316411e-06, 3.7353429036214163e-06, 2.8924400872983105e-06, 2.246929325641689e-06]
            )
        )
    y = gv.correlate(
        y, [[1.0, 0.9784345395506512, 0.9687378755868289, 0.9624851858548873, 0.9537466741381387], [0.9784345395506512, 1.0000000000000002, 0.9816755120154124, 0.9739582290580883, 0.9626760365167806], [0.9687378755868289, 0.9816755120154124, 1.0, 0.9817294853744025, 0.9718940239158071], [0.9624851858548873, 0.9739582290580883, 0.9817294853744025, 1.0, 0.9823139467782855], [0.9537466741381387, 0.9626760365167806, 0.9718940239158071, 0.9823139467782855, 1.0]]
        )
    return y

if __name__ == '__main__':
    main()