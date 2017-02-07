"""
Creates a fit-test script using files XXX.txt from NIST, which
describe in detail NIST's test. These are test cases NIST provides
nonlinear fitters. See http://www.itl.nist.gov/div898/strd/nls/nls_main.shtml.

Methods:
    make_script(): Returns the full testing script, with all the tests.

    make_fcn(filename): Returns the fit-test function corresponding to
        file ``filename``.

The fit-test functions use one or the other of the starting values provided by
NIST. The one used in the fit-test function is determined by the flag
USE_2ND_STARTING_VALUE. The default fitter (but not others) fails with the 1st
starting value in mgh10, but is fine with the other. The default fitter works
for the 2nd starting value (and others), and for both starting values for all
other tests.

Have tested various fitters (2016-12-29) and find that all work for all
problems using the 2nd starting value. That includes:

    gsl_multifit: lm/more/qr, lmaccel, subspace2D, dogleg, ddogleg
    gsl_v1_multifit: lmsder, lmder
    scipy_least_squares: trf, lm

They all agree with each other and with NIST's "certified values".
"""
import re
import gvar as gv
import numpy as np

USE_2ND_STARTING_VALUE = True

def main():
    if False:
        print make_fcn('enso.txt')
    if True:
        text = make_script()
        open('_nist.py', 'w').write(text)

def make_script():
    """ Return the full testing script, with all the tests. """
    # tests
    easy = [
        'misra1a', 'chwirut2', 'chwirut1', 'lanczos3',
        'gauss1', 'gauss2', 'danwood', 'misra1b',
        ]
    medium = [
        'kirby2', 'hahn1', 'nelson', 'mgh17', 'lanczos1',
        'lanczos2', 'gauss3', 'misra1c', 'misra1d', 'roszman1', 'enso',
        ]
    hard = [
        'mgh09', 'thurber', 'boxbod', 'rat42', 'mgh10',
        'eckerle4', 'rat43', 'bennett5',
        ]

    # beginning material
    text = \
"""
from __future__ import print_function

import gvar as gv
import numpy as np
import lsqfit

log = np.log
exp = np.exp
arctan = np.arctan
cos = np.cos
sin = np.sin
pi = np.pi

"""
    if USE_2ND_STARTING_VALUE:
        text += '# 2nd starting values\n\n'
    else:
        text += '# 1st starting values\n\n'
    # main() program
    text += 'def main():\n'
    text += '    # easy\n'
    for n in easy:
        text += '    ' + n + '()\n'
    text += '\n    # medium\n'
    for n in medium:
        text += '    ' + n + '()\n'
    text += '\n    # hard\n'
    for n in hard:
        text += '    ' + n + '()\n'

    # add test-fit functions
    for n in easy:
        text += '\n'
        text += make_fcn(n + '.txt')
    for n in medium:
        text += '\n'
        text += make_fcn(n + '.txt')
    for n in hard:
        text += '\n'
        text += make_fcn(n + '.txt')

    # ending material
    text += \
"""

if __name__ == '__main__':
    main()
"""
    return text

# answers that differ in format from NIST values, but not value
answers = {
    'lanczos1': '[0.0950994 +- 5.3e-11 0.999997 +- 2.7e-10 0.8607 +- 1.4e-10 3 +- 3.3e-10\n 1.5576 +- 1.9e-10 5 +- 1.1e-10]',
    }

def make_fcn(filename):
    """ Return the fit-test function corresponding to file ``filename``.  """
    # collect lines and parse first several lines for fit info
    with open(filename, 'r') as ifile:
        lines = ifile.readlines()
    name = filename.split('.')[0]
    _certified_values = re.compile('.*Certified Values\s*\(lines\s*([0-9]*)\s*to\s*([0-9]*)\)')
    _data = re.compile('.*Data\s*\(lines\s*([0-9]*)\s*to\s*([0-9]*)\)')
    _model = re.compile('^Model:')
    values = None
    data = None
    fcn = None
    for i,line in enumerate(lines):
        m = _certified_values.match(line)
        if m is not None:
            values = (int(m.group(1)), int(m.group(2)))
        m = _data.match(line)
        if m is not None:
            data = (int(m.group(1)), int(m.group(2)))
        m = _model.match(line)
        if m is not None:
            fcn = ""
            for j in range(i+3,values[0] - 1):
                s = lines[j].strip()
                if s == '':
                    break
                fcn += s
        if None not in [values, data, fcn]:
            break
    # get rid of + e
    fcn = fcn[:-4]
    # fix [] brackets
    s = fcn.split('[')
    if len(s) > 1:
        _bracket = re.compile('(.*)\](.*)')
        nfcn = s[0]
        for si in s[1:]:
            m = _bracket.match(si)
            if m is not None:
                si = m.group(1) + ')' + m.group(2)
                si = '(' + m.group(1) + ')' + m.group(2)
            nfcn += si
        fcn = nfcn
    # parameter names
    nparam = values[1] - values[0] + 1 - 5
    pnames = ','.join(['b' + str(i+1) for i in range(nparam)])
    p = []
    p0 = []
    prior = []
    for i in range(values[0] - 1, values[0] - 1 + nparam):
        s = lines[i].split()
        p.append(gv.gvar(s[-2] + ' +- ' + s[-1]))
        if USE_2ND_STARTING_VALUE:
            p0.append(float(s[3]))
        else:
            p0.append(float(s[2]))
        prior.append(gv.gvar('0 +- ' + str(200 * p[-1].mean)))
    p = np.array(p)
    p0 = np.array(p0)
    prior = list(gv.fmt(prior))
    nistp = answers.get(name, str(p))
    # error on y
    s = lines[values[1] - 3].split()
    yerr = '0 +- ' + s[-1]
    # collect x, y
    x = []
    y = []
    for i in range(data[0] - 1, data[1]):
        s = lines[i].split()
        if len(s) == 2:
            x.append(float(s[-1]))
            y.append(float(s[-2]))
        elif len(s) > 2:
            y.append(float(s[0]))
            x.append([float(si) for si in s[1:]])
    x = np.array(x)
    if len(x.shape) > 1:
        xnames = ','.join(['x' + str(i+1) for i in range(x.shape[1])])
        x = x.T
    else:
        xnames = 'x'
    y = np.array(y)
    if len(nistp.split('\n')) > 1:
        nistp = '\\n'.join(nistp.split('\n'))
    if name == 'nelson':
        fcn = fcn.replace('log(y)', 'y')
        yname = 'log(y)'
    else:
        yname = 'y'
    if name == 'mgh10' and not USE_2ND_STARTING_VALUE:
        p0 = gv.sdev(gv.gvar(prior)) / 100.
    template = \
"""
def {name}():
    print(20 * '=', '{name}')
    x = np.{x}
    y = np.{y}
    y = {yname} + gv.gvar(len(y) * ['{yerr}'])
    def fcn(x, b):
        {xnames} = x
        {pnames} = b
        {fcn}
        return y
    prior = gv.gvar({prior})
    p0 = np.{p0}
    fit = lsqfit.nonlinear_fit(
        prior=prior, data=(x,y), fcn=fcn, p0=p0, tol=1e-10,
        )
    print(fit)
    assert str(fit.p) == '{nistp}'
"""
    return template.format(
        name=name,
        x=repr(x),
        y=repr(y),
        p0=repr(p0),
        yerr=yerr,
        fcn=fcn,
        prior=repr(prior),
        nistp=nistp,
        pnames=pnames,
        xnames=xnames,
        yname=yname,
        )

if __name__ == '__main__':
    main()