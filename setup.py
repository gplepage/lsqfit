""" 
build in place: python setup.py build_ext --inplace 
install in ddd: python setup.py install --install-lib ddd

Created by G. Peter Lepage (Cornell University) on 9/2011.
Copyright (c) 2011-14 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.command.build_py import build_py
import numpy

LSQFIT_VERSION = '4.8.5.1'

# create lsqfit/_version.py so lsqfit knows its version number 
with open("src/lsqfit/_version.py","w") as version_file:
    version_file.write(
        "# File created by lsqfit setup.py\nversion = '%s'\n" 
        % LSQFIT_VERSION
        )
with open("src/gvar/_version.py","w") as version_file:
    version_file.write(
        "# File created by lsqfit setup.py\nversion = '%s'\n" 
        % LSQFIT_VERSION
        )

# extension modules 
# Add explicit directories to the ..._dirs variables if 
# the build process has trouble finding the gsl library 
# or the numpy headers. This should not be necessary if
# gsl and numpy are installed in standard locations.
ext_args = dict(
    libraries=["gsl", "gslcblas"],
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[], # ['-framework','vecLib'], # for Mac OSX ?
    )
ext_modules = [     #
    Extension("gvar._gvarcore", ["src/gvar/_gvarcore.pyx"], **ext_args),
    Extension( "gvar._svec_smat", ["src/gvar/_svec_smat.pyx"], **ext_args),
    Extension("gvar._utilities", ["src/gvar/_utilities.pyx"], **ext_args),
    Extension("gvar.dataset", ["src/gvar/dataset.pyx"], **ext_args),
    Extension("gvar._bufferdict", ["src/gvar/_bufferdict.pyx"], **ext_args),
    Extension("lsqfit._utilities", ["src/lsqfit/_utilities.pyx"],  **ext_args),
    ]

# packages
packages = ["gvar","lsqfit"]
package_dir = {"lsqfit":"src/lsqfit", "gvar":"src/gvar"}
package_data = {"gvar":['../gvar.pxd']}

setup(name='lsqfit',
    version=LSQFIT_VERSION,
    description='Utilities for nonlinear least-squares fits.',
    author='G. Peter Lepage',
    author_email='g.p.lepage@cornell.edu',
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    ext_modules=ext_modules,
    cmdclass={'build_ext':build_ext,'build_py':build_py},
    requires=["cython (>=0.17)","numpy (>=1.7)"],
    url="https://github.com/gplepage/lsqfit.git",
    license='GPLv3+',
    platforms='Any',
    long_description="""\
    These packages facilitate least-squares fitting of noisy data by
    multi-dimensional, nonlinear functions of arbitrarily many
    parameters. The central package is :mod:`lsqfit` which provides
    the fitting capability. :mod:`lsqfit` makes heavy use of package
    :mod:`gvar`, which provides tools for the analysis of error
    propagation, and also for the creation of complicated
    multi-dimensional gaussian distributions. :mod:`lsqfit` supports
    Bayesian priors for the fit parameters, with arbitrarily
    complicated multidimensional gaussian distributions. It uses
    automatic differentiation to compute gradients, greatly simplifying
    the design of fit functions.

    These packages use the Gnu Scientific Library (GSL) to do the
    fitting, numpy for efficient array arithmetic, and cython to
    compile efficient core routines and interface code.
    """ 
    ,
    classifiers = [                     #
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
        ]
)