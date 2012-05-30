""" 
build in place: python setup.py build_ext --inplace 
install in ddd: python setup.py install --install-lib ddd

Created by G. Peter Lepage (Cornell University) on 9/2011.
Copyright (c) 2011,2012 G. Peter Lepage.

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
try:
   from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
   from distutils.command.build_py import build_py
import numpy
import lsqfit__version__

ext_modules = [ 
    Extension("gvar",["gvar.pyx"],libraries=["gsl","gslcblas"],
                include_dirs=[numpy.get_include()]), #,extra_link_args=['-framework','vecLib']),
    Extension("lsqfit_util",["lsqfit_util.pyx"],libraries=["gsl","gslcblas"],
                include_dirs=[numpy.get_include()]) #,extra_link_args=['-framework','vecLib']),
    ]

py_modules = [] # ["xxx"]

setup(name='lsqfit',
    version=lsqfit__version__.__version__,
    description='Utilities for nonlinear least-squares fits.',
    author='G. Peter Lepage',
    author_email='g.p.lepage@cornell.edu',
    license='GPLv3',
    py_modules = ['lsqfit','lsqfit__version__'],
    ext_modules = ext_modules,
    cmdclass = {'build_ext':build_ext,'build_py':build_py},
    requires = ["cython (>=0.14)","numpy (>=1.0)","gsl (>=1.8)"],
    url="http://pypi.python.org/pypi",
    long_description="""\
    The modules defined here are designed to facilitate least-squares
    fitting of noisy data by multi-dimensional, nonlinear functions of
    arbitrarily many parameters. The central module is :mod:`lsqfit` because
    it provides the fit functions. :mod:`lsqfit` makes heavy use of auxiliary
    module :mod:`gvar`, which provides tools that facilitate the analysis of
    error propagation, and also the creation of complicated multi-dimensional
    gaussian distributions. :mod:`lsqfit` supports Bayesian priors for the fit
    parameters, with arbitrarily complicated multidimensional gaussian
    distributions. An tutorial on fitting is included in the documentation.
    
    These modules use the Gnu Scientific Library (GSL) to do the fitting,
    numpy for efficient array arithmetic, and cython to compile efficient
    interface code.
    """
)