""" 
build in place: python setup.py build_ext --inplace 
install in ddd: python setup.py install --install-lib ddd

Created by G. Peter Lepage (Cornell University) on 2010-08.
Copyright (c) 2011 G. Peter Lepage.

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

name = "lsqfit_util"

ext_modules = [ 
    Extension(name,[name+".pyx"],libraries=["gsl","gslcblas"],
            include_dirs=[numpy.get_include()]),
    ]

py_modules = [] # ["xxx"]

setup(name="lsqfit_util",
    version="1.0",
    author="G.P. Lepage",
    author_email="GPL3@CORNELL.EDU",
    ext_modules=ext_modules, py_modules=py_modules,
    license='GPLv3',
    cmdclass = {'build_ext':build_ext}
    )
