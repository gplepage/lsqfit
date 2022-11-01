"""
Created by G. Peter Lepage (Cornell University) on 9/2011.
Copyright (c) 2011-21 G. Peter Lepage.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version (see <http://www.gnu.org/licenses/>).

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
"""

LSQFIT_VERSION = open('src/lsqfit/_version.py', 'r').readlines()[0].split("'")[1]

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext as _build_ext
from distutils.command.build_py import build_py # as _build_py

# compile from existing .c files if USE_CYTHON is False
from sys import version_info
if version_info[0] == 3 and version_info[1] >= 11:
    USE_CYTHON = True
else:
    USE_CYTHON = False

class build_ext(_build_ext):
    # delays using numpy and cython until they are installed;
    # cython is optional (set USE_CYTHON)
    # this code adapted from https://github.com/pandas-dev/pandas setup.py
    def build_extensions(self):
        import numpy
        # add version number
        if USE_CYTHON:
            from Cython.Build import cythonize
            self.extensions = cythonize(self.extensions)
        numpy_include = numpy.get_include()
        for ext in self.extensions:
            ext.include_dirs.append(numpy_include)
        _build_ext.build_extensions(self)

# class build_py(_build_py):
#     # adds version info
#     def run(self):
#         """ Append version number to lsqfit/__init__.py """
#         with open('src/lsqfit/__init__.py', 'a') as lsfile:
#             lsfile.write("\n__version__ = '%s'\n" % LSQFIT_VERSION)
#         _build_py.run(self)

# extension modules
# Add explicit directories to the ..._dirs variables if
# the build process has trouble finding the gsl library
# or the numpy headers. This should not be necessary if
# gsl and numpy are installed in standard locations.
ext_args_gsl = dict(
    libraries=["gsl", "gslcblas"],
    include_dirs=[],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[],
    )

ext_args_nogsl = dict(
    libraries=[],
    include_dirs=[],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[],
    )

ext = '.pyx' if USE_CYTHON else '.c'

ext_modules = [
    Extension(
        "lsqfit._utilities",
        ["src/lsqfit/_utilities" + ext],
        **ext_args_nogsl
        ),
    Extension(
        "lsqfit._gsl",
        ["src/lsqfit/_gsl" + ext],
        **ext_args_gsl
        ),
    ]

# distutils
requires = (
    ["cython (>=0.17)", "numpy (>=1.7)", "scipy (>=0.16)", "gvar (>=11.1)"]
    if USE_CYTHON else
    ["numpy (>=1.7)", "scipy (>=0.16)", "gvar (>=11.1)"]
    )

# pip
install_requires = (
    ['cython>=0.17', 'numpy>=1.7', 'scipy>=0.16', 'gvar>=11.1']
    if USE_CYTHON else
    ['numpy>=1.7', 'scipy>=0.16', 'gvar>=11.1']
    )

# pypi
with open('README.rst', 'r') as file:
    long_description = file.read()

# packages
packages = ["lsqfit"]
package_dir = {"lsqfit":"src/lsqfit"}
package_data = {}

setup_args = dict(
    name='lsqfit',
    version=LSQFIT_VERSION,
    description='Utilities for nonlinear least-squares fits.',
    author='G. Peter Lepage',
    author_email='g.p.lepage@cornell.edu',
    cmdclass={'build_ext':build_ext, 'build_py':build_py},
    packages=packages,
    package_dir=package_dir,
    package_data=package_data,
    ext_modules= ext_modules,
    install_requires=install_requires,   # for pip (distutils ignores)
    requires=requires,  # for distutils
    url="https://github.com/gplepage/lsqfit.git",
    license='GPLv3+',
    platforms='Any',
    long_description=long_description,
    classifiers = [                     #
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Cython',
        'Topic :: Scientific/Engineering'
        ],
    )

try:
    # will fail if gsl not installed
    setup(**setup_args)
except:
    # install without gsl
    print('\n*** Install failed. Trying again without gsl.\n')
    setup_args['ext_modules'] = ext_modules[:1]
    setup(**setup_args)