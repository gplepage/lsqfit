from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import subprocess
import sys

# install gsl interface if gsl installed 
# looks for gsl by calling gsl-config (needs to be in PATH)

ext_modules =[
    Extension(name='lsqfit._utilities', sources=['src/lsqfit/_utilities.pyx']),
    ]

try:
    gsl_args = {}
    p = subprocess.run(['gsl-config', '--cflags'], capture_output=True)
    if p.returncode != 0:
        raise FileNotFoundError('bad return code from gsl-config - cflags')
    cflags = p.stdout.decode('utf-8')[2:-1]  # get rid of -I at start and \n at end
    gsl_args['include_dirs'] = [cflags]
    p = subprocess.run(['gsl-config', '--libs'], capture_output=True)
    if p.returncode != 0:
        raise FileNotFoundError('bad return code from gsl-config - libs')
    libs = p.stdout.decode('utf-8').split(' ')[0][2:]  # get rid of -lgsl etc and -L at start
    gsl_args['library_dirs'] = [libs]
    gsl_args['libraries'] = ['gsl', 'gslcblas']
    ext_modules.append(
        Extension(name='lsqfit._gsl', sources=['src/lsqfit/_gsl.pyx'], **gsl_args)
        )
except FileNotFoundError:
    # no gsl
    ext_modules = ext_modules[:1]

setup(ext_modules=cythonize(ext_modules))

