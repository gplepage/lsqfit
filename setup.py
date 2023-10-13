from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import subprocess
import sys

# install gsl interface if gsl installed 

ext_args_nogsl = dict(
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_args_gsl = dict(
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=['gsl', 'gslcblas'],
    runtime_library_dirs=[],
    extra_link_args=[]
    )

ext_modules =[
    Extension(name='lsqfit._utilities', sources=['src/lsqfit/_utilities.pyx'], **ext_args_nogsl),
    ]

if sys.version_info >= (3, 7):
    try:
        p = subprocess.run(['gsl-config', '--cflags'], capture_output=True)
        if p.returncode != 0:
            raise FileNotFoundError('bad return code from gsl-config')
        cflags = p.stdout.decode('utf-8')[2:-1]
        ext_args_nogsl['include_dirs'].append(cflags)
        p = subprocess.run(['gsl-config', '--libs'], capture_output=True)
        if p.returncode != 0:
            raise FileNotFoundError('bad return code from gsl-config')
        libs = p.stdout.decode('utf-8').split(' ')[0][2:]
        ext_args_gsl['library_dirs'].append(libs)
        ext_modules.append(
            Extension(name='lsqfit._gsl', sources=['src/lsqfit/_gsl.pyx'], **ext_args_gsl)
            )
    except FileNotFoundError:
        # no gsl
        ext_modules = ext_modules[:1]
else:
    ext_modules.append(
        Extension(name='lsqfit._gsl', sources=['src/lsqfit/_gsl.pyx'], **ext_args_gsl)
        )

try:    
    setup(ext_modules=cythonize(ext_modules))
except:
    print("*** can't find gsl; re-install without it")
    if len(ext_modules) > 1:
        setup(ext_modules=cythonize(ext_modules[:1]))
    else:
        raise RuntimeError("can't find gsl or scipy")

