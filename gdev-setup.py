from distutils.core import setup

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.command.build_py import build_py

ext_modules = [ Extension("gdev",["src/gdev.pyx"]) ]

setup(name='gdev', version='1.0', 
      description="Aliases gvar to gdev for legacy code.",
      author='G. Peter Lepage',
      author_email='g.p.lepage@cornell.edu',
      ext_modules=ext_modules,
      cmdclass={'build_ext':build_ext,'build_py':build_py}
      )

# setup(name='gdev', version='1.0', 
#       description="Aliases gvar to gdev for legacy code.",
#       py_modules=['gdev'])