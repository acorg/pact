import os
import shutil
import numpy

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

cdir = os.path.dirname(os.path.realpath(__file__))

if os.path.exists(f'{cdir}/__init__.py'):
    os.remove(f'{cdir}/__init__.py')

if os.path.exists(f'{cdir}/build'):
    shutil.rmtree(f'{cdir}/build')


extensions = [Extension('calculus', ['calculus.pyx', 'ctools.c'],
                        include_dirs=[numpy.get_include()],
                        extra_compile_args = ["-ffast-math","-O3","-mavx2","-mfma","-march=native"]
                        )]

setup(
    ext_modules=cythonize(extensions)
)


#python setup.py build_ext --inplace -lm
