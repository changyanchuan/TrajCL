from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy


ext_modules=[
    Extension("edwp", ["edwp.pyx"]),
]
   
setup(
        include_dirs=[numpy.get_include()],
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules,
    )
# python setup.py build_ext --inplace --force
# rm -rf build lib