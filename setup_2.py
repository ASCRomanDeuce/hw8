from setuptools import setup
from Cython.Build import cythonize
import numpy 

setup(
    ext_modules=cythonize("two_helper.pyx"),
    include_dirs=[numpy.get_include()]
)


# python setup.py build_ext --inplace