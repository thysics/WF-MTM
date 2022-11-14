from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("sample_a.pyx"),
    include_dirs=[numpy.get_include()]
)


setup(
    ext_modules=[
        Extension("sample_a", ["sample_a.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()


