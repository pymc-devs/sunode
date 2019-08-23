from setuptools import setup

setup(
    name='pysundials-cffi',
    version='0.0.1',
    author='Adrian Seyboldt',
    author_email='adrian.seyboldt@gmail.com',
    description='Python wrapper of sundials for solving ordinary differential equations',
    url='https://github.com/aseyboldt/pysundials_cffi',
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=[
        #"pysundials_cffi/build_cvode.py:ffibuilder",
        "pysundials_cffi/build_cvodes.py:ffibuilder",
    ],
    install_requires=[
        "cffi>=1.0.0",
        "sympy",
        "numpy",
        "numba",
    ],
)
