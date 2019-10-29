from setuptools import setup, find_packages

setup(
    name='sunode',
    version='0.0.1',
    author='Adrian Seyboldt',
    author_email='adrian.seyboldt@gmail.com',
    description='Python wrapper of sundials for solving ordinary differential equations',
    url='https://github.com/aseyboldt/sunode',
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=[
        "sunode/build_cvodes.py:ffibuilder",
    ],
    packages=find_packages(),
    install_requires=[
        "cffi>=1.0.0",
        "sympy",
        "numpy",
        "numba",
    ],
)
