import os
import re
from setuptools import setup, find_packages


def get_version():
    VERSIONFILE = os.path.join("sunode", "__init__.py")
    lines = open(VERSIONFILE).readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version in {VERSIONFILE}.")


setup(
    name='sunode',
    version=get_version(),
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
