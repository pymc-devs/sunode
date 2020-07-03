#!/usr/bin/env python
import sys

from cffi import FFI  # type: ignore
import glob
import os

ffibuilder = FFI()

base = os.path.dirname(__file__)
common = glob.glob(os.path.join(base, "../include/common/*.h"))
common.sort()
linsolve = glob.glob(os.path.join(base, "../include/sunlinsol/*.h"))
linsolve.sort()
cvodes = glob.glob(os.path.join(base, "../include/cvodes/*.h"))
cvodes.sort()

headers = common + linsolve + cvodes
include = []
library_dirs = []

if sys.platform == 'win32':
    with open(os.path.join(base, "source_cvodes_win.c")) as fsource:
        source_content = fsource.read()
    include.append(os.path.join(os.environ["CONDA_PREFIX"], "Library", "include"))
    library_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "Library", "lib"))

    extra_libs = []
    # lapackdense is not supported by the windows build of sundials
    for name in ['sunlinsol_lapackdense', 'sunlinsol_klu']:
        headers = [fn for fn in headers if name not in fn]
else:
    with open(os.path.join(base, "source_cvodes.c")) as fsource:
        source_content = fsource.read()

    #test if we can use conda libraries
    if "CONDA_PREFIX" in os.environ:
        include.append(os.path.join(os.environ["CONDA_PREFIX"], "include"))
        library_dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
    else:
        include.append("/usr/include/suitesparse/")    

    extra_libs = [
        "openblas",
        "pthread",
        "klu",
        "sundials_sunlinsollapackdense",
        "sundials_sunlinsollapackband",
        "sundials_sunlinsolklu",
    ]

for fname in headers:
    with open(fname, "r") as fheader:
        content = fheader.read()
    print(fname)
    ffibuilder.cdef(content)


ffibuilder.set_source(
    "_sundials_cvodes",
    source_content,
    libraries=[
        "sundials_nvecserial",
        "sundials_sunmatrixdense",
        "sundials_sunmatrixband",
        "sundials_sunmatrixsparse",
        "sundials_sunlinsoldense",
        "sundials_sunlinsolband",
        "sundials_sunlinsolspgmr",
        "sundials_cvodes",
    ] + extra_libs,
    include_dirs=include,
    library_dirs=library_dirs,
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
