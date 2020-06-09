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

for fname in headers:
    with open(fname, "r") as fheader:
        content = fheader.read()
    print(fname)
    ffibuilder.cdef(content)

with open(os.path.join(base, "source_cvodes.c")) as fsource:
    content = fsource.read()


if sys.platform == 'win32':
    include = [os.path.join(os.environ["CONDA_PREFIX"], "Library", "include")]
    library_dirs = [
        os.path.join(os.environ["CONDA_PREFIX"], "Library", "lib")
    ]
    extra_libs = []
else:
    include = [os.path.join(os.environ["CONDA_PREFIX"], "include")]
    library_dirs = [os.path.join(os.environ["CONDA_PREFIX"], "lib")]
    extra_libs = [
        "blas",
        "lapack",
        "pthread",
        "klu",
        "sundials_sunlinsollapackdense",
        "sundials_sunlinsollapackband",
        "sundials_sunlinsolklu",
    ]

ffibuilder.set_source(
    "_sundials_cvodes",
    content,
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
