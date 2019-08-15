#!/usr/bin/env python
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
ffibuilder.set_source(
    "_sundials_cvodes",
    content,
    libraries=[
        "sundials_nvecserial",
        "sundials_sunmatrixdense",
        "sundials_sunmatrixband",
        "sundials_sunmatrixsparse",
        "sundials_sunlinsollapackdense",
        "sundials_sunlinsollapackband",
        "sundials_sunlinsoldense",
        "sundials_sunlinsolband",
        "sundials_sunlinsolklu",
        "sundials_cvodes",
        "blas",
        "cblas",
        "lapack",
        "pthread",
        "klu",
    ],
    include_dirs=[os.environ["CONDA_PREFIX"] + "/include"],
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
