#!/usr/bin/env python
from cffi import FFI  # type: ignore
import glob
import os

ffibuilder = FFI()

base = os.path.dirname(__file__)
common = glob.glob(os.path.join(base, "../include/common/*.h"))
common.sort()
cvode = glob.glob(os.path.join(base, "../include/cvode/*.h"))
cvode.sort()

headers = common + cvode

for fname in headers:
    with open(fname, "r") as fheader:
        content = fheader.read()
    print(fname)
    ffibuilder.cdef(content)

with open(os.path.join(base, "source_cvode.c")) as fsource:
    content = fsource.read()
ffibuilder.set_source(
    "_sundials_cvode",
    content,
    libraries=[
        "sundials_nvecserial",
        "sundials_sunmatrixdense",
        "sundials_sunmatrixband",
        "sundials_sunmatrixsparse",
        "sundials_sunlinsollapackdense",
        "sundials_cvode",
        "blas",
        "cblas",
        "lapack",
        "pthread",
    ],
    include_dirs=[os.path.join(os.environ["CONDA_PREFIX"], "include")],
)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
