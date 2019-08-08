#/usr/bin/env python
from cffi import FFI
import glob
import os

ffibuilder = FFI()

base = os.path.dirname(__file__)
common = glob.glob(os.path.join(base, '../include/common/*.h'))
common.sort()
cvodes = glob.glob(os.path.join(base, '../include/cvodes/*.h'))
cvodes.sort()

headers = common + cvodes

for fname in headers:
    with open(fname, 'r') as fheader:
        content = fheader.read()
    print(fname)
    ffibuilder.cdef(content)

with open(os.path.join(base, 'source_cvodes.c')) as fsource:
    content = fsource.read()
ffibuilder.set_source(
    '_sundials_cvodes',
    content,
    libraries=[
        'sundials_nvecserial',
        'sundials_sunmatrixdense',
        'sundials_sunmatrixband',
        'sundials_sunmatrixsparse',
        'sundials_sunlinsollapackdense',
        'sundials_cvodes',
        'blas',
        'cblas',
        'lapack',
        'pthread',
    ],
    include_dirs=[
        os.environ['CONDA_PREFIX'] + '/include',
    ]
)


if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
