from pysundials_cffi import _ffi

import numpy as np
import numba
import numba.cffi_support

lib = _ffi.lib
ffi = _ffi.ffi
data_dtype = numba.cffi_support.map_type(ffi.typeof('realtype'))
index_dtype = numba.cffi_support.map_type(ffi.typeof('sunindextype'))


def empty(length, kind='serial'):
    assert kind == 'serial'
    ptr = lib.N_VNew_Serial(length)
    # assert ptr is not NULL
    return Vector(ptr, None)


def from_numpy(array, copy=False):
    if array.dtype != Vector.dtype:
        raise ValueError('Must have dtype %s' % Vector.dtype)
    if not array.flags['C_CONTIGUOUS']:
        raise ValueError('Array must be contiguous')
    if not array.ndim == 1:
        raise ValueError('Array must have rank 1')
    if copy:
        array = array.copy()

    data = _ffi.ffi.cast('void *', array.ctypes.get_data())
    ptr = lib.N_VMake_Serial(len(array), data)
    return Vector(ptr, array)


class Vector:
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr, data_owner):
        self._c_ptr = c_ptr
        self._size = lib.N_VGetLength_Serial(c_ptr)
        self._data_ptr = lib.N_VGetArrayPointer_Serial(c_ptr)
        self._buffer = ffi.buffer(self._data_ptr, self._size * self.dtype.itemsize)
        self._data_owner = None
    
    def __del__(self):
        del self._buffer
        self._data_ptr = None
        lib.N_VDestroy_Serial(self._c_ptr)
        self._c_ptr = None
        self._data_owner = None
    
    def c_print(self):
        lib.N_VPrint_Serial(self._c_ptr)
    
    @property
    def data(self):
        return np.frombuffer(self._buffer, self.dtype)
