from __future__ import annotations

import sys
import weakref
import logging
from typing import Optional, Tuple, Union, NewType, List, Any, cast, TextIO, Callable

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
import numba  # type: ignore
import numba.cffi_support  # type: ignore

from sunode import _cvodes


__all__ = ["lib", "ffi", "ERRORS", "Borrows", "notnull", "check", "check_ptr", "check_code", "as_numpy"]


logger = logging.getLogger("sunode.basic")

lib: Any = _cvodes.lib
ffi: Any = _cvodes.ffi

numba.cffi_support.register_module(_cvodes)
numba.cffi_support.register_type(
    ffi.typeof("N_Vector").item, numba.types.Opaque("N_Vector")
)
numba.cffi_support.register_type(
    ffi.typeof("SUNMatrix").item, numba.types.Opaque("SUNMatrix")
)

_data_dtype = numba.cffi_support.map_type(ffi.typeof("realtype"))
_index_dtype = numba.cffi_support.map_type(ffi.typeof("sunindextype"))
data_dtype: Any = np.dtype(_data_dtype.name)
index_dtype: Any = np.dtype(_index_dtype.name)


CPointer = NewType("CPointer", int)


ERRORS = {}
for name in dir(lib):
    item = getattr(lib, name)
    if not isinstance(item, int):
        continue
    if name.startswith('CV_') or name.startswith('CVLS_') or name.startswith('SUN_NLS_'):
        ERRORS[item] = name


class Borrows:
    def __init__(self) -> None:
        self._borrowed: List[Any] = []

    def borrow(self, arg: Any) -> None:
        self._borrowed.append(arg)

    def release_borrowed_func(self) -> Callable[[], None]:
        borrowed = self._borrowed
        # Does not keep a reference to self
        def release() -> None:
            borrowed.clear()

        return release


def notnull(ptr: CPointer, msg: Optional[str] = None) -> CPointer:
    if ptr == ffi.NULL:
        if msg is None:
            raise ValueError("CPointer is NULL.")
        else:
            raise ValueError(msg)
    return ptr


def check(retcode: Union[int, CPointer]) -> Union[None, CPointer]:
    if isinstance(retcode, int) and retcode != 0:
        raise ValueError('Bad return code from sundials: %s (%s)' % (ERRORS[retcode], retcode))
    if isinstance(retcode, ffi.CData):
        if retcode == ffi.NULL:
            raise ValueError('Return value of sundials is NULL.')
        return retcode
    return None


def check_ptr(retval: CPointer) -> CPointer:
    if retval == ffi.NULL:
        raise ValueError('Return value of sundials is NULL.')
    return retval


def check_code(retval: int) -> int:
    if retval != 0:
        raise ValueError('Bad return code from sundials: %s (%s)' % (ERRORS[retval], retval))
    return retval


class RefCount:
    def __init__(self) -> None:
        self.count: int = 0

    def borrow(self) -> None:
        self.count += 1

    def release(self) -> None:
        assert self.count > 0
        self.count -= 1

    def is_zero(self) -> bool:
        assert self.count >= 0
        return self.count == 0


def as_numpy(
    owner: Any,
    ptr: CPointer,
    size: int,
    dtype: np.dtype,
    counter: Optional[RefCount] = None,
) -> np.ndarray:
    if size < 0:
        raise ValueError("Array size must not be negative.")

    if size != 0:
        notnull(ptr)

    def release(ptr: CPointer) -> None:
        nonlocal owner
        if counter is not None:
            counter.release()

    if counter is not None:
        counter.borrow()
    ptr = ffi.gc(ptr, release)
    buffer = ffi.buffer(ptr, size * dtype.itemsize)
    return np.frombuffer(buffer, dtype)
