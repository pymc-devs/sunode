import logging
import weakref
import sys
from typing import Optional, Callable, TextIO, Tuple

import numpy as np  # type: ignore

from sunode import basic
from sunode.basic import lib, ffi, data_dtype, index_dtype, Borrows, notnull, as_numpy, CPointer


logger = logging.getLogger("sunode.vector")


def empty_vector(length: int, kind: str = "serial") -> Vector:
    if kind not in ["serial"]:
        raise ValueError("Vector backend %s not supported." % kind)
    if length < 0:
        raise ValueError("Length must not be negative.")
    if kind != "serial":
        raise NotImplementedError()
    ptr = lib.N_VNew_Serial(length)
    if ptr == ffi.NULL:
        raise MemoryError("Could not allocate vector.")
    return Vector(ptr)


def from_numpy(array: np.ndarray, copy: bool = False) -> Vector:
    if array.dtype != Vector.dtype:
        raise ValueError("Must have dtype %s" % Vector.dtype)
    if not array.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be contiguous")
    if not array.ndim == 1:
        raise ValueError("Array must have rank 1")
    if copy:
        array = array.copy()

    data = ffi.cast("void *", ffi.from_buffer(array))
    notnull(data)
    ptr = lib.N_VMake_Serial(len(array), data)
    notnull(ptr)
    vec = Vector(ptr)
    vec.borrow(array)
    return vec


class Vector(Borrows):
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr: CPointer, *, name: Optional[str] = None) -> None:
        super().__init__()
        self._name = name
        notnull(c_ptr)
        self.c_ptr = c_ptr

        def finalize(ptr: CPointer, name: str, release_borrowed: Callable[[], None]) -> None:
            if ptr == ffi.NULL:
                logger.error("Trying to free c_ptr of vector %s but it is NULL" % name)
            else:
                logger.debug("Freeing vector %s" % name)
                lib.N_VDestroy_Serial(ptr)
            release_borrowed()

        weakref.finalize(
            self, finalize, self.c_ptr, self.name, self.release_borrowed_func()
        )

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return str(self.c_ptr)

    def c_print(self, file: Optional[TextIO] = None) -> None:
        if file is None:
            file = sys.stdout
        else:
            lib.N_VPrintFile_Serial(self.c_ptr, file.fileno())

    @property
    def shape(self) -> Tuple[int]:
        return (lib.N_VGetLength_Serial(self.c_ptr),)

    @property
    def data(self) -> np.ndarray:
        ptr = lib.N_VGetArrayPointer_Serial(self.c_ptr)
        return as_numpy(self, ptr, len(self), self.dtype)


