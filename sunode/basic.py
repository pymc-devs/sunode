from __future__ import annotations

import sys
import weakref

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
import numba  # type: ignore
import numba.cffi_support  # type: ignore
import logging
from typing import Optional, Tuple, Union, NewType, List, Any, cast, TextIO, Callable

from sunode import _cvodes


__all__ = ["from_numpy", "empty_vector", "empty_matrix"]


logger = logging.getLogger("sunode.basic")

lib = _cvodes.lib
ffi = _cvodes.ffi

numba.cffi_support.register_module(_cvodes)
numba.cffi_support.register_type(
    ffi.typeof("N_Vector").item, numba.types.Opaque("N_Vector")
)
numba.cffi_support.register_type(
    ffi.typeof("SUNMatrix").item, numba.types.Opaque("SUNMatrix")
)

data_dtype = numba.cffi_support.map_type(ffi.typeof("realtype"))
index_dtype = numba.cffi_support.map_type(ffi.typeof("sunindextype"))
data_dtype = np.dtype(data_dtype.name)
index_dtype = np.dtype(index_dtype.name)


CPointer = NewType("CPointer", int)


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


MATRIX_TYPES = {
    "sparse": lib.SUNMATRIX_SPARSE,
    "dense": lib.SUNMATRIX_DENSE,
    "band": lib.SUNMATRIX_BAND,
}

MATRIX_TYPES_REV = {v: k for k, v in MATRIX_TYPES.items()}


def empty_matrix(
    shape: Tuple[int, int],
    kind: str = "dense",
    format: Optional[str] = None,
    sparsity: Union[None, np.ndarray, sparse.csr_matrix, sparse.csc_matrix] = None,
) -> Union[DenseMatrix, SparseMatrix]:
    rows, columns = shape
    if rows < 0 or columns < 0:
        raise ValueError("Number of rows and columns must not be negative.")
    if kind == "dense":
        ptr = lib.SUNDenseMatrix(rows, columns)
        if ptr == ffi.NULL:
            raise MemoryError("Could not allocate matrix.")
        return DenseMatrix(ptr)
    elif kind == "band":
        raise NotImplementedError()  # TODO
    elif kind == "sparse":
        if sparsity is None:
            raise ValueError(
                "Sparsity must a scipy.sparse matrix or "
                "a numpy array that indicates the "
                "sparsity structure of the matrix."
            )
        if format == "csr":
            sparsity = sparse.csr_matrix(sparsity, shape=shape)
            c_format = lib.CSR_MAT
        elif format == "csc":
            sparsity = sparse.csc_matrix(sparsity, shape=shape)
            c_format = lib.CSC_MAT
        else:
            raise ValueError("Format must be one of csr and csc.")
        ptr = lib.SUNSparseMatrix(rows, columns, sparsity.nnz, c_format)
        if ptr == ffi.NULL:
            raise MemoryError("Could not allocate matrix.")
        matrix = SparseMatrix(ptr)
        matrix.indptr[...] = sparsity.indptr
        matrix.indices[...] = sparsity.indices
        return matrix
    else:
        raise ValueError("Unknown matrix type %s" % kind)


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


def _as_numpy(
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


class SparseMatrix(Borrows):
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr: CPointer, *, name: Optional[str] = None):
        super().__init__()
        notnull(c_ptr)
        self._buffer_refcount = RefCount()
        self._name = name
        self.c_ptr = c_ptr

        def finalize(
            ptr: CPointer, name: str, release_borrowed: Callable[[], None]
        ) -> None:
            if ptr == ffi.NULL:
                logger.error("Trying to free matrix %s, but c_ptr is NULL" % name)
            else:
                logger.debug("Freeing matrix %s" % name)
                lib.SUNMatDestroy(ptr)
            release_borrowed()

        weakref.finalize(self, finalize, c_ptr, self.name, self.release_borrowed_func())

        c_kind = lib.SUNMatGetID(c_ptr)
        kind = MATRIX_TYPES_REV.get(c_kind, c_kind)
        if kind != "sparse":
            raise ValueError("Not a sparse matrix, but of type %s" % kind)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return str(self.c_ptr)

    @property
    def format(self) -> str:
        c_type = lib.SUNSparseMatrix_SparseType(self.c_ptr)
        if c_type == lib.CSR_MAT:
            return "csr"
        elif c_type == lib.CSC_MAT:
            return "csc"
        else:
            raise ValueError("Unknown matrix format: %s" % c_type)

    @property
    def shape(self) -> Tuple[int, int]:
        rows = lib.SUNSparseMatrix_Rows(self.c_ptr)
        columns = lib.SUNSparseMatrix_Columns(self.c_ptr)
        return (rows, columns)

    @property
    def nnz(self) -> int:
        return cast(int, lib.SUNSparseMatrix_NNZ(self.c_ptr))

    @property
    def scipy(self) -> Union[sparse.csc_matrix, sparse.csr_matrix]:
        vals = self.data, self.indices, self.indptr
        if self.format == "csr":
            return sparse.csr_matrix(vals, shape=self.shape)
        elif self.format == "csc":
            return sparse.csc_matrix(vals, shape=self.shape)
        assert False

    @property
    def indices(self) -> np.ndarray:
        size = self.nnz
        ptr = lib.SUNSparseMatrix_IndexValues(self.c_ptr)
        return _as_numpy(self, ptr, size, self.index_dtype, self._buffer_refcount)

    @property
    def indptr(self) -> np.ndarray:
        size = lib.SUNSparseMatrix_NP(self.c_ptr)
        size += 1  #
        ptr = lib.SUNSparseMatrix_IndexPointers(self.c_ptr)
        return _as_numpy(self, ptr, size, self.index_dtype, self._buffer_refcount)

    @property
    def data(self) -> np.ndarray:
        size = self.nnz
        ptr = lib.SUNSparseMatrix_Data(self.c_ptr)
        return _as_numpy(self, ptr, size, self.dtype, self._buffer_refcount)

    def c_print(self, file: Optional[TextIO] = None) -> None:
        if file is None:
            file = sys.stdout
        lib.SUNSparseMatrix_Print(self.c_ptr, file)

    def realloc(self, size: Optional[int] = None) -> None:
        if not self._buffer_refcount.is_zero():
            raise RuntimeError(
                "Can not reallocate matrix while numpy views of data are alive."
            )
        if size is None:
            ret = lib.SUNSparseMatrix_Realloc(self.c_ptr)
        else:
            ret = lib.SUNSparseMatrix_Reallocate(self.c_ptr, size)
        if ret != 0:
            raise RuntimeError("Could not reallocate matrix storage.")


class BandMatrix(Borrows):
    pass


class DenseMatrix(Borrows):
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr: CPointer, *, name: Optional[str] = None):
        super().__init__()
        notnull(c_ptr)
        self._name = name
        self.c_ptr = c_ptr

        def finalize(ptr: CPointer, name: str, release_borrowed: Callable[[], None]) -> None:
            if ptr == ffi.NULL:
                logger.error("Trying to free matrix %s, but c_ptr is NULL" % name)
            else:
                logger.debug("Freeing matrix %s" % name)
                lib.SUNMatDestroy(ptr)
            release_borrowed()

        weakref.finalize(self, finalize, c_ptr, self.name, self.release_borrowed_func())

        c_kind = lib.SUNMatGetID(c_ptr)
        kind = MATRIX_TYPES_REV.get(c_kind, c_kind)
        if kind != "dense":
            raise ValueError("Not a dense matrix, but of type %s" % kind)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return str(self.c_ptr)

    @property
    def shape(self) -> Tuple[int, int]:
        rows = lib.SUNDenseMatrix_Rows(self.c_ptr)
        columns = lib.SUNDenseMatrix_Columns(self.c_ptr)
        return (rows, columns)

    @property
    def data(self) -> np.ndarray:
        size = lib.SUNDenseMatrix_LData(self.c_ptr)
        ptr = lib.SUNDenseMatrix_Data(self.c_ptr)
        array = _as_numpy(self, ptr, size, self.dtype)

        rows, columns = self.shape
        # Sundials stores dense matrices in fortran order
        return array.reshape((columns, rows)).T

    def as_sparse(
        self, droptol: float = 0.0, format: str = "csr"
    ) -> Union[sparse.csr_matrix, sparse.csc_matrix]:
        if format.lower() == "csr":
            c_format = lib.CSR_MAT
        elif format.lower() == "csc":
            c_format = lib.CSC_MAT
        else:
            raise ValueError("Format must be one of csr or csc.")

        ptr = lib.SUNSparseFromDenseMatrix(self.c_ptr, droptol, c_format)
        if ptr == ffi.NULL:
            raise ValueError("CPointer is NULL.")
        return SparseMatrix(ptr)

    def c_print(self, file: Optional[TextIO] = None) -> None:
        if file is None:
            file = sys.stdout
        lib.SUNDenseMatrix_Print(self.c_ptr, file)


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

        self._size = lib.N_VGetLength_Serial(c_ptr)

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
        return _as_numpy(self, ptr, len(self), self.dtype)


class LinearSolver(Borrows):
    def __init__(self, c_ptr: CPointer) -> None:
        super().__init__()
        notnull(c_ptr, "Linear solver cpointer is NULL.")

        def finalize(c_ptr: CPointer, release_borrowed: Callable[[], None]) -> None:
            if c_ptr == ffi.NULL:
                logger.warn("Trying to free LinearSolver but it is NULL.")
            else:
                logger.debug("Freeing LinearSolver")
                lib.SUNLinSolFree(c_ptr)
            release_borrowed()
        weakref.finalize(
            self, finalize, c_ptr, self.release_borrowed_func()
        )

        self.c_ptr = c_ptr

    def initialize(self) -> None:
        ret = lib.SUNLinSolInitialize(self.c_ptr)
        if ret != 0:
            raise ValueError("Could not initialize linear solver.")

    def reinit(self) -> None:
        raise NotImplementedError()


Matrix = Union[DenseMatrix, SparseMatrix]
