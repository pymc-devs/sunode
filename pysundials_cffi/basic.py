import sys

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore
import numba  # type: ignore
import numba.cffi_support  # type: ignore
import logging
from typing import Optional, Tuple, Union, NewType, List, Any, cast, TextIO

from pysundials_cffi import _cvodes

__all__ = ["from_numpy", "empty_vector", "empty_matrix"]

logger = logging.getLogger("pysundials_cffi.basic")

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


CPointer = NewType("CPointer", int)


class Borrows:
    def __init__(self) -> None:
        self._borrowed: List[Any] = []

    def borrow(self, arg: Any) -> None:
        self._borrowed.append(arg)

    def release_borrowed(self) -> None:
        self._borrowed = []


def notnull(ptr: CPointer, msg: Optional[str] = None) -> CPointer:
    if ptr == ffi.NULL:
        if msg is None:
            raise ValueError("CPointer is NULL.")
        else:
            raise ValueError(msg)
    return ptr


def empty_vector(length: int, kind: str = "serial") -> Vector:
    assert kind == "serial"
    if kind != "serial":
        raise NotImplementedError()
    ptr = lib.N_VNew_Serial(length)
    if ptr is ffi.NULL:
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

    data = ffi.cast("void *", array.ctypes.get_data())
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


class SparseMatrix(Borrows):
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr: CPointer, *, name: Optional[str] = None):
        notnull(c_ptr)
        self._name = name
        self.c_ptr = c_ptr
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

    def __del__(self) -> None:
        logger.debug("Freeing matrix %s" % self.name)
        lib.SUNMatDestroy(self.c_ptr)
        del self.c_ptr
        self.release_borrowed()

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
        if ptr == ffi.NULL:
            raise ValueError("Matrix does not contain data.")

        nbytes = self.index_dtype.itemsize * size
        buffer = ffi.buffer(ptr, nbytes)
        return np.frombuffer(buffer, self.index_dtype)

    @property
    def indptr(self) -> np.ndarray:
        size = lib.SUNSparseMatrix_NP(self.c_ptr)
        size += 1  #
        ptr = lib.SUNSparseMatrix_IndexPointers(self.c_ptr)
        if ptr == ffi.NULL:
            raise ValueError("Matrix does not contain data.")

        nbytes = self.index_dtype.itemsize * size
        buffer = ffi.buffer(ptr, nbytes)
        return np.frombuffer(buffer, self.index_dtype)

    @property
    def data(self) -> np.ndarray:
        size = self.nnz
        ptr = lib.SUNSparseMatrix_Data(self.c_ptr)
        if ptr == ffi.NULL:
            raise ValueError("Matrix does not contain data.")

        nbytes = self.dtype.itemsize * size
        buffer = ffi.buffer(ptr, nbytes)
        return np.frombuffer(buffer, self.dtype)

    def c_print(self, file: Optional[TextIO] = None) -> None:
        if file is None:
            file = sys.stdout
        lib.SUNSparseMatrix_Print(self.c_ptr, file)

    def realloc(self, size: Optional[int] = None) -> None:
        if size is None:
            ret = lib.SUNSparseMatrix_Realloc(self.c_ptr)
        else:
            ret = lib.SUNSparseMatrix_Reallocate(self.c_ptr, size)
        if ret != 0:
            raise RuntimeError("Could not reallocate matrix storage.")


class DenseMatrix(Borrows):
    dtype = np.dtype(data_dtype.name)
    index_dtype = np.dtype(index_dtype.name)

    def __init__(self, c_ptr: CPointer, *, name: Optional[str] = None):
        if c_ptr == ffi.NULL:
            raise ValueError("CPointer is NULL.")
        self._name = name
        self.c_ptr = c_ptr
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

    def __del__(self) -> None:
        logger.debug("Freeing matrix %s" % self.name)
        lib.SUNMatDestroy(self.c_ptr)
        del self.c_ptr
        self.release_borrowed()

    @property
    def shape(self) -> Tuple[int, int]:
        rows = lib.SUNDenseMatrix_Rows(self.c_ptr)
        columns = lib.SUNDenseMatrix_Columns(self.c_ptr)
        return (rows, columns)

    @property
    def data(self) -> np.ndarray:
        size = lib.SUNDenseMatrix_LData(self.c_ptr)
        ptr = lib.SUNDenseMatrix_Data(self.c_ptr)
        assert ptr != ffi.NULL
        nbytes = self.dtype.itemsize * size
        buffer = ffi.buffer(ptr, nbytes)
        array = np.frombuffer(buffer, self.dtype)
        rows, columns = self.shape
        # Sundials stores dense matrices in fortran order
        return array.reshape((columns, rows)).T

    def as_sparse(self, droptol: float = 0.0, format: str = "csr") -> Union[sparse.csr_matrix, sparse.csc_matrix]:
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
        if c_ptr == ffi.NULL:
            raise ValueError("CPointer is NULL.")
        self._name = name
        self.c_ptr = c_ptr
        self._size = lib.N_VGetLength_Serial(c_ptr)
        self._data_owner = None

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        else:
            return str(self.c_ptr)

    def __del__(self) -> None:
        logger.debug("Freeing vector %s" % self.name)
        lib.N_VDestroy_Serial(self.c_ptr)
        del self.c_ptr
        self.release_borrowed()

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
        data_ptr = lib.N_VGetArrayPointer_Serial(self.c_ptr)
        buffer = ffi.buffer(data_ptr, self.shape[0] * self.dtype.itemsize)
        return np.frombuffer(buffer, self.dtype)
