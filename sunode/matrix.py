import logging
import weakref
from abc import ABCMeta, abstractproperty
from typing import Tuple, overload, Optional, Union, Callable, cast, TextIO, ClassVar
from typing_extensions import Literal
import sys

import numpy as np  # type: ignore
from scipy import sparse  # type: ignore

from sunode import basic
from sunode.basic import lib, ffi, data_dtype, index_dtype, Borrows, notnull, as_numpy, CPointer, RefCount

__all__ = ["empty_matrix", "Sparse", "Dense", "Band"]


logger = logging.getLogger("sunode.matrix")


MATRIX_TYPES = {
    "sparse": lib.SUNMATRIX_SPARSE,
    "dense": lib.SUNMATRIX_DENSE,
    "band": lib.SUNMATRIX_BAND,
}

MATRIX_TYPES_REV = {v: k for k, v in MATRIX_TYPES.items()}

@overload
def empty_matrix(
    shape: Tuple[int, int],
    kind: Literal["dense"],
    format: Literal[None],
    sparsity: Literal[None],
) -> Dense:
    ...

@overload
def empty_matrix(
    shape: Tuple[int, int],
    kind: str = "dense",
    format: Optional[str] = None,
    sparsity: Union[None, np.ndarray, sparse.csr_matrix, sparse.csc_matrix] = None,
) -> Matrix:
    ...

def empty_matrix(
    shape: Tuple[int, int],
    kind: str = "dense",
    format: Optional[str] = None,
    sparsity: Union[None, np.ndarray, sparse.csr_matrix, sparse.csc_matrix] = None,
) -> Matrix:
    rows, columns = shape
    if rows < 0 or columns < 0:
        raise ValueError("Number of rows and columns must not be negative.")
    if kind == "dense":
        ptr = lib.SUNDenseMatrix(rows, columns)
        if ptr == ffi.NULL:
            raise MemoryError("Could not allocate matrix.")
        return Dense(ptr)
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
        matrix = Sparse(ptr)
        matrix.indptr[...] = sparsity.indptr
        matrix.indices[...] = sparsity.indices
        return matrix
    else:
        raise ValueError("Unknown matrix type %s" % kind)


class Matrix(Borrows, metaclass=ABCMeta):
    dtype: ClassVar[np.dtype] = np.dtype(data_dtype.name)
    index_dtype: ClassVar[np.dtype] = np.dtype(index_dtype.name)

    c_ptr: CPointer

    @abstractproperty
    def data(self) -> np.ndarray:
        pass

    @abstractproperty
    def shape(self) -> Tuple[int, int]:
        pass

    def __len__(self) -> int:
        return self.shape[0]

class Sparse(Matrix):
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
        return as_numpy(self, ptr, size, self.index_dtype, self._buffer_refcount)

    @property
    def indptr(self) -> np.ndarray:
        size = lib.SUNSparseMatrix_NP(self.c_ptr)
        size += 1  #
        ptr = lib.SUNSparseMatrix_IndexPointers(self.c_ptr)
        return as_numpy(self, ptr, size, self.index_dtype, self._buffer_refcount)

    @property
    def data(self) -> np.ndarray:
        size = self.nnz
        ptr = lib.SUNSparseMatrix_Data(self.c_ptr)
        return as_numpy(self, ptr, size, self.dtype, self._buffer_refcount)

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


class Band(Matrix):
    pass


class Dense(Matrix):
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
        array = as_numpy(self, ptr, size, self.dtype)

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
        return Sparse(ptr)

    def c_print(self, file: Optional[TextIO] = None) -> None:
        if file is None:
            file = sys.stdout
        lib.SUNDenseMatrix_Print(self.c_ptr, file)
