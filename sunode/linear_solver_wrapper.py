import logging
import weakref
from typing import Callable, Optional, Dict, List

from sunode import basic
from sunode.basic import (
    lib, ffi, data_dtype, index_dtype, Borrows,
    notnull, as_numpy, CPointer, check, check_ptr
)
from sunode.vector import Vector
from sunode.matrix import Matrix, Dense, Sparse, Band


logger = logging.getLogger("sunode.linear_solver")


class BaseLinearSolver(Borrows):
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

    @property
    def solver_type(self) -> str:
        assert self.c_ptr != 0
        type_int = lib.SUNLinSolGetType(self.c_ptr)
        if type_int == 0:
            return 'direct'
        elif type_int == 1:
            return 'iterative'
        elif type_int == 2:
            return 'matrix_iterative'
        else:
            raise RuntimeError(f'Sundials reported unknown solver type {type_int}')

    @property
    def id(self) -> int:
        assert self.c_ptr != 0
        return lib.SUNLinSolGetID(self.c_ptr)

    @property
    def num_iters(self) -> int:
        assert self.c_ptr != 0
        return lib.SUNLinSolNumIters(self.c_ptr)

    @property
    def resid_norm(self) -> float:
        assert self.c_ptr != 0
        return lib.SUNLinSolResidNorm(self.c_ptr)

    @property
    def last_flag(self) -> int:
        assert self.c_ptr != 0
        return lib.SUNLinSolLastFlag(self.c_ptr)

    def solve(self, A: Matrix, x: Vector, b: Vector, tol: float) -> None:
        assert self.c_ptr != 0
        assert A.c_ptr != 0
        assert b.c_ptr != 0
        assert x.c_ptr != 0
        check(lib.SUNLinSolSolve(self.c_ptr, A.c_ptr, x.c_ptr, b.c_ptr, tol))

    def initialize(self) -> None:
        ret = lib.SUNLinSolInitialize(self.c_ptr)
        if ret != 0:
            raise ValueError("Could not initialize linear solver.")


class LinearSolverDense(BaseLinearSolver):
    def __init__(self, vector: Vector, matrix: Dense):
        c_ptr = check_ptr(lib.SUNLinSol_Dense(vector.c_ptr, matrix.c_ptr))
        super().__init__(c_ptr)


class LinearSolverBand(BaseLinearSolver):
    def __init__(self, vector: Vector, matrix: Band):
        c_ptr = check_ptr(lib.SUNLinSol_Band(vector.c_ptr, matrix.c_ptr))
        super().__init__(c_ptr)


class LinearSolverLapackDense(BaseLinearSolver):
    def __init__(self, vector: Vector, matrix: Dense):
        c_ptr = check_ptr(lib.SunLinSol_LapackDense(vector.c_ptr, matrix.c_ptr))
        super().__init__(c_ptr)


class LinearSolverKLU(BaseLinearSolver):
    def __init__(self, vector: Vector, matrix: Sparse):
        c_ptr = check_ptr(lib.SunLinSol_KLU(vector.c_ptr, matrix.c_ptr))
        self._last_nnz = matrix.nnz
        super().__init__(c_ptr)

    def reinit(self, matrix: Sparse, nnz: int, reinit_type: str) -> None:
        assert self.c_ptr != 0
        assert matrix.c_ptr != 0
        if reinit_type == 'full':
            check(lib.SUNLinSol_KLUReInit(self.c_ptr, matrix.c_ptr, nnz, lib.SUNKLU_REINIT_FULL))
        elif reinit_type == 'partial':
            assert matrix.nnz <= nnz
            assert self._last_nnz >= nnz
            check(lib.SUNLinSol_KLUReInit(self.c_ptr, matrix.c_ptr, nnz, lib.SUNKLU_REINIT_PARTIAL))

    def set_ordering(self, ordering: str) -> None:
        assert self.c_ptr != 0
        if ordering == 'amd':
            check(lib.SUNLinSol_KLUSetOrdering(self.c_ptr, 0))
        elif ordering == 'colamd':
            check(lib.SUNLinSol_KLUSetOrdering(self.c_ptr, 1))
        elif ordering == 'natural':
            check(lib.SUNLinSol_KLUSetOrdering(self.c_ptr, 2))
