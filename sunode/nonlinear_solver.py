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


logger = logging.getLogger("sunode.nonlinear_solver")


class NonlinearSolver(Borrows):
    def __init__(self, c_ptr):
        check_ptr(c_ptr)
        self.c_ptr = c_ptr

        def finalize(c_ptr: CPointer, release_borrowed: Callable[[], None]) -> None:
            if c_ptr == ffi.NULL:
                logger.warn("Trying to free Solver, but it is NULL.")
            else:
                logger.debug("Freeing Solver")
                lib.SUNNonlinSolFree(c_ptr)
            release_borrowed()
        weakref.finalize(self, finalize, self.c_ptr, self.release_borrowed_func())
