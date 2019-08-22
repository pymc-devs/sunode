import logging
from typing import Optional, Tuple, List, cast

from pysundials_cffi import _cvodes
from pysundials_cffi.basic import notnull, DenseMatrix, SparseMatrix
from pysundials_cffi import basic
from pysundials_cffi.solver import SolverBuilder, SolverOption, jacobian


__all__ = ["linear_solver"]

logger = logging.getLogger("pysundials_cffi.linear_solver")

lib = _cvodes.lib
ffi = _cvodes.ffi


LINEAR_SOLVERS = {"direct": lib.SUNLINEARSOLVER_DIRECT}


@SolverBuilder._option
class superlu_threads(SolverOption):
    def __call__(self, num_threads: int):
        """Change the number of threads superlu will use for matrix factorization."""
        self.builder._option_data.superlu_threads = num_threads
        # The option is used in `linear_solver` in the constructor of the solver.


@SolverBuilder._option
class superlu_ordering(SolverOption):
    def __call_(self, ordering: str) -> SolverBuilder:
        """Choose which ordering method SuperLU should use.

        Options are:
        - 'natural' for a natural ordering.
        - 'AtA' for a minimal degree ordering on :math:`A^TA`
        - 'At+A' for a minimal degree ordering on :math:`A^T + A`
        - 'colamd' (default) for a COLAMD ordering for unsymmetric matrices.
        """
        num = {"natural": 0, "AtA": 1, "At+A": 2, "colamd": 3}
        self.superlu_ordering = num[ordering]
        return self.builder

    def build(self) -> None:
        if hasattr(self, "superlu_ordering"):
            ret = lib.SUNLinSol_SuperLUMTSetOrdering(self.superlu_ordering)
            if ret != lib.SUNLS_SUCCESS:
                raise ValueError("Could not set ordering.")


@SolverBuilder._option
class klu_ordering(SolverOption):
    def __call__(self, ordering: str) -> SolverBuilder:
        """Change the ordering klu will use for matrix factorization."""
        self.ordering = ordering
        return self.builder

    def build(self) -> None:
        if not hasattr(self, "ordering"):
            return
        if self.ordering == 1:  # NotImplementedError
            lib.SUNLinSol_KLUSetOrdering()


@SolverBuilder._option
class linear_solver(SolverOption):
    def __call__(self, kind: Optional[str] = None) -> SolverBuilder:
        jac_type = self.builder._option_data.jacobian
        if kind is None:
            if jac_type is None:
                cast(jacobian, self.builder.jacobian)("dense")
                jac_type = self.builder._option_data.jacobian

            if jac_type == "dense":
                kind = "dense"
            elif jac_type == "band":
                kind = "band"
            elif jac_type == "sparse":
                kind = "klu"
            elif jac_type == "matmult":
                raise NotImplementedError()
            else:
                raise ValueError("Unknown jacobian %s." % jac_type)
        elif jac_type is None:
            jac_option = cast(jacobian, self.builder.jacobian)
            if kind in ["dense", "lapack-dense"]:
                jac_option("dense")
            elif kind in ["band", "lapack-band"]:
                jac_option("band")
            elif kind in ["klu", "superlu"]:
                jac_option("sparse")
            elif kind in [...]:
                raise NotImplementedError()
            else:
                raise ValueError("Unknown linear solver %s" % kind)

        jac_type = self.builder._option_data.jacobian
        opts: List[SolverOption] = []
        required: List[str] = []
        if kind in ["klu", "superlu"]:
            opts_, required_ = self._prepare_sparse_options(kind)
            opts.extend(opts_)
            required.extend(required_)
        elif kind in ["dense", "band", "lapack-dense", "lapack-band"]:
            pass
        else:
            raise NotImplementedError()

        self.builder._add(opts)
        self.builder._make_required(required)
        self._dependent_options = opts

        self.kind = kind
        return self.builder

    def _prepare_sparse_options(
        self, kind: str
    ) -> Tuple[List[SolverOption], List[str]]:
        opts: List[SolverOption] = []
        req: List[SolverOption] = []
        if kind == "klu":
            pass
        raise NotImplementedError()

        return opts, req

    def build(self) -> None:
        if self.kind in []:
            self._build_matvec(self.kind)
        else:
            self._build_direct(self.kind)

    def _build_matvec(self, kind: str) -> None:
        raise NotImplementedError()

    def _build_direct(self, kind: str) -> None:
        data = self.builder._build_data
        y = data.y_template
        jac = data.jac_template
        assert y is not None
        assert jac is not None

        if kind == "dense":
            assert isinstance(jac, DenseMatrix)
            ptr = lib.SUNLinSol_Dense(y.c_ptr, jac.c_ptr)
        elif kind == "band":
            raise NotImplementedError()
            assert isinstance(jac, basic.BandMatrix)
            ptr = lib.SUNLinSol_Band(y.c_ptr, jac.c_ptr)
        elif kind == "lapack-dense":
            assert isinstance(jac, DenseMatrix)
            ptr = lib.SUNLinSol_LapackDense(y.c_ptr, jac.c_ptr)
        elif kind == "lapack-band":
            raise NotImplementedError()
            assert isinstance(jac, basic.BandMatrix)
            ptr = lib.SUNLinSol_LapackBand(y.c_ptr, jac.c_ptr)
        elif kind == "klu":
            assert isinstance(jac, SparseMatrix)
            ptr = lib.SUNLinSol_KLU(y.c_ptr, jac.c_ptr)
        elif kind == "superlu":
            assert isinstance(jac, SparseMatrix)
            num_threads = self.builder._option_data.superlu_threads
            if num_threads is None:
                num_threads = 1
            ptr = lib.SUNLinSol_SuperLUMT(y.c_ptr, jac.c_ptr, num_threads)
        else:
            raise ValueError("Unknown linear solver %s" % kind)

        solver = basic.LinearSolver(ptr)
        solver.borrow(y)
        solver.borrow(jac)
        data.linear_solver = solver

        notnull(ptr, "Linear solver creation failed.")
        for option in self._dependent_options:
            option.build()

        solver.initialize()
