from pysundials_cffi.builder import Builder, BuilderOption
from pysundials_cffi.basic import (
    Vector,
    DenseMatrix,
    SparseMatrix,
    empty_vector,
    empty_matrix,
)
from pysundials_cffi.problem import OdeProblem
from pysundials_cffi.linear_solver import LinearSolver
import xarray  # type: ignore
import dataclasses
from typing import Optional, Union
from typing_extensions import Protocol


Matrix = Union[DenseMatrix, SparseMatrix]


@dataclasses.dataclass
class SolverOptionData:
    vector_backend: Optional[str] = None
    jacobian: Optional[str] = None
    superlu_threads: Optional[int] = None
    klu_ordering: Optional[str] = None


@dataclasses.dataclass
class SolverBuildData:
    y_template: Optional[Vector] = None
    jac_template: Optional[Matrix] = None
    linear_solver: Optional[LinearSolver] = None


class SolverBuilder(Builder):
    def __init__(self, problem: OdeProblem) -> None:
        self._problem = problem
        self._build_data = SolverBuildData()
        self._option_data = SolverOptionData()

    def solve(self) -> xarray.Dataset:
        raise NotImplementedError()


class SolverOption(BuilderOption):
    def __init__(self, builder: SolverBuilder) -> None:
        self._builder: SolverBuilder = builder


@SolverBuilder._option
class vector_backend(SolverOption):
    def __call__(self, kind: str) -> SolverBuilder:
        assert kind in ["serial"]
        self._builder._option_data.vector_backend = kind
        return self._builder._modify(["vector_backend"])

    def build(self) -> None:
        ndim = self._builder._problem.n_states
        kind = self._builder._option_data.vector_backend
        if kind is None:
            kind = "serial"
        vector = empty_vector(ndim, kind=kind)
        self._builder._build_data.y_template = vector


@SolverBuilder._option
class jacobian(SolverOption):
    def __call__(self, kind: str) -> SolverBuilder:
        assert kind in ["dense", "sparse"]
        self._builder._option_data.jacobian = kind
        return self._builder._modify(["jacobian"])

    def build(self) -> None:
        ndim = self._builder._problem.n_states
        kind = self._builder._option_data.jacobian
        if kind is None:
            kind = "dense"
        matfunc = self._builder._problem.request_jac_func(kind)
        if matfunc is None:
            raise ValueError("Problem does not support jacobian")
        self._builder._opt
