import dataclasses
from typing import Optional, Union

import xarray  # type: ignore

from pysundials_cffi.builder import Builder, BuilderOption
from pysundials_cffi.basic import (
    Vector,
    DenseMatrix,
    SparseMatrix,
    empty_vector,
    empty_matrix,
    LinearSolver,
)

from pysundials_cffi.problem import OdeProblem

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
        super().__init__()
        self._problem = problem
        self._build_data = SolverBuildData()
        self._option_data = SolverOptionData()
        self._add([jacobian()])

    def solve(self) -> xarray.Dataset:
        raise NotImplementedError()


class SolverOption(BuilderOption[SolverBuilder]):
    def __init__(self) -> None:
        self._builder: Optional[SolverBuilder] = None

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def builder(self) -> SolverBuilder:
        if self._builder is None:
            raise ValueError('Option can only be called through the Builder.')
        return self._builder

    def take_builder(self, builder: SolverBuilder) -> None:
        self._builder = builder

    def release_builder(self) -> None:
        self._builder = None


@SolverBuilder._option
class vector_backend(SolverOption):
    def __call__(self, kind: str) -> SolverBuilder:
        assert kind in ["serial"]
        self.builder._option_data.vector_backend = kind
        return self.builder

    def build(self) -> None:
        ndim = self.builder._problem.n_states
        kind = self.builder._option_data.vector_backend
        if kind is None:
            kind = "serial"
        vector = empty_vector(ndim, kind=kind)
        self.builder._build_data.y_template = vector


@SolverBuilder._option
class jacobian(SolverOption):
    def __call__(self, kind: str) -> SolverBuilder:
        """Hallo"""
        assert kind in ["dense", "sparse"]
        self.builder._option_data.jacobian = kind
        return self.builder

    def build(self) -> None:
        ndim = self.builder._problem.n_states
        kind = self.builder._option_data.jacobian
        if kind is None:
            kind = "dense"
        matfunc = self.builder._problem.request_jac_func(kind)
        if matfunc is None:
            raise ValueError("Problem does not support jacobian")
        self.builder._opt
