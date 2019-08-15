from .builder import Builder, BuilderOption
from .basic import Vector, DenseMatrix, SparseMatrix, empty_vector, empty_matrix
from .problem import OdeProblem
from .linear_solver import LinearSolver
#from .linear_solver import linear_solver
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
        pass


class SolverOption(BuilderOption):
    def __init__(self, builder: SolverBuilder) -> None:
        self._builder: SolverBuilder = builder


@SolverBuilder._option
class vector_backend(SolverOption):
    def __call__(self, kind: str) -> SolverBuilder:
        assert kind in ['serial']
        self._builder._option_data.vector_backend = kind
        return self._builder

    def build(self) -> None:
        ndim = self._builder._problem.n_states
        kind = self._builder._option_data.vector_backend
        if kind is None:
            kind = 'serial'
        vector = empty_vector(ndim, kind=kind)
        self._builder._build_data.y_template = vector
