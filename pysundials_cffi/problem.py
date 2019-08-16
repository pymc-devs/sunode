from typing import Union, TypeVar, Tuple
from typing_extensions import Protocol
from .basic import DenseMatrix, SparseMatrix, BandMatrix


T = TypeVar("T")
Maybe = Union[NotImplemented, T]
Matrix = Union[DenseMatrix, SparseMatrix, BandMatrix]


class OdeProblem(Protocol):
    n_states: int
    n_sensitivity: int

    def request_dy_func(self) -> Maybe[DyCallback]:
        ...

    def request_jac_func(self, kind) -> Maybe[Tuple[Matrix, JacCallback]]:
        ...

    def request_dinit_dp_func(self) -> Maybe[DinitDpCallback]:
        ...

    def request_dy_dp_all_func(self) -> Maybe[DyDpAllCallback]:
        ...

    def request_dy_dp_one_func(self) -> Maybe[DyDpOneCallback]:
        ...


class SympyOde(OdeProblem):
    def __init__(self):
        pass

    def add_dy(self, dy):
        pass

    def request_dy(self, blubb):
        pass

    def request_jac_func(self, kind) -> Maybe[Tuple[Matrix, JacCallback]]:
        if kind == "dense":
            pass
        elif kind == "band":
            pass
        elif kind == "sparse-csr":
            pass
        elif kind == "sparse-csc":
            pass
        elif kind == "jacmult":
            pass
        else:
            return NotImplemented

        raise FileNotFoundError()

    @property
    def n_states(self) -> None:
        pass

    @property
    def n_sensitivity(self):
        pass

    def request_dinit_dp_func(self):
        pass

    def request_dy_dp_all_func(self):
        pass

    def request_dy_dp_one_func(self):
        pass


class JaxOde(OdeProblem):
    pass
