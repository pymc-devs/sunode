from __future__ import annotations

from typing import Union, TypeVar, Tuple, Callable, Any
from typing_extensions import Protocol
from pysundials_cffi.basic import Matrix, DenseMatrix, SparseMatrix, BandMatrix
from pysundials_cffi.builder import Option, Builder
from pysundials_cffi import _cvodes
import numba  # type: ignore

lib = _cvodes.lib
ffi = _cvodes.ffi

#T = TypeVar("T")
#Maybe = Union[NotImplemented, T]
#Matrix = Union[DenseMatrix, SparseMatrix, BandMatrix]


"""
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
"""


@Builder._option
class dy_numba(Option):
    def __call__(self, func: Any, use_dtypes: bool = True) -> None:
        data = self.builder._option_data
        if data.extra_dtype is None:
            self.builder.extra_dtype()
        extra_dtype = data.extra_dtype
        assert extra_dtype is not None

        self.dy_numba = func
        self.use_dtypes = use_dtypes

    def build(self) -> None:
        data = self.builder._build_data
        options = self.builder._option_data

        state_dtype = data.state_dtype
        user_data_dtype = data.user_data_dtype

        if options.vector_backend == 'serial':
            N_VGetArrayPointer = lib.N_VGetArrayPointer_Serial
        else:
            raise NotImplementedError()

        func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFn'))
        assert data.y_template is not None
        ndim = len(data.y_template)
        #func_type = func_type.return_type(*(func_type.args[:-1] + (user_ndtype_p,)))

        rhs = self.dy_numba

        if self.use_dtypes:
            @numba.cfunc(func_type)
            def rhs_wrapper(t, y_, out_, user_data_):
                y_ptr = N_VGetArrayPointer(y_)
                out_ptr = N_VGetArrayPointer(out_)
                y = numba.carray(y_ptr, (), state_dtype)
                out = numba.carray(out_ptr, (), state_dtype)
                user_data = numba.carray(user_data_, (), user_data_dtype)
                extra = user_data['extra']
                grad = user_data['grad_params']
                section = user_data['section']
                
                return rhs(out, t, y, grad, extra, section)
        else:
            @numba.cfunc(func_type)
            def rhs_wrapper(t, y_, out_, user_data_):
                y_ptr = N_VGetArrayPointer(y_)
                out_ptr = N_VGetArrayPointer(out_)
                y = numba.carray(y_ptr, (ndim,))
                out = numba.carray(out_ptr, (ndim,))
                user_data = numba.carray(user_data_, (), user_data_dtype)
                extra = user_data['extra']
                grad = user_data['grad_params']
                section = user_data['section']
                
                return rhs(out, t, y, grad, extra, section)

"""
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
"""
