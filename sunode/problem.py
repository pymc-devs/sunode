from __future__ import annotations

from typing import Union, TypeVar, Tuple, Callable, Any
from typing_extensions import Protocol
from sunode.basic import Matrix, DenseMatrix, SparseMatrix, BandMatrix
from sunode.builder import Option, Builder
from sunode import _cvodes
import numba  # type: ignore
import numpy as np

lib = _cvodes.lib
ffi = _cvodes.ffi


class Ode(Protocol):
    all_params_type: np.dtype
    params_type: np.dtype
    user_data_type: np.dtype
    state_type: np.dtype
    n_params: int
    n_states: int

    def make_rhs(self):
        pass

    def make_rhs_jac_dense(self):
        pass

    def make_rhs_jac_sparse(self):
        pass

    def make_rhs_jac_band(self):
        pass

    def make_sensitivity_rhs(self):
        pass

    def make_sensitivity_rhs_one(self):
        pass

    def make_adjoint_rhs(self):
        pass

    def make_adjoint_quad_rhs(self):
        pass

    def make_user_data(self):
        pass

    def update_changeable(self, user_data, params):
        pass

    def extract_params(self, user_data, out=None):
        pass

    @property
    def n_states(self):
        return sum(
            np.prod(self.state_type[name].shape, dtype=int)
            for name in self.state_type.names
        )

    def solution_to_xarray(self, tvals, solution, user_data, sensitivity=None,
                           *, unstack_state=True, unstack_params=True):
        import xarray as xr

        assert sensitivity is None, 'TODO'
        solution = solution.view(self.state_type)[..., 0]
        params = self.extract_params(user_data)

        def as_dict(array, prepend=None):
            if prepend is None:
                prepend = []
            dtype = array.dtype
            out = {}
            for name in dtype.names:
                if array[name].dtype == np.float64:
                    out['_'.join(prepend + [name])] = array[name]
                else:
                    out.update(as_dict(array[name], prepend + [name]))
            return out

        data = xr.Dataset()
        data['time'] = ('time', tvals)
        # TODO t0?
        if unstack_state:
            state = as_dict(solution, ['solution'])
            for name in state:
                assert name not in data
                data[name] = ('time', state[name])
        else:
            data['solution'] = ('time', solution)

        if unstack_params:
            params = as_dict(params, ['parameters'])
            for name in params:
                assert name not in data
                data[name] = params[name]
        else:
            data['parameters'] = params

        return data

    def make_sundials_rhs(self):
        rhs = self.make_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        user_dtype = self.user_data_type
        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)
        func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFn'))
        func_type = func_type.return_type(*(func_type.args[:-1] + (user_ndtype_p,)))

        @numba.cfunc(func_type)
        def rhs_wrapper(t, y_, out_, user_data_):
            y_ptr = N_VGetArrayPointer_Serial(y_)
            n_vars = N_VGetLength_Serial(y_)
            out_ptr = N_VGetArrayPointer_Serial(out_)
            y = numba.carray(y_ptr, (n_vars,))
            out = numba.carray(out_ptr, (n_vars,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            rhs(out, t, y, user_data)
            return 0

        return rhs_wrapper

    def make_sundials_adjoint_rhs(self):
        user_dtype = self.user_data_type
        adj = self.make_adjoint_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFnB'))
        args = list(func_type.args)
        args[-1] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def adj_rhs_wrapper(t, y_, yB_, yBdot_, user_data_):
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n_vars,))

            yB_ptr = N_VGetArrayPointer_Serial(yB_)
            yB = numba.carray(yB_ptr, (n_vars))

            yBdot_ptr = N_VGetArrayPointer_Serial(yBdot_)
            yBdot = numba.carray(yBdot_ptr, (n_vars,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            adj(
                yBdot,
                t,
                y,
                yB,
                user_data,
            )
            return 0

        return adj_rhs_wrapper

    def make_sundials_adjoint_quad_rhs(self):
        user_dtype = self.user_data_type
        adjoint_quad = self.make_adjoint_quad_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVQuadRhsFnB'))
        args = list(func_type.args)
        args[-1] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def quad_rhs_wrapper(t, y_, yB_, yBdot_, user_data_):
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n_vars,))

            yB_ptr = N_VGetArrayPointer_Serial(yB_)
            n_params = N_VGetLength_Serial(yB_)
            yB = numba.carray(yB_ptr, (n_params,))

            yBdot_ptr = N_VGetArrayPointer_Serial(yBdot_)
            yBdot = numba.carray(yBdot_ptr, (n_params,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            adjoint_quad(
                yBdot,
                t,
                y,
                yB,
                user_data,
            )
            return 0

        return quad_rhs_wrapper

    def make_sundials_sensitivity_rhs(self):
        sens_rhs = self.make_sensitivity_rhs()
        user_dtype = self._user_data_type

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVSensRhsFn'))
        args = list(func_type.args)
        args[-3] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def sens_rhs_wrapper(n_params, t, y_, ydot_, yS_, out_, user_data_, tmp1_, tmp2_):
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n_vars,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            tmp_sens_rhs = user_data.tmp_nparams_nvars
            tmp_sens_yS = user_data.tmp2_nparams_nvars

            
            for i in range(n_params):
                yS_i_ptr = N_VGetArrayPointer_Serial(yS_[i])
                yS_i = numba.carray(yS_i_ptr, (n_vars,))
                tmp_sens_yS[i, :] = yS_i

            sens_rhs(
                tmp_sens_rhs,
                t,
                y,
                tmp_sens_yS,
                user_data,
            )

            for i in range(n_params):
                out_i_ptr = N_VGetArrayPointer_Serial(out_[i])
                out_i = numba.carray(out_i_ptr, (n_vars,))
                out_i[:] = tmp_sens_rhs[i][:]

            return 0

        return sens_rhs_wrapper
