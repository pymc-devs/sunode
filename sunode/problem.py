from __future__ import annotations

from typing import Union, TypeVar, Tuple, Callable, Any, Optional, Dict
from typing_extensions import Protocol

import numba
import numpy as np

from sunode.matrix import Matrix, Dense, Sparse, Band
from sunode.basic import lib, ffi
from sunode.dtypesubset import as_nested, DTypeSubset


class Problem(Protocol):
    params_dtype: np.dtype
    params_subset: DTypeSubset
    state_dtype: np.dtype
    state_subset: DTypeSubset
    user_data_dtype: np.dtype

    def make_rhs(self):  # type: ignore
        pass

    def make_rhs_jac_dense(self):  # type: ignore
        return NotImplemented

    def make_rhs_sparse_jac_template(self) -> Sparse:
        return NotImplemented

    def make_rhs_jac_sparse(self):  # type: ignore
        return NotImplemented

    def make_rhs_jac_band(self):  # type: ignore
        return NotImplemented

    def make_sensitivity_rhs(self):  # type: ignore
        return NotImplemented

    def make_sensitivity_rhs_one(self):  # type: ignore
        return NotImplemented

    def make_adjoint_rhs(self):  # type: ignore
        return NotImplemented

    def make_adjoint_quad_rhs(self):  # type: ignore
        return NotImplemented

    def make_rhs_jac_prod(self):  # type: ignore
        return NotImplemented

    def make_user_data(self) -> np.ndarray:
        return np.recarray((), dtype=self.user_data_dtype)

    def update_params(self, user_data: np.ndarray, params: np.ndarray) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `update_params`.')
        user_data.fill(params)

    def update_subset_params(self, user_data: np.ndarray, params: np.ndarray) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `update_subset_params`.')
        user_data.view(self.params_subset.subset_view_dtype).fill(params)

    def update_remaining_params(self, user_data: np.ndarray, params: np.ndarray) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `update_subset_params`.')
        user_data.view(self.params_subset.remainder.subset_view_dtype).fill(params)

    def extract_params(self, user_data: np.ndarray, out: Optional[np.ndarray] = None) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `extract_params`.')
        if out is None:
            out = np.empty((1,), dtype=self.params_subset.dtype)[0]
        out.fill(user_data)

    def extract_subset_params(self, user_data: np.ndarray, out: Optional[np.ndarray] = None) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `extract_subset_params`.')
        if out is None:
            out = np.empty((1,), dtype=self.params_subset.subset_dtype)[0]
        out.fill(user_data.view(self.params_subset.subset_dtype))

    def extract_remaining_params(self, user_data: np.ndarray, out: Optional[np.ndarray] = None) -> None:
        if not self.user_data_dtype == self.params_dtype:
            raise ValueError('Problem needs to overwrite `extract_remaining_params`.')
        subset_dtype = self.params_subset.remainder.subset_view_dtype
        dtype = self.params_subset.remainder.subset_dtype
        if out is None:
            out = np.empty((1,), dtype=dtype)[0]
        out.fill(user_data.view(subset_dtype))

    @property
    def n_states(self) -> int:
        return self.state_subset.n_items

    @property
    def n_params(self) -> int:
        return self.params_subset.n_subset

    def solution_to_xarray(  # type: ignore
        self, tvals, solution, user_data, sensitivity=None,
        *, unstack_state=True, unstack_params=True
    ):
        import xarray as xr

        assert sensitivity is None, 'TODO'
        solution = solution.view(self.state_dtype)[..., 0]
        params = self.extract_params(user_data)

        def as_dict(array, prepend=None):  # type: ignore
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
        #  TODO t0?
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

    def flat_solution_as_dict(self, solution: np.ndarray) -> Dict[str, Any]:
        slices = self.state_subset.flat_slices
        shapes = self.state_subset.flat_shapes
        flat_views = {}
        for path in self.state_subset.paths:
            shape = (-1,) + shapes[path]
            flat_views[path] = solution[:, slices[path]].reshape(shape)
        return as_nested(flat_views)

    def make_sundials_rhs(self) -> Any:
        rhs = self.make_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        state_dtype = self.state_dtype
        user_dtype = self.user_data_dtype
        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)
        func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFn'))
        func_type = func_type.return_type(*(func_type.args[:-1] + (user_ndtype_p,)))

        @numba.cfunc(func_type)
        def rhs_wrapper(t, y_, out_, user_data_):  # type: ignore
            y_ptr = N_VGetArrayPointer_Serial(y_)
            n_vars = N_VGetLength_Serial(y_)
            out_ptr = N_VGetArrayPointer_Serial(out_)
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            out = numba.carray(out_ptr, (n_vars,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            return rhs(out, t, y, user_data)

        return rhs_wrapper

    def make_sundials_adjoint_rhs(self):  # type: ignore
        user_dtype = self.user_data_dtype
        adj = self.make_adjoint_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        state_dtype = self.state_dtype
        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFnB'))
        args = list(func_type.args)
        args[-1] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def adj_rhs_wrapper(t, y_, yB_, yBdot_, user_data_):  # type: ignore
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]

            yB_ptr = N_VGetArrayPointer_Serial(yB_)
            yB = numba.carray(yB_ptr, (n_vars,))

            yBdot_ptr = N_VGetArrayPointer_Serial(yBdot_)
            yBdot = numba.carray(yBdot_ptr, (n_vars,))

            #print(n_vars)
            #print(N_VGetLength_Serial(yB_))
            #print(N_VGetLength_Serial(yBdot_))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            return adj(
                yBdot,
                t,
                y,
                yB,
                user_data,
            )

        return adj_rhs_wrapper

    def make_sundials_adjoint_quad_rhs(self):  # type: ignore
        user_dtype = self.user_data_dtype
        adjoint_quad = self.make_adjoint_quad_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        state_dtype = self.state_dtype
        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVQuadRhsFnB'))
        args = list(func_type.args)
        args[-1] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def quad_rhs_wrapper(t, y_, yB_, qBdot_, user_data_):  # type: ignore
            n = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n,)).view(state_dtype)[0]

            yB_ptr = N_VGetArrayPointer_Serial(yB_)
            n = N_VGetLength_Serial(yB_)
            yB = numba.carray(yB_ptr, (n,))

            qBdot_ptr = N_VGetArrayPointer_Serial(qBdot_)
            n = N_VGetLength_Serial(qBdot_)
            qBdot = numba.carray(qBdot_ptr, (n,))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            return adjoint_quad(
                qBdot,
                t,
                y,
                yB,
                user_data,
            )

        return quad_rhs_wrapper

    def make_sundials_sensitivity_rhs(self):  # type: ignore
        sens_rhs = self.make_sensitivity_rhs()
        user_dtype = self.user_data_dtype

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        state_dtype = self.state_dtype
        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVSensRhsFn'))
        args = list(func_type.args)
        args[-3] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def sens_rhs_wrapper(  # type: ignore
            n_params, t, y_, ydot_, yS_, out_, user_data_, tmp1_, tmp2_
        ):
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]

            out_array = user_data.tmp_nparams_nstates
            yS_array = user_data.tmp2_nparams_nstates
            for i in range(n_params):
                yS_i_ptr = N_VGetArrayPointer_Serial(yS_[i])
                yS_i = numba.carray(yS_i_ptr, (n_vars,))
                yS_array[i, :] = yS_i

            retcode = sens_rhs(out_array, t, y, yS_array, user_data)
            if retcode != 0:
                return retcode

            for i in range(n_params):
                out_i_ptr = N_VGetArrayPointer_Serial(out_[i])
                out_i = numba.carray(out_i_ptr, (n_vars,))
                out_i[:] = out_array[i, :]

            return retcode

        return sens_rhs_wrapper

    def make_sundials_adjoint_jac_dense(self):  # type: ignore
        jac_dense = self.make_adjoint_jac_dense()
        user_dtype = self.user_data_dtype
        state_dtype = self.state_dtype

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial
        SUNDenseMatrix_Data = lib.SUNDenseMatrix_Data

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVLsJacFnB'))
        args = list(func_type.args)
        args[5] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def jac_dense_wrapper(t, y_, yB_, fyB_, out_, user_data_, tmp1_, tmp2_, tmp3_):  # type: ignore
            n_vars = N_VGetLength_Serial(y_)
            n_lamda = N_VGetLength_Serial(yB_)

            y_ptr = N_VGetArrayPointer_Serial(y_)
            yB_ptr = N_VGetArrayPointer_Serial(yB_)
            fyB_ptr = N_VGetArrayPointer_Serial(fyB_)
            out_ptr = SUNDenseMatrix_Data(out_)

            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            yB = numba.carray(yB_ptr, (n_lamda,))
            fyB = numba.carray(fyB_ptr, (n_lamda,))
            out = numba.farray(out_ptr, (n_lamda, n_lamda))

            user_data = numba.carray(user_data_, (1,), user_dtype)[0]
            
            return jac_dense(out, t, y, yB, fyB, user_data)

        return jac_dense_wrapper

    def make_sundials_jac_dense(self):  # type: ignore
        jac_dense = self.make_jac_dense()
        user_dtype = self.user_data_dtype
        state_dtype = self.state_dtype

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial
        SUNDenseMatrix_Data = lib.SUNDenseMatrix_Data

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVLsJacFn'))
        args = list(func_type.args)
        args[4] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def jac_dense_wrapper(t, y_, fy_, out_, user_data_, tmp1_, tmp2_, tmp3_):  # type: ignore
            n_vars = N_VGetLength_Serial(y_)
            y_ptr = N_VGetArrayPointer_Serial(y_)
            out_ptr = SUNDenseMatrix_Data(out_)
            fy_ptr = N_VGetArrayPointer_Serial(fy_)
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            out = numba.farray(out_ptr, (n_vars, n_vars))
            fy = numba.carray(fy_ptr, (n_vars,))
            user_data = numba.carray(user_data_, (1,), user_dtype)[0]
            
            return jac_dense(out, t, y, fy, user_data)

        return jac_dense_wrapper

    def make_sundials_jac_sparse(self, format='CSR'):  # type: ignore
        jac_sparse = self.make_jac_sparse(format=format)

        user_dtype = self.user_data_dtype
        state_dtype = self.state_dtype

        N_VGetArrayPointer = lib.N_VGetArrayPointer
        N_VGetLength = lib.N_VGetLength
        SUNSparseMatrix_Data = lib.SUNSparseMatrix_Data

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVLsJacFn'))
        args = list(func_type.args)
        args[4] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def jac_dense_wrapper(t, y_, fy_, out_, user_data_, tmp1_, tmp2_, tmp3_):  # type: ignore
            n_vars = N_VGetLength(y_)
            y_ptr = N_VGetArrayPointer(y_)
            out_ptr = SUNSparseMatrix_Data(out_)
            fy_ptr = N_VGetArrayPointer(fy_)
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            out = numba.farray(out_ptr, (n_vars, n_vars))
            fy = numba.carray(fy_ptr, (n_vars,))
            user_data = numba.carray(user_data_, (1,), user_dtype)[0]
            
            return jac_dense(out, t, y, fy, user_data)

        return jac_dense_wrapper

    def make_sundials_jac_prod(self):  # type: ignore
        jac_prod = self.make_rhs_jac_prod()

        user_dtype = self.user_data_dtype
        state_dtype = self.state_dtype

        N_VGetArrayPointer = lib.N_VGetArrayPointer
        N_VGetLength = lib.N_VGetLength_Serial

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVLsJacTimesVecFn'))
        args = list(func_type.args)
        args[-2] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def jac_prod_wrapper(v_, out_, t, y_, fy_, user_data_, tmp_,):  # type: ignore
            n_vars = N_VGetLength(v_)

            v_ptr = N_VGetArrayPointer(v_)
            out_ptr = N_VGetArrayPointer(out_)
            y_ptr = N_VGetArrayPointer(y_)
            fy_ptr = N_VGetArrayPointer(fy_)

            v = numba.carray(v_ptr, (n_vars,))
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            out = numba.carray(out_ptr, (n_vars,))
            fy = numba.carray(fy_ptr, (n_vars,))
            user_data = numba.carray(user_data_, (1,), user_dtype)[0]
            
            return jac_prod(out, v, t, y, fy, user_data)

        return jac_prod_wrapper

    def make_sundials_adjoint_jac_prod(self):  # type: ignore
        jac_prod = self.make_adjoint_jac_prod()

        user_dtype = self.user_data_dtype
        state_dtype = self.state_dtype

        N_VGetArrayPointer = lib.N_VGetArrayPointer
        N_VGetLength = lib.N_VGetLength_Serial

        user_ndtype = numba.from_dtype(user_dtype)
        user_ndtype_p = numba.types.CPointer(user_ndtype)

        func_type = numba.cffi_support.map_type(ffi.typeof('CVLsJacTimesVecFnB'))
        args = list(func_type.args)
        args[-2] = user_ndtype_p
        func_type = func_type.return_type(*args)

        @numba.cfunc(func_type)
        def jac_prod_wrapper(vB_, out_, t, y_, yB_, fyB_, user_data_, tmp_,):  # type: ignore
            n_vars = N_VGetLength(y_)
            n_varsB = N_VGetLength(vB_)

            if n_vars != n_varsB:
                return -1

            vB_ptr = N_VGetArrayPointer(vB_)
            out_ptr = N_VGetArrayPointer(out_)
            y_ptr = N_VGetArrayPointer(y_)
            yB_ptr = N_VGetArrayPointer(yB_)
            fyB_ptr = N_VGetArrayPointer(fyB_)

            vB = numba.carray(vB_ptr, (n_vars,))
            out = numba.carray(out_ptr, (n_vars,))
            y = numba.carray(y_ptr, (n_vars,)).view(state_dtype)[0]
            yB = numba.carray(yB_ptr, (n_vars,))
            fyB = numba.carray(fyB_ptr, (n_vars,))
            user_data = numba.carray(user_data_, (1,), user_dtype)[0]
            
            return jac_prod(out, vB, t, y, yB, fyB, user_data)

        return jac_prod_wrapper
