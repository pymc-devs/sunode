from __future__ import annotations

import numpy as np
import sympy as sym
import dataclasses
import numba
import xarray as xr

from sunode import problem
import sunode.basic
from sunode.symode.lambdify import lambdify_consts
from sunode.symode.paramset import DTypeSubset


def make_sympy_ode(params, states, rhs_sympy, *, grad_params=None):
    paramset = Paramset(params, grad_params)
    pass


class SympyOde(problem.Ode):
    def __init__(self, params, states, rhs_sympy, derivative_params, coords=None):
        #self.derivative_subset = DTypeSubset(params, derivative_params, coords=coords)  # TODO allow other dtypes
        self.derivative_subset = DTypeSubset(params, derivative_params, fixed_dtype=sunode.basic.data_dtype, coords=coords)
        self.remainder_subset = self.derivative_subset.remainder()
        self.coords = self.derivative_subset.coords
        self.params_dtype = self.derivative_subset.dtype
        self.state_subset = DTypeSubset(states, [], fixed_dtype=sunode.basic.data_dtype, coords=self.coords)
        self.state_dtype = self.state_subset.dtype
        self.coords = self.state_subset.coords

        self._rhs_sympy_func = rhs_sympy

        def check_dtype(dtype, path=None):
            if dtype.fields is None:
                if dtype.base != sunode.basic.data_dtype:
                    raise ValueError('Derivative param %s has incorrect dtype %s. Should be %s'
                                     % (path, path.base, sunode.basic.data_dtype))
                return
            for name, (dt, _) in dtype.fields.items():
                if path is None:
                    path_ = name
                else:
                    path_ = '.'.join([path, name])
                check_dtype(dt, path_)

        check_dtype(self.derivative_subset.subset_dtype)

        self._sym_time = sym.Symbol('t', real=True)
        n_fixed = self.derivative_subset.n_items - self.n_params
        self._sym_fixed = sym.MatrixSymbol('_sym_fixed', 1, n_fixed)
        self._sym_deriv = sym.MatrixSymbol('_sym_deriv', 1, self.n_params)
        fixed_items = np.array([self._sym_fixed[0, i] for i in range(n_fixed)], dtype=object)
        deriv_items = np.array([self._sym_deriv[0, i] for i in range(self.n_params)], dtype=object)
        self._sym_params = self.derivative_subset.as_dataclass('Params', deriv_items, fixed_items)

        self._sym_statevec = sym.MatrixSymbol('y', 1, self.n_states)
        state_items = np.array([self._sym_statevec[0, i] for i in range(self.n_states)], dtype=object)
        self._sym_states = self.state_subset.as_dataclass('State', [], state_items)

        rhs = self._rhs_sympy_func(self._sym_time, self._sym_states, self._sym_params)
        dims = sunode.symode.paramset.as_flattened(self.state_subset.dims)
        dims = {k: dim_names for k, (dtype, dim_names) in dims.items()}

        def as_flattened(path, value, shape, dims, coords):
            total = 1
            for length in shape:
                total *= length

            if hasattr(value, 'shape'):
                if value.shape != shape:
                    raise ValueError('Invalid shape for right-hand-side state %s. It is %s but we expected %s.'
                                     % (path, value.shape, shape))
                if hasattr(value, 'dims') and value.dims != dims:
                    raise ValueError('Invalid dims for right-hand-side state %s.' % path)

                if isinstance(value, sym.NDimArray):
                    return value.reshape(total)
                elif isinstance(value, xr.DataArray):
                    return value.data.ravel()
                else:
                    return value.reshape((total,))
            elif isinstance(value, list):
                if len(value) != shape[0]:
                    raise ValueError('Invalid shape for right-hand-side state %s.' % path)
                out = []
                for val in value:
                    out.extend(as_flattened(path, val, shape[1:], dims[1:], coords))
                return out
            elif isinstance(value, dict):
                if len(value) != shape[0]:
                    raise ValueError('Invalid shape for right-hand-side state %s.' % path)
                out = []
                for idx in coords[dims[0]]:
                    out.extend(as_flattened(path, value[idx], shape[1:], dims[1:], coords))
                return out
            elif shape == ():
                return [value]
            else:
                raise ValueError('Unknown righ-hand-side for state %s.' % path)

        #rhs = sunode.symode.paramset.as_flattened(rhs)
        rhs_list = []
        for path in self.state_subset.paths:
            item = rhs
            for name in path[:-1]:
                if name not in rhs:
                    raise ValueError('No right-hand-side for state %s' % '.'.join(path))
                item = item[name]
            item = item.pop(path[-1])

            item_dims = dims[path]
            item_dtype = self.state_dtype
            for name in path:
                item_dtype = item_dtype[name]

            name = '.'.join(path)
            rhs_list.extend(as_flattened(name, item, item_dtype.shape, item_dims, self.coords))

        rhs = sunode.symode.paramset.as_flattened(rhs)
        if rhs:
            keys = ['.'.join(path) for path in rhs.keys()]
            raise ValueError('Unknown state variables: %s' % keys)

        self._sym_dydt = sym.Matrix(rhs_list)
        self._sym_sens = sym.MatrixSymbol('sens', self.n_params, self.n_states)
        self._sym_lamda = sym.MatrixSymbol('lamda', 1, self.n_states)

        self._sym_dydt_jac = self._sym_dydt.jacobian(self._sym_statevec.as_explicit())
        self._sym_dydp = self._sym_dydt.jacobian(self._sym_deriv.as_explicit())
        #jacprotsens = (self._sym_dydt_jac * self._sym_sens.T.as_explicit()).as_explicit()
        #self._sym_rhs_sens = (jacprotsens + self._sym_dydp).as_explicit().T
        self._sym_dlamdadt = (-self._sym_lamda.as_explicit() * self._sym_dydt_jac).as_explicit()
        self._quad_rhs = (self._sym_lamda.as_explicit() * self._sym_dydp).as_explicit()

        self.user_data_dtype = np.dtype([
            ('fixed_params', (np.float64, (n_fixed,))),
            ('changeable_params', (np.float64, self.n_params)),
            ('tmp_nparams_nstates', np.float64, (self.n_params, self.n_states)),
            ('tmp2_nparams_nstates', np.float64, (self.n_params, self.n_states)),
        ])

    def make_user_data(self):
        user_data = np.recarray((), dtype=self.user_data_dtype)
        return user_data

    def update_changeable(self, user_data, params):
        user_data.changeable_params[:] = params.view(np.float64)

    def update_params(self, user_data, params):
        view_dtype = self.derivative_subset.subset_view_dtype
        dtype = self.derivative_subset.subset_dtype
        out = user_data.changeable_params.view(dtype)
        out.fill(params.view(view_dtype))

        view_dtype = self.remainder_subset.subset_view_dtype
        dtype = self.remainder_subset.subset_dtype
        out = user_data.fixed_params.view(dtype)
        out.fill(params.view(view_dtype))

    def update_derivative_params(self, user_data, params):
        user_data.changeable_params[:] = params

    def update_remaining_params(self, user_data, params):
        user_data.fixed_params[:] = params

    def extract_params(self, user_data, out=None):
        if out is None:
            out = np.full((1,), np.nan, dtype=self.params_dtype)[0]
        (
            out
            .view(self.derivative_subset.subset_view_dtype)
            .fill(
                user_data.changeable_params.view(self.derivative_subset.subset_dtype)[0]
            )
        )
        (
            out
            .view(self.remainder_subset.subset_view_dtype)
            .fill(
                user_data.fixed_params.view(self.remainder_subset.subset_dtype)[0]
            )
        )

        return out

    def extract_changeable(self, user_data, out=None):
        if out is None:
            out = np.empty(self.n_params)
        out[...] = user_data.changeable_params
        return out

    def make_rhs(self, *, debug=False):
        rhs_pre, rhs_calc = lambdify_consts(
            "_rhs",
            const_args=[],
            var_args=[
                self._sym_time,
                self._sym_statevec,
                self._sym_fixed,
                self._sym_deriv,
            ],
            expr=self._sym_dydt.T,
            debug=debug,
        )

        @numba.njit(inline='always')
        def rhs(out, t, y, user_data):
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params
            pre = rhs_pre()
            rhs_calc(
                out.reshape((1, -1)),
                pre,
                t,
                y.reshape((1, -1)),
                fixed.reshape((1, -1)),
                changeable.reshape((1, -1)),
            )
        return rhs

    def make_adjoint_rhs(self, *, debug=False):
        adj_pre, adj_calc = lambdify_consts(
            "_adj",
            const_args=[],
            var_args=[
                self._sym_time,
                self._sym_statevec,
                self._sym_lamda,
                self._sym_fixed,
                self._sym_deriv,
            ],
            expr=self._sym_dlamdadt,
            debug=debug,
        )

        @numba.njit(inline='always')
        def adjoint(out, t, y, lamda, user_data):
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params
            pre = adj_pre()
            adj_calc(
                out.reshape((1, -1)),
                pre,
                t,
                y.reshape((1, -1)),
                lamda.reshape((1, -1)),
                fixed.reshape((1, -1)),
                changeable.reshape((1, -1)),
            )

        return adjoint

    def make_adjoint_quad_rhs(self, *, debug=False):
        quad_pre, quad_calc = lambdify_consts(
            "_quad",
            const_args=[],
            var_args=[
                self._sym_time,
                self._sym_statevec,
                self._sym_lamda,
                self._sym_fixed,
                self._sym_deriv,
            ],
            expr=self._quad_rhs,
            debug=debug,
        )

        @numba.njit(inline='always')
        def quad_rhs(out, t, y, lamda, user_data):
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params
            pre = quad_pre()
            quad_calc(
                out.reshape((1, -1)),
                pre,
                t,
                y.reshape((1, -1)),
                lamda.reshape((1, -1)),
                fixed.reshape((1, -1)),
                changeable.reshape((1, -1)),
            )

        return quad_rhs

    def make_sensitivity_rhs(self, *, debug=False):
        sens_pre, sens_calc = lambdify_consts(
            "_sens",
            const_args=[],
            var_args=[
                self._sym_time,
                self._sym_statevec,
                self._sym_sens,
                self._sym_fixed,
                self._sym_deriv,
            ],
            expr=self._sym_rhs_sens,
            debug=debug,
        )

        n_params = self.n_params

        @numba.njit(inline='always')
        def wrapper(out, t, y, yS, user_data):
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params

            out_array = user_data.tmp_nparams_nstates
            yS_array = user_data.tmp2_nparams_nstates
            for i in range(n_params):
                yS_array[i, :] = yS[i]

            pre = sens_pre()
            sens_calc(
                out_array,
                pre,
                t,
                y.reshape((1, -1)),
                yS_array,
                fixed.reshape((1, -1)),
                changeable.reshape((1, -1)),
            )

            for i in range(n_params):
                out[i][:] = out_array[i, :]

            return 0

        return wrapper
