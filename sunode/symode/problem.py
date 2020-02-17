from __future__ import annotations

from itertools import product

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

        self._sym_time = sym.Symbol('time', real=True)
        n_fixed = self.derivative_subset.n_items - self.n_params

        def make_vars(var_shapes, **kwargs):
            vars = {}
            for path, shape in var_shapes.items():
                name = '_'.join(path)
                var = sym.symarray(name, shape, **kwargs)
                vars[path] = var
            return vars

        self._sym_states = make_vars(self.state_subset.flat_shapes, positive=True)
        self._sym_params = make_vars(self.derivative_subset.flat_shapes, real=True)

        self._varmap = {}
        for path, vars in self._sym_states.items():
            for idxs in product(*[range(i) for i in vars.shape]):
                var = vars[idxs]
                if idxs == ():
                    self._varmap[var.name] = ('state', *path)
                else:
                    self._varmap[var.name] = ('state', *path, idxs)
        for path, vars in self._sym_params.items():
            for idxs in product(*[range(i) for i in vars.shape]):
                var = vars[idxs]
                if idxs == ():
                    self._varmap[var.name] = ('params', *path)
                else:
                    self._varmap[var.name] = ('params', *path, idxs)

        deriv_params = {
            k: v for k, v in self._sym_params.items()
            if k in self.derivative_subset.subset_paths
        }
        raveled_deriv = np.concatenate([var.ravel() for var in deriv_params.values()])
        fixed_params = {
            k: v for k, v in self._sym_params.items()
            if k not in self.derivative_subset.subset_paths
        }
        raveled_fixed = np.concatenate([var.ravel() for var in fixed_params.values()])

        def item_map(item):
            if hasattr(item, 'shape') and item.shape == ():
                return item.item()
            return item

        self._sym_deriv_paramsvec = raveled_deriv
        self._sym_params = self.derivative_subset.as_dataclass('Params', raveled_deriv, raveled_fixed, item_map=item_map)

        self._sym_statevec = np.concatenate([var.ravel() for var in self._sym_states.values()])
        self._sym_states = self.state_subset.as_dataclass('State', [], self._sym_statevec, item_map=item_map)

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

        dydt = sym.Matrix(rhs_list)
        self._sym_dydt = np.array(dydt).ravel()
        self._sym_sens = sym.symarray('sens', (self.n_params, self.n_states))
        self._sym_lamda = sym.symarray('lamda', self.n_states)

        for idxs in product(*[range(i) for i in self._sym_lamda.shape]):
            var = self._sym_lamda[idxs]
            self._varmap[var.name] = ('lamda', idxs)

        self._sym_dydt_jac = np.array(dydt.jacobian(self._sym_statevec))
        self._sym_dydp = np.array(dydt.jacobian(self._sym_deriv_paramsvec))
        #jacprotsens = (self._sym_dydt_jac * self._sym_sens.T.as_explicit()).as_explicit()
        #self._sym_rhs_sens = (jacprotsens + self._sym_dydp).as_explicit().T
        self._sym_dlamdadt = -self._sym_lamda @ self._sym_dydt_jac
        self._sym_quad_rhs = self._sym_lamda @ self._sym_dydp

        self.user_data_dtype = np.dtype([
            ('params', self.derivative_subset.dtype),
            ('tmp_nparams_nstates', np.float64, (self.n_params, self.n_states)),
            ('tmp2_nparams_nstates', np.float64, (self.n_params, self.n_states)),
            ('error_states', self.state_dtype),
            ('error_rhs', np.float64, (self.n_states,)),
            ('error_jac', np.float64, (self.n_states, self.n_states)),
        ])

    def make_user_data(self):
        user_data = np.recarray((), dtype=self.user_data_dtype)
        #user_data = np.zeros((), dtype=self.user_data_dtype)
        return user_data

    def update_params(self, user_data, params):
        user_data.params.fill(params)

    def update_derivative_params(self, user_data, params):
        view_dtype = self.derivative_subset.subset_view_dtype
        view = user_data.params.view(view_dtype)
        view.fill(params)

    def update_remaining_params(self, user_data, params):
        view_dtype = self.remainder_subset.subset_view_dtype
        view = user_data.params.view(view_dtype)
        view.fill(params)

    def extract_params(self, user_data, out=None):
        if out is None:
            out = np.full((1,), np.nan, dtype=self.params_dtype)[0]
        out.fill(user_data.params)
        return out

    def extract_changeable(self, user_data, out=None):
        if out is None:
            out = np.empty(self.n_params)
        out[...] = user_data.changeable_params
        return out

    def make_rhs(self, *, debug=False):
        rhs_calc = lambdify_consts(
            "_rhs",
            argnames=['time', 'state', 'params'],
            expr=np.array(self._sym_dydt.T),
            varmap=self._varmap,
            debug=debug,
        )

        @numba.njit(inline='always')
        def rhs(out, t, y, user_data):
            params = user_data.params
            rhs_calc(out, t, y, params)

            if (~np.isfinite(out)).any():
                user_data.error_rhs[:] = out
                user_data.error_states = y
                return 1
            return 0

        return rhs

    def make_adjoint_rhs(self, *, debug=False):
        adj_calc = lambdify_consts(
            "_adj",
            argnames=['time', 'state', 'lamda', 'params'],
            expr=self._sym_dlamdadt,
            varmap=self._varmap,
            debug=debug,
        )

        @numba.njit(inline='always')
        def adjoint(out, t, y, lamda, user_data):
            params = user_data.params
            adj_calc(out, t, y, lamda, params)

            if (~np.isfinite(out)).any():
                return 1
            return 0

        return adjoint

    def make_adjoint_quad_rhs(self, *, debug=False):
        quad_calc = lambdify_consts(
            "_quad",
            argnames=['time', 'state', 'lamda', 'params'],
            expr=self._sym_quad_rhs,
            varmap=self._varmap,
            debug=debug,
        )

        @numba.njit(inline='always')
        def quad_rhs(out, t, y, lamda, user_data):
            params = user_data.params
            quad_calc(out, t, y, lamda, params)

            if (~np.isfinite(out)).any():
                return 1
            return 0

        return quad_rhs

    def make_jac_dense(self, *, debug=False):
        jac_calc = lambdify_consts(
            "_jac_dense",
            argnames=['time', 'state', 'params'],
            expr=self._sym_dydt_jac,
            varmap=self._varmap,
            debug=debug,
        )

        @numba.njit(inline='always')
        def jac_dense(out, t, y, fy, user_data):
            params = user_data.params
            jac_calc(out, t, y, params)
            if (~np.isfinite(out)).any():
                user_data.error_jac[:] = out
                user_data.error_states = y
                return 1
            return 0

        return jac_dense

    def make_adjoint_jac_dense(self, *, debug=False):
        jac_calc = lambdify_consts(
            "_jac_dense",
            argnames=['time', 'state', 'params'],
            expr=-self._sym_dydt_jac.T,
            varmap=self._varmap,
            debug=debug,
        )

        @numba.njit(inline='always')
        def jac_dense(out, t, y, yB, fyB, user_data):
            params = user_data.params
            jac_calc(out, t, y, params)

            if (~np.isfinite(out)).any():
                return 1
            return 0

        return jac_dense

    def make_sensitivity_rhs1(self, *, debug=False):
        funcs = []

        for i in range(self.n_params):
            sens_calc = lambdify_consts(
                "_sens",
                argnames=['time', 'state', 'sens', 'params'],
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

            if (~np.isfinite(out)).any():
                return 1

            for i in range(n_params):
                out[i][:] = out_array[i, :]

            return 0

        return wrapper

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

            if (~np.isfinite(out)).any():
                return 1

            for i in range(n_params):
                out[i][:] = out_array[i, :]

            return 0

        return wrapper
