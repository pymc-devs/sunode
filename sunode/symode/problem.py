from __future__ import annotations

import numpy as np
import sympy as sym
import dataclasses
import numba

from sunode import problem
from sunode.symode.lambdify import lambdify_consts


class SympyOde(problem.Ode):
    def __init__(self, paramset, states, rhs_sympy):
        self._paramset = paramset
        self._states = states
        self.state_type = np.dtype([
            (name, np.float64, shape) for name, shape in states.items()
        ])
        self._rhs_sympy_func = rhs_sympy
        self.all_param_type = paramset.record.dtype

        self.n_params = len(paramset.changeable_array())
        self.params_type = np.dtype([('params', np.float64, (self.n_params,))])

        self._sym_time = sym.Symbol('t', real=True)
        self._sym_params, self._sym_fixed, self._sym_changeable = paramset.as_sympy('p')

        self._sym_statevec = sym.MatrixSymbol('y', 1, self.n_states)
        self._sym_states = dataclasses.make_dataclass('State', list(states.keys()))
        count = 0
        for name, shape in states.items():
            length = np.prod(shape, dtype=int)
            var = list(self._sym_statevec[0, count:count + length])
            if len(shape) > 0:
                var = sym.Array(var, shape=shape)
            else:
                var = var[0]
            setattr(self._sym_states, name, var)
            count += length
        self._state_vars = list(states.keys())

        rhs = self._rhs_sympy_func(self._sym_time, self._sym_states, self._sym_params)
        rhs_list = []
        for path in self.state_type.names:
            assert path in rhs
            rhs_list.append(rhs[path])

        self._sym_dydt = sym.Matrix(rhs_list)
        self._sym_sens = sym.MatrixSymbol('sens', self.n_params, self.n_states)
        self._sym_lamda = sym.MatrixSymbol('lamda', 1, self.n_states)

        self._sym_dydt_jac = self._sym_dydt.jacobian(self._sym_statevec)
        self._sym_dydp = self._sym_dydt.jacobian(self._sym_changeable)
        jacprotsens = (self._sym_dydt_jac * self._sym_sens.T.as_explicit()).as_explicit()
        self._sym_rhs_sens = (jacprotsens + self._sym_dydp).as_explicit().T
        self._sym_dlamdadt = (-self._sym_lamda.as_explicit() * self._sym_dydt_jac).as_explicit()
        self._quad_rhs = (self._sym_lamda.as_explicit() * self._sym_dydp).as_explicit()

        self.user_data_type = np.dtype([
            ('fixed_params', (np.float64, len(paramset.fixed_array()))),
            ('changeable_params', (np.float64, self.n_params)),
            ('tmp_nparams_nstates', np.float64, (self.n_params, self.n_states)),
            ('tmp2_nparams_nstates', np.float64, (self.n_params, self.n_states)),
        ])

    def make_user_data(self):
        user_data = np.recarray((), dtype=self.user_data_type)
        user_data.fixed_params[:] = self._paramset.fixed_array()
        user_data.changeable_params[:] = self._paramset.changeable_array()
        return user_data

    def update_changeable(self, user_data, params):
        user_data.changeable_params[:] = params

    def extract_params(self, user_data, out=None):
        self._paramset.set_fixed(user_data.fixed_params)
        self._paramset.set_changeable(user_data.changeable_params)
        params = self._paramset.record
        if out is not None:
            out[...] = params
            return out
        return params

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
                self._sym_changeable
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
                self._sym_changeable,
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
                self._sym_changeable
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
                self._sym_changeable
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
