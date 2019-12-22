#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SympyOde(Ode):
    def __init__(self, paramset, states, rhs_sympy):
        self.paramset = paramset
        self.states = states
        self.rhs_sympy = rhs_sympy

        self.n_params = len(paramset.changeable_array())
        self.n_states = sum(np.prod(shape, dtype=int) for _, shape in self.states.items())

        self._sym_time = sym.Symbol('t', real=True)
        self._sym_params, self._sym_changeable, self._sym_fixed = paramset.as_sympy('p')

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
        self.state_vars = list(states.keys())

        rhs = self.rhs_sympy(self._sym_time, self._sym_states, self._sym_params)
        rhs_list = []
        for path in self.state_vars:
            assert path in rhs
            rhs_list.append(rhs[path])
        self._sym_dydt = sym.Matrix(rhs_list)
        self._sym_sens = sym.MatrixSymbol('sens', self.n_params, self.n_states)
        self._sym_lamda = sym.MatrixSymbol('lamda', 1, self.n_states)

        self._sym_dydt_jac = self._sym_dydt.jacobian(self._sym_statevec)
        self._sym_dydp = self._sym_dydt.jacobian(self._sym_changeable)
        jacprotsens = (self._sym_dydt_jac * self._sym_sens.T.as_explicit()).as_explicit()
        #self._sym_rhs_sens = (jacprotsens + self._sym_dydp).as_explicit()
        self._sym_dlamdadt = (-self._sym_lamda.as_explicit() * self._sym_dydt_jac).as_explicit()
        self._quad_rhs = (self._sym_lamda.as_explicit() * self._sym_dydp).as_explicit()

        self._user_dtype = np.dtype([
            ('fixed_params', (np.float64, len(paramset.fixed_array())),),
            ('changeable_params', (np.float64, len(paramset.changeable_array())),),
        ])

        self.user_data = np.recarray((), dtype=self._user_dtype)
        self.user_data.fixed_params[:] = paramset.fixed_array()
        self.user_data.changeable_params[:] = paramset.changeable_array()

        self._state_dtype = np.dtype([
            (name, np.float64, shape) for name, shape in states.items()
        ])

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
        def rhs(out, t, y, fixed, changeable):
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

    def make_adjoint(self, *, debug=False):
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
        def adjoint(out, t, y, lamda, fixed, changeable):
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

    def make_adjoint_quad(self, *, debug=False):
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
        def quad_rhs(out, t, y, lamda, fixed, changeable):
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
