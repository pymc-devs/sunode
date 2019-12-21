import numpy as np
import sunode
import sympy as sym
import numba
import dataclasses
import theano
import theano.tensor as tt

from nitrogene.common.better_lambdify import lambdify_consts


ffi = sunode._cvodes.ffi
lib = sunode._cvodes.lib


ERROR_CODES = [name for name in dir(lib) if name.startswith('CV_')]
ERROR_CODES = {getattr(lib, name): name for name in ERROR_CODES}


class SolverError(RuntimeError):
    pass


def check(retcode):
    if isinstance(retcode, int) and retcode != 0:
        raise ValueError('Bad return code from sundials: %s (%s)' % (ERROR_CODES[retcode], retcode))
    if isinstance(retcode, ffi.CData):
        if retcode == ffi.NULL:
            raise ValueError('Return value of sundials is NULL.')
        return retcode


class Solver:
    def __init__(self, n_vars, rhs, tvals, user_data, *,
                 compute_sens=False, n_params=None, sens_rhs=None, abstol=1e-8, reltol=1e-7,
                 sens_mode=None, scaling_factors=None, constraints=None):
        self.n_vars = n_vars
        self.n_params = n_params
        self.tvals = tvals
        self._user_data = user_data.copy()
        self._state_buffer = sunode.empty_vector(self.n_vars)
        self._state_buffer.data[:] = 0
        self._jac = check(lib.SUNDenseMatrix(self.n_vars, self.n_vars))
        self._constraints = constraints

        self._ode = check(lib.CVodeCreate(lib.CV_BDF))
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        self._set_tolerances(abstol, reltol)
        if self._constraints is not None:
            assert constraints.shape == (n_vars,)
            self._constraints_vec = sunode.from_numpy(constraints)
            check(lib.CVodeSetConstraints(self._ode, self._constraints_vec.c_ptr))

        self._make_linsol()
        self._set_user_data(self._user_data)
        self._compute_sens = compute_sens
        if compute_sens:
            if sens_rhs is None or n_params is None:
                raise ValueError('Missing sens_rhs or n_params.')
            self._init_sens(sens_rhs, sens_mode)

    @property
    def user_data(self):
        return self._user_data

    def _make_linsol(self):
        linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
        check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

    def _set_user_data(self, user_data):
        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(user_data.data)))
        check(lib.CVodeSetUserData(self._ode, user_data_p))

    def _init_sens(self, sens_rhs, sens_mode, scaling_factors=None):
        if sens_mode == 'simultaneous':
            sens_mode = lib.CV_SIMULTANEOUS
        elif sens_mode == 'staggered':
            sens_mode = lib.CV_STAGGERED
        elif sens_mode == 'staggered1':
            raise ValueError('staggered1 requires work.')
        else:
            raise ValueError('sens_mode must be one of "simultaneous" and "staggered".')

        self._sens_mode = sens_mode

        yS = check(lib.N_VCloneVectorArray(self.n_params, self._state_buffer.c_ptr))
        vecs = [sunode.basic.Vector(yS[i]) for i in range(self.n_params)]
        for vec in vecs:
            vec.data[:] = 0
        self._sens_buffer_array = yS
        self._sens_buffers = vecs

        check(lib.CVodeSensInit(self._ode, self.n_params, sens_mode, sens_rhs.cffi, yS))

        if scaling_factors is not None:
            if scaling_factors.shape != (self.n_params,):
                raise ValueError('Invalid shape of scaling_factors.')
            self._scaling_factors = scaling_factors
            NULL_D = ffi.cast('double *', 0)
            NULL_I = ffi.cast('int *', 0)
            pbar_p = ffi.cast('double *', ffi.addressof(ffi.from_buffer(scaling_factors.data)))
            check(lib.CVodeSetSensParams(ode, NULL_D, pbar_p, NULL_I))

        check(lib.CVodeSensEEtolerances(self._ode))  # TODO
        check(lib.CVodeSetSensErrCon(self._ode, 1))  # TODO

    def _set_tolerances(self, atol=None, rtol=None):
        atol = np.array(atol)
        rtol = np.array(rtol)
        if atol.ndim == 1 and rtol.ndim == 1:
            atol = sunode.from_numpy(atol)
            rtol = sunode.from_numpy(rtol)
            check(lib.CVodeVVtolerances(self._ode, rtol.c_ptr, atol.c_ptr))
        elif atol.ndim == 1 and rtol.ndim == 0:
            atol = sunode.from_numpy(atol)
            check(lib.CVodeSVtolerances(self._ode, rtol, atol.c_ptr))
        elif atol.ndim == 0 and rtol.ndim == 1:
            rtol = sunode.from_numpy(rtol)
            check(lib.CVodeVStolerances(self._ode, rtol.c_ptr, atol))
        elif atol.ndim == 0 and rtol.ndim == 0:
            check(lib.CVodeSStolerances(self._ode, rtol, atol))
        else:
            raise ValueError('Invalid tolerance.')
        self._atol = atol
        self._rtol = rtol

    def make_output_buffers(self, tvals):
        y_vals = np.zeros((len(tvals), self.n_vars))
        if self._compute_sens:
            sens_vals = np.zeros((len(tvals), self.n_params, self.n_vars))
            return y_vals, sens_vals
        return y_vals

    def solve(self, t0, tvals, y0, y_out, *, sens0=None, sens_out=None):
        if self._compute_sens and (sens0 is None or sens_out is None):
            raise ValueError('"sens_out" and "sens0" are required when computin sensitivities.')
        CVodeReInit = lib.CVodeReInit
        CVodeSensReInit = lib.CVodeSensReInit
        CVode = lib.CVode
        CVodeGetSens = lib.CVodeGetSens
        ode = self._ode
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        if self._compute_sens:
            sens_buffer_array = self._sens_buffer_array
            sens_data = tuple(buffer.data for buffer in self._sens_buffers)
            for i in range(self.n_params):
                sens_data[i][:] = sens0[i, :]

        state_data[:] = y0

        time_p = ffi.new('double*')
        time_p[0] = t0

        check(CVodeReInit(ode, t0, state_c_ptr))
        if self._compute_sens:
            check(CVodeSensReInit(ode, self._sens_mode, self._sens_buffer_array))

        for i, t in enumerate(tvals):
            if t == t0:
                y_out[0, :] = y0
                if self._compute_sens:
                    sens_out[0, :, :] = sens0
                continue

            retval = TOO_MUCH_WORK
            while retval == TOO_MUCH_WORK:
                retval = CVode(ode, t, state_c_ptr, time_p, lib.CV_NORMAL)
                if retval != TOO_MUCH_WORK and retval != 0:
                    raise SolverError("Bad sundials return code while solving ode: %s (%s)"
                                      % (ERROR_CODES[retval], retval))
            y_out[i, :] = self._state_buffer.data

            if self._compute_sens:
                check(CVodeGetSens(ode, time_p, sens_buffer_array))
                for j in range(self.n_params):
                    sens_out[i, j, :] = sens_data[j]


class AdjointSolver:
    def __init__(self, n_vars, rhs, adj_rhs, quad_rhs, tvals, user_data, *,
                 compute_sens=False, n_params=None, sens_rhs=None, abstol=1e-12, reltol=1e-12,
                 sens_mode=None, scaling_factors=None, checkpoint_n=200, interpolation='polynomial',
                 constraints=None):
        self.n_vars = n_vars
        self.n_params = n_params
        self.tvals = tvals
        self._user_data = user_data.copy()
        self._state_buffer = sunode.empty_vector(self.n_vars)
        self._state_buffer.data[:] = 0
        self._jac = check(lib.SUNDenseMatrix(self.n_vars, self.n_vars))
        self._jacB = check(lib.SUNDenseMatrix(self.n_vars, self.n_vars))
        self._adj_rhs = adj_rhs
        self._quad_rhs = quad_rhs
        self._rhs = rhs
        self._constraints = constraints

        self._ode = check(lib.CVodeCreate(lib.CV_BDF))
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        self._set_tolerances(abstol, reltol)
        if self._constraints is not None:
            assert constraints.shape == (n_vars,)
            self._constraints_vec = sunode.from_numpy(constraints)
            check(lib.CVodeSetConstraints(self._ode, self._constraints_vec.c_ptr))
        self._make_linsol()
        self._set_user_data(self._user_data)

        if interpolation == 'polynomial':
            interpolation = lib.CV_POLYNOMIAL
        elif interpolation == 'hermite':
            interpolation = lib.CV_HERMITE
        else:
            assert False
        self._init_backward(checkpoint_n, interpolation)

    @property
    def user_data(self):
        return self._user_data

    def _init_backward(self, checkpoint_n, interpolation):
        check(lib.CVodeAdjInit(self._ode, checkpoint_n, interpolation))

        # Initialized by CVodeCreateB
        backward_ode = ffi.new('int*')
        check(lib.CVodeCreateB(self._ode, lib.CV_BDF, backward_ode))
        self._odeB = backward_ode[0]

        self._state_bufferB = sunode.empty_vector(self.n_vars)
        check(lib.CVodeInitB(self._ode, self._odeB, self._adj_rhs.cffi, 0., self._state_bufferB.c_ptr))

        # TODO
        check(lib.CVodeSStolerancesB(self._ode, self._odeB, 1e-12, 1e-12))

        linsolver = check(lib.SUNLinSol_Dense(self._state_bufferB.c_ptr, self._jacB))
        check(lib.CVodeSetLinearSolverB(self._ode, self._odeB, linsolver, self._jacB))

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserDataB(self._ode, self._odeB, user_data_p))

        self._quad_buffer = sunode.from_numpy(np.zeros(self.n_params))
        self._quad_buffer_out = sunode.from_numpy(np.zeros(self.n_params))
        check(lib.CVodeQuadInitB(self._ode, self._odeB, self._quad_rhs.cffi, self._quad_buffer.c_ptr))

        check(lib.CVodeQuadSStolerancesB(self._ode, self._odeB, 1e-12, 1e-12))
        check(lib.CVodeSetQuadErrConB(self._ode, self._odeB, 1))

    def _make_linsol(self):
        linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
        check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

    def _set_user_data(self, user_data):
        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(user_data.data)))
        check(lib.CVodeSetUserData(self._ode, user_data_p))

    def _set_tolerances(self, atol=None, rtol=None):
        atol = np.array(atol)
        rtol = np.array(rtol)
        if atol.ndim == 1 and rtol.ndim == 0:
            atol = sunode.from_numpy(atol)
            check(lib.CVodeSVtolerances(self._ode, rtol, atol.c_ptr))
        elif atol.ndim == 0 and rtol.ndim == 0:
            check(lib.CVodeSStolerances(self._ode, rtol, atol))
        else:
            raise ValueError('Invalid tolerance.')
        self._atol = atol
        self._rtol = rtol

    def make_output_buffers(self, tvals):
        y_vals = np.zeros((len(tvals), self.n_vars))
        grad_out = np.zeros(self.n_params)
        lamda_out = np.zeros(self.n_vars)
        return y_vals, grad_out, lamda_out

    def solve_forward(self, t0, tvals, y0, y_out):
        CVodeReInit = lib.CVodeReInit
        CVodeAdjReInit = lib.CVodeAdjReInit
        CVodeF = lib.CVodeF
        ode = self._ode
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        state_data[:] = y0

        time_p = ffi.new('double*')
        time_p[0] = t0

        n_check = ffi.new('int*')
        n_check[0] = 0

        check(CVodeReInit(ode, t0, state_c_ptr))
        check(CVodeAdjReInit(ode))

        for i, t in enumerate(tvals):
            if t == t0:
                y_out[0, :] = y0
                continue

            retval = TOO_MUCH_WORK
            while retval == TOO_MUCH_WORK:
                retval = CVodeF(ode, t, state_c_ptr, time_p, lib.CV_NORMAL, n_check)
                if retval != TOO_MUCH_WORK and retval != 0:
                    raise SolverError("Bad sundials return code while solving ode: %s (%s)"
                                      % (ERROR_CODES[retval], retval))
            y_out[i, :] = self._state_buffer.data

    def solve_backward(self, t0, tend, tvals, grads, grad_out, lamda_out):
        CVodeReInitB = lib.CVodeReInitB
        CVodeQuadReInitB = lib.CVodeQuadReInitB
        CVodeGetQuadB = lib.CVodeGetQuadB
        CVodeB = lib.CVodeB
        CVodeGetB = lib.CVodeGetB
        ode = self._ode
        odeB = self._odeB
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        state_data = self._state_bufferB.data
        state_c_ptr = self._state_bufferB.c_ptr

        quad_data = self._quad_buffer.data
        quad_c_ptr = self._quad_buffer.c_ptr

        quad_out_data = self._quad_buffer_out.data
        quad_out_c_ptr = self._quad_buffer_out.c_ptr

        state_data[:] = -grads[-1]
        quad_data[:] = 0

        time_p = ffi.new('double*')
        time_p[0] = t0

        check(CVodeReInitB(ode, odeB, t0, state_c_ptr))
        check(CVodeQuadReInitB(ode, odeB, quad_c_ptr))

        for i, (t, grad) in enumerate(zip(reversed(tvals), reversed(grads))):
            if i == 0:
                continue

            retval = TOO_MUCH_WORK
            while retval == TOO_MUCH_WORK:
                retval = CVodeB(ode, t, lib.CV_NORMAL)
                if retval != TOO_MUCH_WORK and retval != 0:
                    raise SolverError("Bad sundials return code while solving ode: %s (%s)"
                                      % (ERROR_CODES[retval], retval))

            check(CVodeGetB(ode, odeB, time_p, state_c_ptr))
            check(CVodeGetQuadB(ode, odeB, time_p, quad_out_c_ptr))
            quad_data[:] = quad_out_data[:]
            assert time_p[0] == t, (time_p[0], t)
            state_data[:] -= grad
            check(CVodeReInitB(ode, odeB, t, state_c_ptr))
            check(CVodeQuadReInitB(ode, odeB, quad_c_ptr))


        retval = TOO_MUCH_WORK
        while t > tend and retval == TOO_MUCH_WORK:
            assert False
            retval = CVodeB(ode, tend, lib.CV_NORMAL)
            if retval != TOO_MUCH_WORK and retval != 0:
                raise SolverError("Bad sundials return code while solving ode: %s (%s)"
                                  % (ERROR_CODES[retval], retval))

        check(CVodeGetB(ode, odeB, time_p, state_c_ptr))
        check(CVodeGetQuadB(ode, odeB, time_p, quad_out_c_ptr))
        grad_out[:] = quad_out_data
        lamda_out[:] = state_data


class SympyOde:
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

    def xarray_solution(self, tvals, solution, user_data, sensitivity=None,
                        *, unstack_state=True, unstack_params=True):
        import xarray as xr

        assert sensitivity is None, 'TODO'
        solution = solution.view(self._state_dtype)[..., 0]
        self.paramset.set_fixed(user_data.fixed_params)
        self.paramset.set_changeable(user_data.changeable_params)
        params = self.paramset.record

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

    def make_sundials_rhs(self):
        rhs = self.make_rhs()

        N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
        N_VGetLength_Serial = lib.N_VGetLength_Serial

        user_dtype = self._user_dtype
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
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params

            rhs(out, t, y, fixed, changeable)
            return 0

        return rhs_wrapper

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

    def make_sundials_adjoint(self):
        user_dtype = self._user_dtype
        adj = self.make_adjoint()

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
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params

            adj(
                yBdot,
                t,
                y,
                yB,
                fixed,
                changeable,
            )
            return 0

        return adj_rhs_wrapper

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

    def make_sundials_adjoint_quad(self):
        user_dtype = self._user_dtype
        adjoint_quad = self.make_adjoint_quad()

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
            fixed = user_data.fixed_params
            changeable = user_data.changeable_params

            adjoint_quad(
                yBdot,
                t,
                y,
                yB,
                fixed,
                changeable,
            )
            return 0

        return quad_rhs_wrapper


class SolveODEAdjoint(tt.Op):
    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)

    def perform(self, node, inputs, outputs):
        y0, params = inputs

        self._solver.user_data.changeable_params[:] = params

        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
        except SolverError:
            self._y_out[:] = np.nan

        outputs[0][0] = self._y_out.copy()

    def grad(self, inputs, g):
        g, = g

        y0, params = inputs
        backward = SolveODEAdjointBackward(self._solver, self._t0, self._tvals)
        lamda, gradient = backward(y0, params, g)
        if self._t0 == self._tvals[0]:
            lamda = lamda - g[0]
        #return [-lamda + g[0], gradient]
        return [-lamda, gradient]


class SolveODEAdjointBackward(tt.Op):
    itypes = [tt.dvector, tt.dvector, tt.dmatrix]
    otypes = [tt.dvector, tt.dvector]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)

    def perform(self, node, inputs, outputs):
        y0, params, grads = inputs

        self._solver.user_data.changeable_params[:] = params

        # TODO We don't really need to do the forward solve if we make sure
        # that it was executed previously, but it isn't very expensive
        # compared with the backward pass anyway.
        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
            self._solver.solve_backward(self._tvals[-1], self._t0, self._tvals,
                                        grads, self._grad_out, self._lamda_out)
        except SolverError:
            self._lamda_out[:] = np.nan
            self._grad_out[:] = np.nan

        outputs[0][0] = self._lamda_out.copy()
        outputs[1][0] = self._grad_out.copy()
