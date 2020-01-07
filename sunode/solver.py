import numpy as np
import sunode
import numba
import dataclasses

from sunode.symode.lambdify import lambdify_consts


ffi = sunode._cvodes.ffi
lib = sunode._cvodes.lib


ERROR_CODES = [name for name in dir(lib) if name.startswith('CV_')]
ERROR_CODES = {getattr(lib, name): name for name in ERROR_CODES}


def _as_dict(data):
    if data.dtype.fields is not None:
        return {name: _as_dict(data[name]) for name in data.dtype.fields}
    else:
        return data


def _from_dict(data, vals):
    if data.dtype.fields is not None:
        for name, (subtype, _) in data.dtype.fields.items():
            if subtype.fields is not None:
                _from_dict(data[name], vals[name])
            else:
                data[name] = vals[name]
    else:
        data[...] = vals


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
    def __init__(self, problem, *,
                 compute_sens=False, abstol=1e-12, reltol=1e-12,
                 sens_mode=None, scaling_factors=None, constraints=None):
        self._problem = problem
        self._user_data = problem.make_user_data()

        n_states = self._problem.n_states
        n_params = self._problem.n_params

        self._state_buffer = sunode.empty_vector(n_states)
        self._state_buffer.data[:] = 0
        self._jac = check(lib.SUNDenseMatrix(n_states, n_states))
        self._constraints = constraints

        self._ode = check(lib.CVodeCreate(lib.CV_BDF))
        rhs = problem.make_sundials_rhs()
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        self._set_tolerances(abstol, reltol)
        if self._constraints is not None:
            assert constraints.shape == (n_states,)
            self._constraints_vec = sunode.from_numpy(constraints)
            check(lib.CVodeSetConstraints(self._ode, self._constraints_vec.c_ptr))

        self._make_linsol()

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserData(self._ode, user_data_p))

        self._compute_sens = compute_sens
        if compute_sens:
            sens_rhs = self._problem.make_sundials_sensitivity_rhs()
            self._init_sens(sens_rhs, sens_mode)

    def _make_linsol(self):
        linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
        check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

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

        n_params = self._problem.n_params
        yS = check(lib.N_VCloneVectorArray(n_params, self._state_buffer.c_ptr))
        vecs = [sunode.basic.Vector(yS[i]) for i in range(n_params)]
        for vec in vecs:
            vec.data[:] = 0
        self._sens_buffer_array = yS
        self._sens_buffers = vecs

        check(lib.CVodeSensInit(self._ode, n_params, sens_mode, sens_rhs.cffi, yS))

        if scaling_factors is not None:
            if scaling_factors.shape != (n_params,):
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
        n_states = self._problem.n_states
        n_params = self._problem.n_params
        y_vals = np.zeros((len(tvals), n_states))
        if self._compute_sens:
            sens_vals = np.zeros((len(tvals), n_params, n_states))
            return y_vals, sens_vals
        return y_vals

    def as_xarray(self, tvals, out, sens_out=None, unstack_state=True, unstack_params=True):
        return self._problem.solution_to_xarray(
            tvals, out, self._user_data,
            sensitivity=sens_out,
            unstack_state=unstack_state, unstack_params=unstack_params
        )

    @property
    def params_dtype(self):
        return self._problem.params_dtype

    def set_params(self, params):
        self._problem.update_params(self._user_data, params)

    def get_params(self):
        return self._problem.extract_params(self._user_data)

    def set_derivative_params(self, params):
        self._problem.update_derivative_params(self._user_data, params)

    def set_remaining_params(self, params):
        self._problem.update_remaining_params(self._user_data, params)

    def set_params_dict(self, params):
        data = self.get_params()
        _from_dict(data, params)
        self.set_params(data)

    def get_params_dict(self):
        return _as_dict(self.get_params())

    def set_params_array(self, params):
        self._problem.update_changeable(self._user_data, params)

    def get_params_array(self, out=None):
        return self._problem.extract_changeable(self._user_data, out=out)

    def solve(self, t0, tvals, y0, y_out, *, sens0=None, sens_out=None):
        if self._compute_sens and (sens0 is None or sens_out is None):
            raise ValueError('"sens_out" and "sens0" are required when computin sensitivities.')
        CVodeReInit = lib.CVodeReInit
        CVodeSensReInit = lib.CVodeSensReInit
        CVode = lib.CVode
        CVodeGetSens = lib.CVodeGetSens
        ode = self._ode
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        n_params = self._problem.n_params

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        if self._compute_sens:
            sens_buffer_array = self._sens_buffer_array
            sens_data = tuple(buffer.data for buffer in self._sens_buffers)
            for i in range(n_params):
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
            y_out[i, :] = state_data

            if self._compute_sens:
                check(CVodeGetSens(ode, time_p, sens_buffer_array))
                for j in range(n_params):
                    sens_out[i, j, :] = sens_data[j]


class AdjointSolver:
    def __init__(self, problem, *,
                 abstol=1e-12, reltol=1e-12,
                 checkpoint_n=500, interpolation='polynomial', constraints=None):
        self._problem = problem

        n_states, n_params = problem.n_states, problem.n_params
        self._user_data = problem.make_user_data()

        self._state_buffer = sunode.empty_vector(n_states)
        self._state_buffer.data[:] = 0

        self._jac = check(lib.SUNDenseMatrix(n_states, n_states))
        self._jacB = check(lib.SUNDenseMatrix(n_states, n_states))

        rhs = problem.make_sundials_rhs()
        self._adj_rhs = problem.make_sundials_adjoint_rhs()
        self._quad_rhs = problem.make_sundials_adjoint_quad_rhs()
        self._rhs = problem.make_rhs()
        self._constraints = constraints

        self._ode = check(lib.CVodeCreate(lib.CV_BDF))
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        self._set_tolerances(abstol, reltol)
        if self._constraints is not None:
            assert constraints.shape == (n_states,)
            self._constraints_vec = sunode.from_numpy(constraints)
            check(lib.CVodeSetConstraints(self._ode, self._constraints_vec.c_ptr))

        self._make_linsol()

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserData(self._ode, user_data_p))

        if interpolation == 'polynomial':
            interpolation = lib.CV_POLYNOMIAL
        elif interpolation == 'hermite':
            interpolation = lib.CV_HERMITE
        else:
            assert False
        self._init_backward(checkpoint_n, interpolation)

    def _init_backward(self, checkpoint_n, interpolation):
        check(lib.CVodeAdjInit(self._ode, checkpoint_n, interpolation))

        # Initialized by CVodeCreateB
        backward_ode = ffi.new('int*')
        check(lib.CVodeCreateB(self._ode, lib.CV_BDF, backward_ode))
        self._odeB = backward_ode[0]

        self._state_bufferB = sunode.empty_vector(self._problem.n_states)
        check(lib.CVodeInitB(self._ode, self._odeB, self._adj_rhs.cffi, 0., self._state_bufferB.c_ptr))

        # TODO
        check(lib.CVodeSStolerancesB(self._ode, self._odeB, 1e-12, 1e-12))

        linsolver = check(lib.SUNLinSol_Dense(self._state_bufferB.c_ptr, self._jacB))
        check(lib.CVodeSetLinearSolverB(self._ode, self._odeB, linsolver, self._jacB))

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserDataB(self._ode, self._odeB, user_data_p))

        self._quad_buffer = sunode.empty_vector(self._problem.n_params)
        self._quad_buffer_out = sunode.empty_vector(self._problem.n_params)
        check(lib.CVodeQuadInitB(self._ode, self._odeB, self._quad_rhs.cffi, self._quad_buffer.c_ptr))

        check(lib.CVodeQuadSStolerancesB(self._ode, self._odeB, 1e-12, 1e-12))
        check(lib.CVodeSetQuadErrConB(self._ode, self._odeB, 0))

    def _make_linsol(self):
        linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
        check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

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
        y_vals = np.zeros((len(tvals), self._problem.n_states))
        grad_out = np.zeros(self._problem.n_params)
        lamda_out = np.zeros(self._problem.n_states)
        return y_vals, grad_out, lamda_out

    def as_xarray(self, tvals, out, sens_out=None, unstack_state=True, unstack_params=True):
        return self._problem.solution_to_xarray(
            tvals, out, self._user_data,
            sensitivity=sens_out,
            unstack_state=unstack_state, unstack_params=unstack_params
        )

    @property
    def params_dtype(self):
        return self._problem.params_dtype

    def set_params(self, params):
        self._problem.update_params(self._user_data, params)

    def get_params(self):
        return self._problem.extract_params(self._user_data)

    def set_params_dict(self, params):
        data = self.get_params()
        _from_dict(data, params)
        self.set_params(data)

    def get_params_dict(self):
        return _as_dict(self.get_params())

    def set_derivative_params(self, params):
        self._problem.update_derivative_params(self._user_data, params)

    def set_remaining_params(self, params):
        self._problem.update_remaining_params(self._user_data, params)

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
            y_out[i, :] = state_data

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
            retval = CVodeB(ode, tend, lib.CV_NORMAL)
            if retval != TOO_MUCH_WORK and retval != 0:
                raise SolverError("Bad sundials return code while solving ode: %s (%s)"
                                  % (ERROR_CODES[retval], retval))

        check(CVodeGetB(ode, odeB, time_p, state_c_ptr))
        check(CVodeGetQuadB(ode, odeB, time_p, quad_out_c_ptr))
        grad_out[:] = quad_out_data
        lamda_out[:] = state_data


