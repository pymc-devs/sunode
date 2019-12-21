import numpy as np
import sunode


ffi = sunode._cvodes.ffi
lib = sunode._cvodes.lib


def check(retcode):
    if isinstance(retcode, int) and retcode != 0:
        raise ValueError('Bad return code from sundials: %s' % retcode)
    if isinstance(retcode, ffi.CData):
        if retcode == ffi.NULL:
            raise ValueError('Return value of sundials is NULL.')
        return retcode


class Solver:
    def __init__(self, n_vars, rhs, tvals, user_data, *,
                 compute_sens=False, n_params=None, sens_rhs=None, abstol=1e-8, reltol=1e-7,
                 sens_mode=None, scaling_factors=None):
        self.n_vars = n_vars
        self.n_params = n_params
        self.tvals = tvals
        self._user_data = user_data.copy()
        self._state_buffer = sunode.empty_vector(self.n_vars)
        self._state_buffer.data[:] = 0
        self._jac = check(lib.SUNDenseMatrix(self.n_vars, self.n_vars))

        self._ode = check(lib.CVodeCreate(lib.CV_BDF))
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))
        
        self._set_tolerances(abstol, reltol)
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
                    raise RuntimeError("Bad sundials return code while solving ode: %s" % retval)
            y_out[i, :] = self._state_buffer.data
            
            if self._compute_sens:
                check(CVodeGetSens(ode, time_p, sens_buffer_array))
                for j in range(self.n_params):
                    sens_out[i, j, :] = sens_data[j]