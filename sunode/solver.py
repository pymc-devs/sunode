from typing import overload, Union, Optional, Callable, Dict, Any
import logging
import weakref

import numpy as np
import xarray as xr

import sunode
from sunode.basic import CPointer, ERRORS, lib, ffi, check, check_ptr, Borrows, check_code
from sunode.dtypesubset import _as_dict
from sunode.matrix import Matrix
from sunode.linear_solver_wrapper import LinearSolver
from sunode.nonlinear_solver import NonlinearSolver
from sunode.problem import Problem
from sunode import matrix, vector


logger = logging.getLogger('sunode.solver')


class SolverError(RuntimeError):
    pass


class BaseSolver(Borrows):
    problem: Problem
    user_data: np.ndarray
    _input_changed: bool

    def __init__(self, problem: Problem, *, solver: str = 'BDF', jac_kind: str = "dense"):
        super().__init__()

        self.mark_changed()

        self.problem = problem
        self.user_data = problem.make_user_data()

        self._state_buffer = sunode.empty_vector(self.n_states)
        self._state_buffer.data[:] = 0.

        self.borrow(self._state_buffer)

        if jac_kind == 'dense':
            self._jac = matrix.empty_matrix((self.n_states, self.n_states))
        elif jac_kind == 'sparse':
            self._jac = problem.make_rhs_sparse_jac_template()
        else:
            raise ValueError(f'Unknown jac_kind {jac_kind}.')

        self.borrow(self._jac)

        if solver == 'BDF':
            self.c_ptr = check_ptr(lib.CVodeCreate(lib.CV_BDF))
        elif solver == 'ADAMS':
            self.c_ptr = check_ptr(lib.CVodeCreate(lib.CV_ADAMS))
        else:
            raise ValueError(f'Unknown solver {solver}.')

        self._rhs = self.problem.make_sundials_rhs()

        def finalize(c_ptr: CPointer, release_borrowed: Callable[[], None]) -> None:
            if c_ptr == ffi.NULL:
                logger.warn("Trying to free Solver, but it is NULL.")
            else:
                logger.debug("Freeing Solver")
                lib.CVodeFree(c_ptr)
            release_borrowed()
        weakref.finalize(self, finalize, self.c_ptr, self.release_borrowed_func())

    def init(self, t0: float, state: Optional[np.ndarray] = None, recreate_rhs: bool = False) -> None:
        if state is not None:
            self.state[:] = state
        if recreate_rhs:
            self._rhs = self.problem.make_sundials_rhs()
        check(lib.CVodeInit(self.c_ptr, self._rhs.cffi, t0, self._state_buffer.c_ptr))

    def mark_changed(self, set_to: bool = True) -> None:
        self._input_changed = set_to

    @property
    def is_changed(self) -> bool:
        return self._input_changed

    def tolerance(self, rtol: float, atol: Union[np.ndarray, float]) -> None:
        self.mark_changed()

        self._atol = np.array(atol)
        self._rtol = rtol

        if self._atol.ndim == 1:
            if not hasattr(self, '_atol_buffer'):
                self._atol_buffer = sunode.from_numpy(atol)
                self.borrow(self._atol_buffer)
            self._atol_buffer.data[:] = atol
            check(lib.CVodeSVtolerances(self.c_ptr, self._rtol, self._atol_buffer.c_ptr))
        elif self._atol.ndim == 0:
            check(lib.CVodeSStolerances(self.c_ptr, self._rtol, self._atol))
        else:
            raise ValueError('Invalid absolute tolerances.')

    def constraints(self, constraints: Optional[np.ndarray]) -> None:
        self.mark_changed()

        if constraints is None:
            check(lib.CVodeSetConstraints(self.c_ptr, ffi.NULL))
            return

        assert constraints.shape == (self.n_states,)
        if not hasattr(self, '_constraints_buffer'):
            self._constraints_buffer = sunode.from_numpy(constraints)
            self.borrow(self._constraints_buffer)
        self._constraints_buffer.data[:] = constraints
        check(lib.CVodeSetConstraints(self.c_ptr, self._constraints_buffer.c_ptr))

    def make_output_buffers(self, tvals: np.ndarray) -> np.ndarray:
        n_states = self.problem.n_states
        n_params = self.problem.n_params
        y_vals = np.zeros((len(tvals), n_states))
        return y_vals

    def as_xarray(
        self,
        tvals: np.ndarray,
        out: np.ndarray,
        unstack_state: bool = True,
        unstack_params: bool = True
    ) -> xr.Dataset:
        return self.problem.solution_to_xarray(
            tvals, out, self._user_data,
            unstack_state=unstack_state, unstack_params=unstack_params
        )

    def solve(
        self,
        t0: float,
        tvals: np.ndarray,
        y0: np.ndarray,
        y_out: np.ndarray,
        forward_sens: Optional[np.ndarray] = None,
        checkpointing: bool = False
    ) -> None:
        CVodeReInit = lib.CVodeReInit
        CVodeAdjReInit = lib.CVodeAdjReInit
        CVodeF = lib.CVodeF
        ode = self.c_ptr
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        if y0.dtype == self._problem.state_dtype:
            y0 = y0[None].view(np.float64)
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
                                      % (ERRORS[retval], retval))
            y_out[i, :] = state_data

        self.mark_changed(False)

    def linear_solver(self, solver: LinearSolver, jac_template: Matrix) -> None:
        self.mark_changed()
        self._linear_solver = solver
        self.borrow(solver)
        check(lib.CVodeSetLinearSolver(self.c_ptr, solver.c_ptr, jac_template.c_ptr))

    def nonlinear_solver(self, solver: NonlinearSolver) -> None:
        self.mark_changed()
        self._nonlinear_solver = solver
        self.borrow(solver)
        check(lib.CVodeSetNonlinearSolver(self.c_ptr, solver.c_ptr))

    @property
    def state(self) -> np.ndarray:
        return self._state_buffer.data

    @property
    def n_states(self) -> int:
        return self.problem.n_states

    @property
    def n_params(self) -> int:
        return self.problem.n_params

    @property
    def current_stats(self) -> Dict[str, Any]:
        order = ffi.new('int*', 1)
        check_code(lib.CVodeGetCurrentOrder(self.c_ptr, order))
        
        return {
            "order": order[0],
        }


class Solver:
    def _init_sundials(self):
        n_states = self._problem.n_states
        n_params = self._problem.n_params

        self._state_buffer = sunode.empty_vector(n_states)
        self._state_buffer.data[:] = 0

        self._ode = check(lib.CVodeCreate(self._solver_kind))
        rhs = self._problem.make_sundials_rhs()
        self._rhs = rhs
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserData(self._ode, user_data_p))

        self._set_tolerances(self._abstol, self._reltol)
        if self._constraints is not None:
            assert self._constraints.shape == (n_states,)
            self._constraints_vec = sunode.from_numpy(self._constraints)
            check(lib.CVodeSetConstraints(self._ode, self._constraints_vec.c_ptr))

        self._make_linsol(self._linear_solver_kind, **self._linear_solver_kwargs)

        self._compute_sens = self._sens_mode is not None
        if self._compute_sens:
            sens_rhs = self._problem.make_sundials_sensitivity_rhs()
            self._init_sens(sens_rhs, self._sens_mode)

    def __init__(
            self,
            problem: Problem,
            *,
            abstol: float = 1e-10,
            reltol: float = 1e-10,
            sens_mode: Optional[str] = None,
            scaling_factors: Optional[np.ndarray] = None,
            constraints: Optional[np.ndarray] = None,
            solver='BDF',
            linear_solver="dense",
            linear_solver_kwargs=None,
        ):
        """
        Parameters
        ----------
        problem: sunode Problem
        abstol: float, optional
            Absolute tolerance (default is 1e-10).
        reltol: float, optional
            Relative tolerance (default is 1e-10).
        sense_mode: {"simultaneous", "staggered"}, optional
            Forward sensitivity method (see [this explanation in the SUNDIALS documentation][1]).
        scaling_factors: numpy.ndarray, optional
            Vector of positive scaling factors used for the sensitivity calculations.
        constraints: numpy.ndarray, optional
            Vector of inequality constraints for the solution. The length of the vector must correspond
            to the number of states. See the SUNDIALS documentation for the [constraint options][2].
        solver: {"BDF", "ADAMS"}, optional
            Algorithm for solving the ODE (the default is ``"BDF"``).
        linear_solver: {"dense", "dense_finitediff", "spgmr", "spgmr_finitediff", "band"}, optional
            Type of linear solver to use (the default is "dense").
            If linear_solver is ``"band"``, ``linear_solver_kwargs`` must contain ``lower_bandwidth``
            and ``upper_bandwidth``, defining the lower and upper half-bandwidth of the banded matrix
            (see the [SUNDIALS documentation][3] for details).
        linear_solver_kwargs: dict, optional
            Keyword arguments for the linear solver.

        [1]: https://sundials.readthedocs.io/en/latest/idas/Mathematics_link.html#forward-sensitivity-methods
        [2]: https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html#c.CVodeSetConstraints
        [3]: https://sundials.readthedocs.io/en/latest/sunmatrix/SUNMatrix_links.html#the-sunmatrix-band-module
        """
        if linear_solver_kwargs is None:
            linear_solver_kwargs = {}
        self._problem = problem
        self._user_data = problem.make_user_data()
        self._constraints = constraints

        self._abstol = abstol
        self._reltol = reltol

        self._linear_solver_kind = linear_solver
        self._linear_solver_kwargs = linear_solver_kwargs
        self._sens_mode = sens_mode

        if solver == 'BDF':
            self._solver_kind = lib.CV_BDF
        elif solver == 'ADAMS':
            self._solver_kind = lib.CV_ADAMS
        else:
            assert False

        self._state_names = [
            "_problem",
            "_user_data",
            "_constraints",
            "_abstol",
            "_reltol",
            "_linear_solver_kind",
            "_linear_solver_kwargs",
            "_sens_mode",
            "_solver_kind",
            "_state_names",
        ]

        self._init_sundials()

    def __getstate__(self):
        return {name: self.__dict__[name] for name in self._state_names}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_sundials()

    def _make_linsol(self, linear_solver, **kwargs) -> None:
        n_states = self._problem.n_states
        if linear_solver == "dense":
            self._jac = check(lib.SUNDenseMatrix(n_states, n_states))
            linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
            check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

            self._jac_func = self._problem.make_sundials_jac_dense()
            check(lib.CVodeSetJacFn(self._ode, self._jac_func.cffi))
        elif linear_solver == "dense_finitediff":
            self._jac = check(lib.SUNDenseMatrix(n_states, n_states))
            linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
            check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))
        elif linear_solver == "spgmr_finitediff":
            linsolver = check(lib.SUNLinSol_SPGMR(self._state_buffer.c_ptr, lib.PREC_NONE, 5))
            check(lib.CVodeSetLinearSolver(self._ode, linsolver, ffi.NULL))
            check(lib.SUNLinSolInitialize_SPGMR(linsolver))
        elif linear_solver == "spgmr":
            linsolver = check(lib.SUNLinSol_SPGMR(self._state_buffer.c_ptr, lib.PREC_NONE, 5))
            check(lib.CVodeSetLinearSolver(self._ode, linsolver, ffi.NULL))
            check(lib.SUNLinSolInitialize_SPGMR(linsolver))
            jac_prod = self._problem.make_sundials_jac_prod()
            check(lib.CVodeSetJacTimes(self._ode, ffi.NULL, jac_prod.cffi))
        elif linear_solver == "band":
            upper_bandwidth = kwargs.get("upper_bandwidth", None)
            lower_bandwidth = kwargs.get("lower_bandwidth", None)
            if upper_bandwidth is None or lower_bandwidth is None:
                raise ValueError("Specify 'lower_bandwidth' and 'upper_bandwidth' arguments for banded solver.")
            self._jac = check(lib.SUNBandMatrix(n_states, upper_bandwidth, lower_bandwidth))
            linsolver = check(lib.SUNLinSol_Band(self._state_buffer.c_ptr, self._jac))
            check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))
        else:
            raise ValueError(f"Unknown linear solver: {linear_solver}")

    def _init_sens(self, sens_rhs, sens_mode, scaling_factors=None) -> None:
        if sens_mode == 'simultaneous':
            sens_mode = lib.CV_SIMULTANEOUS
        elif sens_mode == 'staggered':
            sens_mode = lib.CV_STAGGERED
        elif sens_mode == 'staggered1':
            raise ValueError('staggered1 not implemented.')
        else:
            raise ValueError('sens_mode must be one of "simultaneous" and "staggered".')

        self._sens_mode = sens_mode

        n_params = self._problem.n_params
        yS = check(lib.N_VCloneVectorArray(n_params, self._state_buffer.c_ptr))
        vecs = [sunode.vector.Vector(yS[i]) for i in range(n_params)]
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
            check(lib.CVodeSetSensParams(self._ode, NULL_D, pbar_p, NULL_I))

        check(lib.CVodeSensEEtolerances(self._ode))  # TODO
        check(lib.CVodeSetSensErrCon(self._ode, 1))  # TODO

    def _set_tolerances(self, atol=None, rtol=None):
        atol = np.array(atol)
        rtol = np.array(rtol)
        n_states = self._problem.n_states
        if atol.ndim == 1 and rtol.ndim == 1:
            atol = sunode.from_numpy(atol)
            rtol = sunode.from_numpy(rtol)
            assert atol.shape == (n_states,)
            assert rtol.shape == (n_states,)
            check(lib.CVodeVVtolerances(self._ode, rtol.c_ptr, atol.c_ptr))
        elif atol.ndim == 1 and rtol.ndim == 0:
            atol = sunode.from_numpy(atol)
            assert atol.shape == (n_states,)
            check(lib.CVodeSVtolerances(self._ode, rtol, atol.c_ptr))
        elif atol.ndim == 0 and rtol.ndim == 1:
            rtol = sunode.from_numpy(rtol)
            assert rtol.shape == (n_states,)
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

    @property
    def derivative_params_dtype(self):
        return self._problem.params_subset.subset_dtype

    @property
    def remainder_params_dtype(self):
        return self._problem.params_subset.remainder.subset_dtype

    def set_params(self, params):
        self._problem.update_params(self._user_data, params)

    def get_params(self):
        return self._problem.extract_params(self._user_data)

    def set_params_dict(self, params):
        data = self.get_params()
        self._problem.params_subset.from_dict(params, data)
        self.set_params(data)

    def get_params_dict(self):
        return _as_dict(self.get_params())

    def set_derivative_params(self, params):
        self._problem.update_subset_params(self._user_data, params)

    def set_remaining_params(self, params):
        self._problem.update_remaining_params(self._user_data, params)

    def solve(self, t0, tvals, y0, y_out, *, sens0=None, sens_out=None, max_retries=5):
        if self._compute_sens and (sens0 is None or sens_out is None):
            raise ValueError('"sens_out" and "sens0" are required when computin sensitivities.')
        CVodeReInit = lib.CVodeReInit
        CVodeSensReInit = lib.CVodeSensReInit
        CVode = lib.CVode
        CVodeGetSens = lib.CVodeGetSens
        ode = self._ode
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        n_params = self._problem.n_params
        n_states = self._problem.n_states

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        if self._compute_sens:
            sens_buffer_array = self._sens_buffer_array
            sens_data = tuple(buffer.data for buffer in self._sens_buffers)
            for i in range(n_params):
                sens_data[i][:] = sens0[i, :]

        if y0.dtype == self._problem.state_dtype:
            y0 = y0[None].view(np.float64)

        if y0.shape != (n_states,):
            raise ValueError(f"y0 should have shape {(n_states,)} but has shape {y0.shape}.")
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

            for retry in range(max_retries):
                retval = CVode(ode, t, state_c_ptr, time_p, lib.CV_NORMAL)
                if retval == 0:
                    assert time_p[0] == t
                    break
                if retval != TOO_MUCH_WORK:
                    error = ERRORS[retval]
                    raise SolverError(f"Solving ode failed before time={t}: {error} ({retval})")
            else:
                raise SolverError(f"Too many solver retries before time={t}.")

            y_out[i, :] = state_data

            if self._compute_sens:
                retval = CVodeGetSens(ode, time_p, sens_buffer_array)
                if retval == 0:
                    for j in range(n_params):
                        sens_out[i, j, :] = sens_data[j]


class AdjointSolver:
    def __init__(self, problem, *,
                 abstol=1e-10, reltol=1e-10,
                 checkpoint_n=500_000, interpolation='polynomial', constraints=None, solver='BDF', adjoint_solver='BDF'):
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

        if solver == 'BDF':
            self._solver_type = lib.CV_BDF
        elif solver == 'ADAMS':
            self._solver_type = lib.CV_ADAMS
        else:
            raise ValueError(f'Unknown solver {solver}.')

        if adjoint_solver == 'BDF':
            self._adjoint_solver_type = lib.CV_BDF
        elif adjoint_solver == 'ADAMS':
            self._adjoint_solver_type = lib.CV_ADAMS
        else:
            raise ValueError(f'Unknown solver {solver}.')

        self._ode = check(lib.CVodeCreate(self._solver_type))
        check(lib.CVodeInit(self._ode, rhs.cffi, 0., self._state_buffer.c_ptr))

        self._set_tolerances(abstol, reltol)
        if self._constraints is not None:
            self._constraints = np.broadcast_to(constraints, (n_states,)).copy()
            self._constraints_vec = sunode.from_numpy(self._constraints)
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
        check(lib.CVodeCreateB(self._ode, self._adjoint_solver_type, backward_ode))
        self._odeB = backward_ode[0]

        self._state_bufferB = sunode.empty_vector(self._problem.n_states)
        check(lib.CVodeInitB(self._ode, self._odeB, self._adj_rhs.cffi, 0., self._state_bufferB.c_ptr))

        # TODO
        check(lib.CVodeSStolerancesB(self._ode, self._odeB, 1e-10, 1e-10))

        linsolver = check(lib.SUNLinSol_Dense(self._state_bufferB.c_ptr, self._jacB))
        check(lib.CVodeSetLinearSolverB(self._ode, self._odeB, linsolver, self._jacB))

        self._jac_funcB = self._problem.make_sundials_adjoint_jac_dense()
        check(lib.CVodeSetJacFnB(self._ode, self._odeB, self._jac_funcB.cffi))

        user_data_p = ffi.cast('void *', ffi.addressof(ffi.from_buffer(self._user_data.data)))
        check(lib.CVodeSetUserDataB(self._ode, self._odeB, user_data_p))

        self._quad_buffer = sunode.empty_vector(self._problem.n_params)
        self._quad_buffer_out = sunode.empty_vector(self._problem.n_params)
        check(lib.CVodeQuadInitB(self._ode, self._odeB, self._quad_rhs.cffi, self._quad_buffer.c_ptr))

        check(lib.CVodeQuadSStolerancesB(self._ode, self._odeB, 1e-10, 1e-10))
        check(lib.CVodeSetQuadErrConB(self._ode, self._odeB, 1))

    def _make_linsol(self):
        linsolver = check(lib.SUNLinSol_Dense(self._state_buffer.c_ptr, self._jac))
        check(lib.CVodeSetLinearSolver(self._ode, linsolver, self._jac))

        self._jac_func = self._problem.make_sundials_jac_dense()
        check(lib.CVodeSetJacFn(self._ode, self._jac_func.cffi))

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

    @property
    def derivative_params_dtype(self):
        return self._problem.params_subset.subset_dtype

    @property
    def remainder_params_dtype(self):
        return self._problem.params_subset.remainder.subset_dtype

    def set_params(self, params):
        self._problem.update_params(self._user_data, params)

    def get_params(self):
        return self._problem.extract_params(self._user_data)

    def set_params_dict(self, params):
        data = self.get_params()
        self._problem.params_subset.from_dict(params, data)
        self.set_params(data)

    def get_params_dict(self):
        return _as_dict(self.get_params())

    def set_derivative_params(self, params):
        self._problem.update_subset_params(self._user_data, params)

    def set_remaining_params(self, params):
        self._problem.update_remaining_params(self._user_data, params)

    def solve_forward(self, t0, tvals, y0, y_out, *, max_retries=5):
        CVodeReInit = lib.CVodeReInit
        CVodeAdjReInit = lib.CVodeAdjReInit
        CVodeF = lib.CVodeF
        ode = self._ode
        TOO_MUCH_WORK = lib.CV_TOO_MUCH_WORK

        state_data = self._state_buffer.data
        state_c_ptr = self._state_buffer.c_ptr

        if y0.dtype == self._problem.state_dtype:
            y0 = y0[None].view(np.float64)
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

            for retry in range(max_retries):
                retval = CVodeF(ode, t, state_c_ptr, time_p, lib.CV_NORMAL, n_check)
                if retval == 0:
                    assert time_p[0] == t
                    break
                if retval != TOO_MUCH_WORK:
                    error = ERRORS[retval]
                    raise SolverError(f"Solving ode failed before time={t}: {error} ({retval})")
            else:
                raise SolverError(f"Too many solver retries before time={t}.")

            y_out[i, :] = state_data

    def solve_backward(self, t0, tend, tvals, grads, grad_out, lamda_out,
                       lamda_all_out=None, quad_all_out=None, max_retries=50):
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

        state_data[:] = 0
        quad_data[:] = 0
        quad_out_data[:] = 0

        time_p = ffi.new('double*')
        time_p[0] = t0

        ts = [t0] + list(tvals[::-1]) + [tend]
        t_intervals = zip(ts[1:], ts[:-1])
        grads = [None] + list(grads)

        for i, ((t_lower, t_upper), grad) in enumerate(zip(t_intervals, reversed(grads))):
            if t_lower < t_upper:
                check(CVodeReInitB(ode, odeB, t_upper, state_c_ptr))
                check(CVodeQuadReInitB(ode, odeB, quad_c_ptr))

                for retry in range(max_retries):
                    retval = CVodeB(ode, t_lower, lib.CV_NORMAL)
                    if retval == 0:
                        break
                    if retval != TOO_MUCH_WORK:
                        error = ERRORS[retval]
                        raise SolverError(f"Solving ode failed between time {t_upper} and "
                                          f"{t_lower}: {error} ({retval})")
                else:
                    raise SolverError(f"Too many solver retries between time {t_upper} and {t_lower}.")

                check(CVodeGetB(ode, odeB, time_p, state_c_ptr))
                check(CVodeGetQuadB(ode, odeB, time_p, quad_out_c_ptr))
                quad_data[:] = quad_out_data[:]
                assert time_p[0] == t_lower, (time_p[0], t_lower)

            if grad is not None:
                state_data[:] -= grad

                if lamda_all_out is not None:
                    lamda_all_out[-i, :] = state_data
                if quad_all_out is not None:
                    quad_all_out[-i, :] = quad_data

        grad_out[:] = quad_out_data
        lamda_out[:] = state_data
