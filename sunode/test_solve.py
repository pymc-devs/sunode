import numpy as np

from sunode import SympyProblem
from sunode.solver import Solver, AdjointSolver


def test_basic():
    def rhs(t, y, p):
        return {
            'x': y.x,
        }

    states = {
        'x': (),
    }

    params = {
        'b': ()
    }
    problem = SympyProblem(params, states, rhs, derivative_params=[])
    Solver(problem)


def test_empty_params():
    def rhs(t, y, p):
        return {
            'x': y.x,
        }

    states = {
        'x': (),
    }

    params = {}
    problem = SympyProblem(params, states, rhs, derivative_params=[])
    Solver(problem)


def test_empty_params_nested():
    def rhs(t, y, p):
        return {
            'x': y.x + p.a.b,
        }

    states = {
        'x': (),
    }

    params = {
        'a': {
            'b': ()
        }
    }
    problem = SympyProblem(params, states, rhs, derivative_params=[])
    Solver(problem)


def test_states_params_nested():
    def rhs(t, y, p):
        return {
            'x': {'y': {'z': y.x.y.z + p.a.b}}
        }

    states = {
        'x': {
            'y': {
                'z': ()
            }
        }
    }

    params = {
        'a': {
            'b': ()
        }
    }
    problem = SympyProblem(params, states, rhs, derivative_params=[])
    Solver(problem)


def check_call_solve(solver, params, deriv):
    solver.set_params_dict(params)

    time = np.linspace(0, 1)
    if deriv == 'forward':
        y_buffer, sense_buffer = solver.make_output_buffers(time)
        solver.solve(
            0,
            time,
            np.ones_like(y_buffer[0]),
            y_buffer,
            sens0=np.zeros_like(sense_buffer[0]),
            sens_out=sense_buffer
        )
    elif deriv == 'backward':
        y_buffer, grads_buffer, lamda_buffer = solver.make_output_buffers(time)
        solver.solve_forward(0, time, np.ones_like(y_buffer[0]), y_buffer)

        grads = np.ones((len(time), y_buffer.shape[-1]))
        solver.solve_backward(
            time[-1],
            time[0],
            time,
            grads,
            grads_buffer,
            lamda_buffer
        )
    elif deriv is None:
        y_buffer = solver.make_output_buffers(time)
        solver.solve(
            0,
            time,
            np.ones_like(y_buffer[0]),
            y_buffer,
        )
    else:
        assert False


def test_declare_sens():
    def rhs(t, y, p):
        return {
            'x': y.x + p.a.b,
        }

    states = {
        'x': (),
    }

    params = {
        'a': {
            'b': ()
        }
    }

    param_vals = {
        'a': {
            'b': 0.2
        }
    }

    problem = SympyProblem(params, states, rhs, derivative_params=[('a', 'b')])

    solver = Solver(problem, sens_mode="simultaneous")
    check_call_solve(solver, param_vals, "forward")

    solver = Solver(problem, sens_mode="staggered")
    check_call_solve(solver, param_vals, "forward")

    solver = Solver(problem)
    check_call_solve(solver, param_vals, None)

    solver = AdjointSolver(problem)
    check_call_solve(solver, param_vals, "backward")


def test_linear_solver_kwarg():
    def rhs(t, y, p):
        return {
            'x': y.x,
        }

    states = {
        'x': (),
    }

    params = {
        'b': ()
    }
    param_vals = {
        'b': 0.2
    }
    problem = SympyProblem(params, states, rhs, derivative_params=[])
    linear_solver_opts = ["dense", "dense_finitediff", "spgmr_finitediff", "spgmr", "band"]
    for linear_solver in linear_solver_opts:
        if linear_solver == "band":
            linear_solver_kwargs = {"upper_bandwidth": 1, "lower_bandwidth": 1}
        else:
            linear_solver_kwargs = {}
        solver = Solver(problem, linear_solver=linear_solver, linear_solver_kwargs=linear_solver_kwargs)
        check_call_solve(solver, param_vals, None)
