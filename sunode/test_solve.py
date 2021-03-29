from sunode import SympyProblem
from sunode.solver import Solver


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
