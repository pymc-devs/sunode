import numpy as np
import pytensor
import pytensor.tensor as pt

import sunode.wrappers


def test_nodiff_params():
    def dydt_dict(t, y, p):
        return {
            'A': y.A,
            'B': y.B,
            'C': y.C,
        }
    A = pt.dscalar("A")
    A.tag.test_value = np.array(0.9)


    time = pt.linspace(0, 1, 5)

    y0 = {
        'A': (A, ()),
        'B': np.array(1.),
        'C': np.array(1.)
    }

    beta = pt.dscalar("beta")

    params = {
        'alpha': np.array(1.),
        'beta': beta,
        'extra': np.array([0.])
    }

    solution, *_ = sunode.wrappers.as_pytensor.solve_ivp(
        y0=y0,
        params=params,
        rhs=dydt_dict,
        tvals=time,
        t0=0.,
        derivatives="forward",
        solver_kwargs=dict(sens_mode="simultaneous")
    )

    grad = pt.grad(solution["A"].sum(), time)

    func = pytensor.function([A, beta], [solution["A"], solution["B"], grad])
    assert func(0.2, 1.)[0].shape == (5,)
    assert func(0.2, 1.)[2].shape == (5,)

    solution, *_ = sunode.wrappers.as_pytensor.solve_ivp(
        y0=y0,
        params=params,
        rhs=dydt_dict,
        tvals=time,
        t0=0.,
        derivatives="adjoint",
    )

    grad = pt.grad(solution["A"].sum(), time)

    func = pytensor.function([A, beta], [solution["A"], solution["B"], grad])
    assert func(0.2, 1.)[0].shape == (5,)
    assert func(0.2, 1.)[2].shape == (5,)
