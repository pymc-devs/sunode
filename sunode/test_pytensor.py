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


    time = np.linspace(0, 1)

    y0 = {
        'A': (A, ()),
        'B': np.array(1.),
        'C': np.array(1.)
    }

    params = {
        'alpha': np.array(1.),
        'beta': np.array(1.),
        'extra': np.array([0.])
    }

    solution, *_ = sunode.wrappers.as_pytensor.solve_ivp(
        y0=y0,
        params=params,
        rhs=dydt_dict,
        tvals=time,
        t0=time[0],
        derivatives="forward",
        solver_kwargs=dict(sens_mode="simultaneous")
    )

    func = pytensor.function([A], [solution["A"], solution["B"]])
    assert func(0.2)[0].shape == time.shape