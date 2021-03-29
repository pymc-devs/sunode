import numpy as np
import pymc3 as pm

import sunode.wrappers


def test_nodiff_params():
    def dydt_dict(t, y, p):
        return {
            'A': y.A,
            'B': y.B,
            'C': y.C,
        }

    with pm.Model():
        t = np.linspace(0, 1)

        y0 = {
            'A': (pm.Uniform("A"), ()),
            'B': np.array(1.),
            'C': np.array(1.)
        }

        params = {
            'alpha': np.array(1.),
            'beta': np.array(1.),
            'extra': np.array([0.])
        }

        _ = sunode.wrappers.as_aesara.solve_ivp(
            y0=y0,
            params=params,
            rhs=dydt_dict,
            tvals=t,
            t0=t[0],
            derivatives="forward",
            solver_kwargs=dict(sens_mode="simultaneous")
        )