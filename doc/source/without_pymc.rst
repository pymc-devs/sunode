.. _usage-basic:

Stand-alone usage
=================

We will use the Lotka-Volterra equations as a simple example: We have two
animal populations (Lynxes and Hares). The Lynxes die of natural causes with a
rate proportianl to their number, and are born with a rate proportional to the
number of lynxes and hares. Hares are born with a rate proportial to their
current number, and die when eaten by a lynx. We get:

.. math::
   \frac{dH}{dt} = \alpha H - \beta LH \\ \frac{dL}{dt} = \delta LH - \gamma L

If we want to solve this ODE without the support of PyTensor or PyMC, we need to
first declare the parameters and states we are using. We have four parameters
and two states, and each one is a scalar values, so it has shape ()::

    params = {
        'alpha': (),
        'beta': (),
        'gamma': (),
        'delta': (),
    }

    states = {
        'hares': (),
        'lynxes': (),
    }

We also need to define the right-hand-side function, the derivaties
:math:`\tfrac{d}{dt}H` and :math:`\tfrac{d}{dt}L`::

    def lotka_volterra(t, y, p):
        """Right hand side of Lotka-Volterra equation.

        All inputs are dataclasses of sympy variables, or in the case
        of non-scalar variables numpy arrays of sympy variables.
        """
        return {
            'hares': p.alpha * y.hares - p.beta * y.lynxes * y.hares,
            'lynxes': p.delta * y.hares * y.lynxes - p.gamma * y.lynxes,
        }

We return a dict with all states of the ODE, containing the derivatives of that
variable. We can access the current time as the first argument to this
function, the current states through the second and the parameters though the
third. The values `y.hares`, `p.alpha` etc. are sympy variables. If they are
declared as arrays, they will be **numpy arrays of sympy variables**. If you
want to apply sympy functions elementwise, you have to wrap the sympy function
with `np.vectorize` first. So for example the log-transformed version of this
ODE might look like this::

    import sympy as sym

    def lotka_volterra_log(t, y, p):
        exp = np.vectorize(sym.exp)

        hares = exp(y.log_hares)
        lynxes = exp(y.log_lynxes)

        dhares = p.alpha * hares - p.beta * lynxes * hares
        dlynxes = p.delta * hares * lynxes - p.gamma * lynxes
        return {
            'log_hares': dhares / hares,
            'log_lynxes': dlynxes / lynxes,
        }

.. warning::
   
   This right-hand-side function is usually only called once to collect the
   sympy expressions of the derivatives. Control flow within this function
   might behave in unexpected ways if you are new to this concept. It is the
   same thing as with PyTensor, pytorch or tensorflow in graph mode. This means
   that something like this will **not** work as expected::

       value = 1
       if y.some_state > 1:
           value += 1

   ``y.some_param`` is a sympy variable, not a number, so this comparison will
   always be False.
   For more details see :ref:`rhs-function`.

After defining states, parameters and right-hand-side function we can create a
`SympyProblem` instance::

    import sunode

    problem = sunode.SympyProblem(
        params=params,
        states=states,
        rhs_sympy=lotka_volterra,
        derivative_params=()
    )

The problem provides structured numpy dtypes for states and parameters
(``problem.state_dtype`` and ``problem.params_dtype``), and can compile
functions necessary for solving the ode and computing gradients. We can
create a solver for no derivatives or with forward derivatives
(``sunode.Solver``), or a solver that can compute gradients using
the adjoint ODE (``sunode.AdjointSolver``).::

    solver = sunode.solver.Solver(problem, solver='BDF')

We can use numpy structured arrays as input, so that we don't need to
think about how the different variables are stored in the array.
This does not introduce runtime overhead.::

    import numpy as np
    y0 = np.zeros((), dtype=problem.state_dtype)
    y0['hares'] = 1
    y0['lynxes'] = 0.1

    # At which time points do we want to evalue the solution
    tvals = np.linspace(0, 10)

We can also specify the parameters by name:::

    solver.set_params_dict({
        'alpha': 0.1,
        'beta': 0.2,
        'gamma': 0.3,
        'delta': 0.4,
    })

    output = solver.make_output_buffers(tvals)
    solver.solve(t0=0, tvals=tvals, y0=y0, y_out=output)

We can convert the solution to an xarray Dataset or access the
individual states as numpy record array::

    solver.as_xarray(tvals, output).solution_hares.plot()
    plt.plot(tvals, output.view(problem.state_dtype)['hares'])
