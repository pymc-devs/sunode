.. _usage-basic:

Usage without theano
====================

We will use the Lotka-Volterra equations as a simple example: We have two
animal populations (Lyxes and Hares). The Lyxes die of natural causes with a
rate proportianl to their number, and are born with a rate proportional to the
number of lyxes and hares. Hares are born with a rate proportial to their
current number, and die when eaten by a lyx. We get:

.. math::
   \frac{dH}{dt} = \alpha H - \beta LH \\ \frac{dL}{dt} = \delta LH - \gamma L

If we want to solve this ODE without the support of theano or PyMC3, we need to
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
            'log_hares': dhares / hares
            'log_lynxes': dlynxes / lynxes,
        }

.. warning::
   
   This right-hand-side function is usually only called once to collect the
   sympy expressions of the derivatives. Control flow within this function
   might behave in unexpected ways if you are new to this concept. It is the
   same thing as with theano, pytorch or tensorflow in graph mode. This means
   that something like this will **not** work as expected::

       value = 1
       if y.some_state > 1:
           value += 1

   ``y.some_param`` is a sympy variable, not a number, so this comparison will
   always be False.
   For more details see :ref:`rhs-function`.

After defining states, parameters and right-hand-side function we can create a
`SympyProblem` instance::

    problem = sunode.SympyProblem(
        params=params,
        states=states,
        rhs_sympy=lotka_volterra
    )
