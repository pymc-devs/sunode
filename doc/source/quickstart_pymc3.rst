.. _quickstart_pymc3:

Quickstart with PyMC3
=====================

Official conda packages are not yet available in conda-forge. You can either
build your own conda package or install the master version from github.

To use the development version, we create a new conda environment with the requirements
and install using pip::

    git clone git@github.com:aseyboldt/sunode
    # Or if no ssh key is configured:
    git clone https://github.com/aseyboldt/sunode

    cd sunode
    conda create -n sunode -c conda-forge python numba pymc3 sympy pandas xarray sundials
    conda activate sunode
    pip install -e .

To build and install a conda package locally, we clone the repo as above, and then run::

    cd sunode
    conda build ./conda

    conda install -c local sunode
    # Install it into a new environment:
    conda create -n sunode -c local -c conda-forge sunode

Instructions for installation on Windows can be found
`here <https://gist.github.com/michaelosthege/5bd75c99cd5e806ee049b02ed528bab3>`_

Sampling Bayesian models with Hamiltonian MCMC involving an ODE is where the
features of sunode shine.  We need to solve the ODE (ofter rather small ODEs) a
large number of times, so Python overhead will hurt us a lot, and we need to
compute gradients as well. Sunode provides some utility functions that make it
easy to include an ODE into a PyMC3 model.  If you want to use it in a
different context, see :ref:`usage-basic`.


We will use the Lotka-Volterra equations as example:

.. math::
   \frac{dH}{dt} = \alpha H - \beta LH \\ \frac{dL}{dt} = \delta LH - \gamma L


We'll use some time artificial data:::

    import numpy as np
    import sunode
    import sunode.wrappers.as_theano
    import pymc3 as pm

    times = np.arange(1900,1921,1)
    lynx_data = np.array([
        4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4,
        8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6
    ])
    hare_data = np.array([
        30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
        27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7
    
We also define a function for the right-hand-side of the ODE:::

    def lotka_volterra(t, y, p):
        """Right hand side of Lotka-Volterra equation.

        All inputs are dataclasses of sympy variables, or in the case
        of non-scalar variables numpy arrays of sympy variables.
        """
        return {
            'hares': p.alpha * y.hares - p.beta * y.lynxes * y.hares,
            'lynxes': p.delta * y.hares * y.lynxes - p.gamma * y.lynxes,
        }

.. note::

   For more details about how this function behaves, see :ref:`rhs-function`.

In the next step we define priors for our parameters in PyMC3. To illustrate
that the parameters do not have to be PyMC3 variables themselves, but can also
be derived though some computations, we put priors not on :math:`\alpha, \beta,
\gamma` and :math:`\delta`, but on parameters like the frequency of the
hare-lynx cycle and the steady-state ratio of hares to lynxes.::

    with pm.Model() as model:
        hares_start = pm.HalfNormal('hares_start', sd=50)
        lynx_start = pm.HalfNormal('lynx_start', sd=50)

        ratio = pm.Beta('ratio', alpha=0.5, beta=0.5)

        fixed_hares = pm.HalfNormal('fixed_hares', sd=50)
        fixed_lynx = pm.Deterministic('fixed_lynx', ratio * fixed_hares)

        period = pm.Gamma('period', mu=10, sd=1)
        freq = pm.Deterministic('freq', 2 * np.pi / period)

        log_speed_ratio = pm.Normal('log_speed_ratio', mu=0, sd=0.1)
        speed_ratio = np.exp(log_speed_ratio)

        # Compute the parameters of the ode based on our prior parameters
        alpha = pm.Deterministic('alpha', freq * speed_ratio * ratio)
        beta = pm.Deterministic('beta', freq * speed_ratio / fixed_hares)
        gamma = pm.Deterministic('gamma', freq / speed_ratio / ratio)
        delta = pm.Deterministic('delta', freq / speed_ratio / fixed_hares / ratio)

Now, we define the names, (symbolic) values and shapes of the parameters and initial values::

    with model:
        y0 = {
            # The initial number of hares is the random variable `hares_start`,
            # and it has shape (), so it is a scalar value.
            'hares': (hares_start, ()),
            'lynxes': (lynx_start, ()),
        }

        params = {
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            # Parameters (or initial states) do not have to be random variables,
            # they can also be fixed numpy values. In this case the shape
            # is infered automatically. Sunode will not compute derivatives
            # with respect to fixed parameters or initial states.
            'unused_extra': np.zeros(5),
        }

We solve the ODE using the ``solve_ivp`` function from sunode::

    with model:
        from sunode.wrappers.as_theano import solve_ivp
        solution, *_ = solve_ivp(
            y0=y0,
            params=params,
            rhs=lotka_volterra,
            # The time points where we want to access the solution
            tvals=times,
            t0=times[0],
        )

We are only missing the likelihood now::

    with model:
        # We can access the individual variables of the solution using the
        # variable names.
        pm.Deterministic('hares_mu', y_hat['hares'])
        pm.Deterministic('lynx_mu', y_hat['lynx'])

        sd = pm.HalfNormal('sd')
        pm.Lognormal('hares', mu=y_hat['hares'], sd=sd, observed=hare_data)
        pm.Lognormal('lynx', mu=y_hat['lynx'], sd=sd, observed=lynx_data)

We can sample from the posterior with the gradient-based PyMC3 samplers:::

    with model:
        trace = pm.sample()

At the moment it is unfortunately not possible to pickle the ODE solver (I'm
working on an implementation), so sampling with multiple chains is only possible,
if the python multiprocessing is using forks instead of spawning new processes.
This is the default on Linux, but on Mac it has to be specified manually::

    import multiprocessing as mp
    mp.set_start_method('fork')

Windows does not support this at all. You can however disable parallel sampling
by setting ``cores=1`` in ``pm.sample()``.
