You can find the documentation [here](https://sunode.readthedocs.io/en/latest/index.html).

# Sunode – Solving ODEs in python

Sunode wraps the sundials solvers ADAMS and BDF, and their support for solving
adjoint ODEs in order to compute gradients of the solutions.  The required
right-hand-side function and some derivatives are either supplied manually or
via sympy, in which case sunode will generate the abstract syntax tree of the
functions using symbolic differentiation and common subexpression elimination.
In either case the functions are compiled using numba and the resulting
C-function is passed to sunode, so that there is no python overhead.

sunode comes with a theano wrapper so that parameters of an ode can be estimated
using pymc3.

### Installation
```bash
git clone git@github.com:aseyboldt/sunode
cd sunode
conda create -n sunode -c conda-forge python numba pymc3 sympy pandas xarray sundials
conda activate sunode
pip install -e .
```

Installation instructions for windows can be found [here](https://gist.github.com/michaelosthege/5bd75c99cd5e806ee049b02ed528bab3).

### Solve an ode outside of a pymc3 model

We will use the Lotka-Volterra equations as an example ODE:

$$
\freq{dH}{dt} = \alpha H - \beta LH\\
\freq{dL}{dt} = \delta LH - \gamma L
$$

```python
def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }


problem = sunode.symode.SympyProblem(
    params={
        # We need to specify the shape of each parameter.
        # Any empty tuple corresponds to a scalar value.
        'alpha': (),
        'beta': (),
        'gamma': (),
        'delta': (),
    },
    states={
        # The same for all state variables
        'hares': (),
        'lynx': (),
    },
    rhs_sympy=lotka_volterra,
    derivative_params=[
        # We need to specify with respect to which variables
        # gradients should be computed.
        ('alpha',),
        ('beta',),
    ],
)

# The solver generates uses numba and sympy to generate optimized C functions
# for the right-hand-side and if necessary for the jacobian, adoint and
# quadrature functions for gradients.
solver = sunode.solver.Solver(problem, compute_sens=False, solver='BDF')


tvals = np.linspace(0, 10)
# We can use numpy structured arrays as input, so that we don't need
# to think about how the different variables are stored in the array.
# This does not introduce any runtime overhead during solving.
y0 = np.zeros((), dtype=problem.state_dtype)
y0['hares'] = 1
y0['lynx'] = 0.1

# We can also specify the parameters by name:
solver.set_params_dict({
    'alpha': 0.1,
    'beta': 0.2,
    'gamma': 0.3,
    'delta': 0.4,
})

output = solver.make_output_buffers(tvals)
solver.solve(t0=0, tvals=tvals, y0=y0, y_out=output)

# We can convert the solution to an xarray Dataset
solver.as_xarray(tvals, output).solution_hares.plot()

# Or we can convert it to a numpy record array
plt.plot(output.view(problem.state_dtype)['hares']
```

For this example the BDF solver (which isn't the best solver for a small non-stiff
example problem like this) takes ~200μs, while the scipy.integrate.solve_ivp solver
take about 40ms at a tolerance of 1e-10, 1e-10. So we are faster by a factor of 200.
This advantage will get somewhat smaller for large problems however, when the
python overhead of the ODE solver has a smaller impact.

### Usage in pymc3

Let's use the same ODE, but fit the parameters using pymc3, and gradients
computed using sunode.

We'll use some time artificial data:
```python
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
])
```

We place priors on the steady-state ratio of hares and lynxes:

```python


def lotka_volterra(t, y, p):
    """Right hand side of Lotka-Volterra equation.

    All inputs are dataclasses of sympy variables, or in the case
    of non-scalar variables numpy arrays of sympy variables.
    """
    return {
        'hares': p.alpha * y.hares - p.beta * y.lynx * y.hares,
        'lynx': p.delta * y.hares * y.lynx - p.gamma * y.lynx,
    }


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
    
    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
	    # The initial conditions of the ode. Each variable
	    # needs to specify a theano or numpy variable and a shape.
	    # This dict can be nested.
            'hares': (hares_start, ()),
            'lynx': (lynx_start, ()),
        },
        params={
	    # Each parameter of the ode. sunode will only compute derivatives
	    # with respect to theano variables. The shape needs to be specified
	    # as well. It it infered automatically for numpy variables.
	    # This dict can be nested.
            'alpha': (alpha, ()),
            'beta': (beta, ()),
            'gamma': (gamma, ()),
            'delta': (delta, ()),
            'extra': np.zeros(1),
        },
	# A functions that computes the right-hand-side of the ode using
	# sympy variables.
        rhs=lotka_volterra,
	# The time points where we want to access the solution
        tvals=times,
        t0=times[0],
    )
    
    # We can access the individual variables of the solution using the
    # variable names.
    pm.Deterministic('hares_mu', y_hat['hares'])
    pm.Deterministic('lynx_mu', y_hat['lynx'])
    
    sd = pm.HalfNormal('sd')
    pm.Lognormal('hares', mu=y_hat['hares'], sd=sd, observed=hare_data)
    pm.Lognormal('lynx', mu=y_hat['lynx'], sd=sd, observed=lynx_data)
```

We can sample using pymc3 (I suspect you might need `cores=1` on windows for the moment):
```
with model:
    trace = pm.sample(tune=1000, draws=1000, chains=6, cores=6)
```

Many solver options can not be specified with a nice interface at the moment,
we can call the raw sundials functions however (this should change in the future):

```
lib = sunode._cvodes.lib
lib.CVodeSStolerances(solver._ode, 1e-10, 1e-10)
lib.CVodeSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeQuadSStolerancesB(solver._ode, solver._odeB, 1e-8, 1e-8)
lib.CVodeSetMaxNumSteps(solver._ode, 5000)
lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 5000)
```
