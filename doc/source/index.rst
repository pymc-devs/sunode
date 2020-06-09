Sunode
======

*Solve ODEs fast and integrate them into autodiff frameworks.*

Scipy has had decent solvers for ordinary differential equations for a long
time, they come with some limitations however:

- They tend to be quite slow. The most recent `solve_ivp` is implemented
  in pure python, and even the older interfaces to C libraries do not
  provide a way to remove python overhead completely.
- They do not have any support for computing derivatives of solutions
  or for computing gradients of functionals of solutions.
- The solver interface assumes that the state variable is a simple vector.
  This works, but in many problems this leaves it to the user to store the
  often mulitple multidimensional variables in this array. This can be error
  prone, and often makes the code much harder to understand and debug.

Sunode aims to work around those issues: We wrap the well established C library
sundials to provide the solver and support for derivatives in forward and
adjoint mode.

In sunode the ODE is declared with named (and possibly nested) state variables
and parameters. The right-hand-side function is defined either manually using
numba, or using sympy variables. In the latter case we generate python code
through AST manipulation for necessary functions and compile them using numba.
This allows us to solve an ode repeatetly with almost no python overhead.

The original use-case for this library was better support for solving ODEs
within bayesian models in PyMC3, but is useable in different contexts as well.

Contents
--------

.. toctree::
   :maxdepth: 2

   quickstart
   manual_calling
   limitations


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
