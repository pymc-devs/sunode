try:
    import pytensor.tensor as pt
    from pytensor.graph.basic import Constant, Variable
    from pytensor.graph.fg import MissingInputError
    from pytensor.graph.op import Op
    from pytensor.gradient import grad_not_implemented
except ModuleNotFoundError:
    import aesara.tensor as pt
    from aesara.graph.basic import Constant, Variable
    from aesara.graph.fg import MissingInputError
    from aesara.graph.op import Op
    from aesara.gradient import grad_not_implemented
import copy
from typing import Dict, Optional, Any, Callable

import numpy as np
import pandas as pd
import sympy as sym

from sunode.solver import SolverError
from sunode.dtypesubset import as_flattened
from sunode import basic, symode, solver
import sunode.symode.problem
from sunode.symode.problem import SympyProblem


def solve_ivp(
    t0: float,
    y0: np.ndarray,
    params: Dict[str, Any],
    tvals: np.ndarray,
    rhs: Callable[[sym.Symbol, np.ndarray, np.ndarray], Dict[str, Any]],
    derivatives: str = 'adjoint',
    coords: Optional[Dict[str, pd.Index]] = None,
    make_solver=None,
    derivative_subset=None,
    solver_kwargs=None,
    simplify=None,
) -> Any:
    dtype = basic.data_dtype
    if solver_kwargs is None:
        solver_kwargs={}

    if derivatives == "forward":
        params = params.copy()
        params["__initial_values"] = y0

    def read_dict(vals, name=None):
        if isinstance(vals, dict):
            return {name: read_dict(item, name) for name, item in vals.items()}
        else:
            if isinstance(vals, tuple):
                tensor, dim_names = vals
            else:
                try:
                    tensor, dim_names = vals, pt.as_tensor_variable(vals, dtype="float64").shape.eval()
                except MissingInputError as e:
                    raise ValueError(
                        'Shapes of tensors need to be statically '
                        'known or given explicitly.') from e
            if isinstance(dim_names, (str, int)):
                dim_names = (dim_names,)
            tensor = pt.as_tensor_variable(tensor, dtype="float64")
            if tensor.ndim != len(dim_names):
                raise ValueError(
                    f"Dimension mismatch for {name}: Value has rank {tensor.ndim}, "
                    f"but {len(dim_names)} was specified."
                )
            assert np.dtype(tensor.dtype) == dtype, tensor
            tensor_dtype = np.dtype(tensor.dtype)
            if tensor_dtype != dtype:
                raise ValueError(
                    f"Dtype mismatch for {name}: Got {tensor_dtype} but expected {dtype}."
                )
            return dim_names

    y0_dims = read_dict(y0)
    params_dims = read_dict(params)

    if derivative_subset is None:
        derivative_subset = []
        for path, val in as_flattened(params).items():
            if isinstance(val, tuple):
                tensor, _ = val
            else:
                tensor = val
            if isinstance(tensor, Variable):
                if not isinstance(tensor, Constant):
                    derivative_subset.append(path)

    problem = symode.problem.SympyProblem(
        params_dims, y0_dims, rhs, derivative_subset, coords=coords, simplify=simplify)

    flat_tensors = as_flattened(params)
    vars = []
    for path in problem.params_subset.subset_paths:
        tensor = flat_tensors[path]
        if isinstance(tensor, tuple):
            tensor, _ = tensor
        vars.append(pt.as_tensor_variable(tensor, dtype="float64").reshape((-1,)))
    if vars:
        params_subs_flat = pt.concatenate(vars)
    else:
        params_subs_flat = pt.as_tensor_variable(np.zeros(0), dtype="float64")

    vars = []
    for path in problem.params_subset.remainder.subset_paths:
        tensor = flat_tensors[path]
        if isinstance(tensor, tuple):
            tensor, _ = tensor
        vars.append(pt.as_tensor_variable(tensor, dtype="float64").reshape((-1,)))
    if vars:
        params_remaining_flat = pt.concatenate(vars)
    else:
        params_remaining_flat = pt.as_tensor_variable(np.zeros(0), dtype="float64")

    flat_tensors = as_flattened(y0)
    vars = []
    for path in problem.state_subset.paths:
        tensor = flat_tensors[path]
        if isinstance(tensor, tuple):
            tensor, _ = tensor
        vars.append(pt.as_tensor_variable(tensor, dtype="float64").reshape((-1,)))
    y0_flat = pt.concatenate(vars)

    if derivatives == 'adjoint':
        sol = solver.AdjointSolver(problem, **solver_kwargs)
        wrapper = SolveODEAdjoint(sol, t0, tvals)
        flat_solution = wrapper(y0_flat, params_subs_flat, params_remaining_flat)
        solution = problem.flat_solution_as_dict(flat_solution)
        return solution, flat_solution, problem, sol, y0_flat, params_subs_flat
    elif derivatives == 'forward':
        if not "sens_mode" in solver_kwargs:
            raise ValueError("When `derivatives=True`, the `solver_kwargs` must contain one of `sens_mode={\"simultaneous\" | \"staggered\"}`.")
        sol = solver.Solver(problem, **solver_kwargs)
        wrapper = SolveODE(sol, t0, tvals)
        flat_solution, flat_sens = wrapper(y0_flat, params_subs_flat, params_remaining_flat)
        solution = problem.flat_solution_as_dict(flat_solution)
        return solution, flat_solution, problem, sol, y0_flat, params_subs_flat, flat_sens, wrapper
    elif derivatives in [None, False]:
        sol = solver.Solver(problem, sens_mode=False)
        assert False


class SolveODE(Op):
    itypes = [pt.dvector, pt.dvector, pt.dvector]
    otypes = [pt.dmatrix, pt.dtensor3]
    
    __props__ = ('_solver_id', '_t0', '_tvals_id')
    
    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._sens_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)
        self._deriv_dtype = self._solver.derivative_params_dtype
        self._fixed_dtype = self._solver.remainder_params_dtype

        n_states, n_params = self._solver._problem.n_states, self._solver._problem.n_params
        problem = self._solver._problem

        def get(val, path):
            if not path:
                return val
            else:
                key, *path = path
                return get(val[key], path)

        sens0 = []
        for path, shape in problem.params_subset.flat_shapes.items():
            if path not in problem.params_subset.subset_paths:
                continue
            n_items = np.prod(shape, dtype=int)

            sens0_item = np.zeros((n_items,), dtype=problem.state_dtype)
            if path and path[0] == '__initial_values':
                _, *path = path

                if shape == ():
                    subset = get(sens0_item[0], path[:-1])
                    subset[path[-1]] = 1.
                else:
                    for i in range(n_items):
                        subset = get(sens0_item[i], path)
                        subset.ravel()[i] = 1.
            sens0.extend(sens0_item)
        sens0 = np.array(sens0).view(np.float64)
        self._sens0 = sens0.reshape((n_params, n_states))

    def perform(self, node, inputs, outputs):
        y0, params, params_fixed = inputs

        self._solver.set_derivative_params(params.view(self._deriv_dtype)[0])
        self._solver.set_remaining_params(params_fixed.view(self._fixed_dtype)[0])

        try:
            self._solver.solve(self._t0, self._tvals, y0, self._y_out,
                            sens0=self._sens0, sens_out=self._sens_out)
        except SolverError:
            self._y_out[...] = np.nan
            self._sens_out[...] = np.nan
        
        outputs[0][0] = self._y_out.copy()
        outputs[1][0] = self._sens_out.copy()

    def grad(self, inputs, g):
        g, g_grad = g
        
        assert str(g_grad) == '<DisconnectedType>'
        solution, sens = self(*inputs)
        return [
            pt.zeros_like(inputs[0]),
            pt.sum(g[:, None, :] * sens, (0, -1)),
            grad_not_implemented(self, 2, inputs[-1])
        ]


class SolveODEAdjoint(Op):
    itypes = [pt.dvector, pt.dvector, pt.dvector]
    otypes = [pt.dmatrix]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)
        self._deriv_dtype = self._solver.derivative_params_dtype
        self._fixed_dtype = self._solver.remainder_params_dtype

    def perform(self, node, inputs, outputs):
        y0, params, params_fixed = inputs

        self._solver.set_derivative_params(params.view(self._deriv_dtype)[0])
        self._solver.set_remaining_params(params_fixed.view(self._fixed_dtype)[0])

        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
        except SolverError as e:
            self._y_out[:] = np.nan

        outputs[0][0] = self._y_out.copy()

    def grad(self, inputs, g):
        g, = g

        y0, params, params_fixed = inputs
        backward = SolveODEAdjointBackward(self._solver, self._t0, self._tvals)
        lamda, gradient = backward(y0, params, params_fixed, g)
        return [-lamda, gradient, grad_not_implemented(self, 2, params_fixed)]


class SolveODEAdjointBackward(Op):
    itypes = [pt.dvector, pt.dvector, pt.dvector, pt.dmatrix]
    otypes = [pt.dvector, pt.dvector]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def make_nodes(self, *inputs):
        if len(inputs) != len(self.itypes):
            raise ValueError(
                f"We expected {len(self.itypes)} inputs but got {len(inputs)}."
            )
        if not all(it.in_same_class(inp.type) for inp, it in zip(inputs, self.itypes)):
            raise TypeError(
                f"Invalid input types for Op {self}:\n"
                + "\n".join(
                    f"Input {i}/{len(inputs)}: Expected {inp}, got {out}"
                    for i, (inp, out) in enumerate(
                        zip(self.itypes, (inp.type for inp in inputs)),
                        start=1,
                    )
                    if inp != out
                )
            )
        return Apply(self, inputs, [o() for o in self.otypes])

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)
        self._deriv_dtype = self._solver.derivative_params_dtype
        self._fixed_dtype = self._solver.remainder_params_dtype

    def perform(self, node, inputs, outputs):
        y0, params, params_fixed, grads = inputs

        self._solver.set_derivative_params(params.view(self._deriv_dtype)[0])
        self._solver.set_remaining_params(params_fixed.view(self._fixed_dtype)[0])

        # TODO We don't really need to do the forward solve if we make sure
        # that it was executed previously, but it isn't very expensive
        # compared with the backward pass anyway.
        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
            self._solver.solve_backward(self._tvals[-1], self._t0, self._tvals,
                                        grads, self._grad_out, self._lamda_out)
        except SolverError as e:
            self._lamda_out[:] = np.nan
            self._grad_out[:] = np.nan

        outputs[0][0] = self._lamda_out.copy()
        outputs[1][0] = self._grad_out.copy()
