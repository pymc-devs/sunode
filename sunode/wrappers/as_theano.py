import theano
import theano.tensor as tt
import copy

class SolveODE(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dmatrix, tt.dtensor3]
    
    __props__ = ('_paramset_id', '_solver_id', '_t0', '_tvals_id')
    
    def __init__(self, paramset, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._sens_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._paramset = copy.deepcopy(paramset)
        self._solver_id = id(solver)
        self._paramset_id = id(self._paramset)
        self._tvals_id = id(self._tvals)

    def perform(self, node, inputs, outputs):
        params, = inputs
        
        self._paramset.set_changeable(params)
        # TODO transform
        self._solver.user_data.changeable_params[:] = self._paramset.changeable_array()
        self._solver.user_data.fixed_params[:] = self._paramset.fixed_array()
        
        y0, sens0 = params_model_qu.generate_initial_values(  # TODO
            paramset, **model_args, **parametrization_args, return_sens0=True
        )
        
        self._solver.solve(self._t0, self._tvals, y0, self._y_out,
                           sens0=sens0, sens_out=self._sens_out)
        outputs[0][0] = self._y_out
        outputs[1][0] = self._sens_out

    def grad(self, inputs, g):
        g, g_grad = g
        
        assert str(g_grad) == '<DisconnectedType>'
        params, = inputs
        solution, sens = self(params)
        return [tt.sum(g[:, None, :] * sens, (0, -1))]


class SolveODEAdjoint(tt.Op):
    itypes = [tt.dvector, tt.dvector]
    otypes = [tt.dmatrix]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)

    def perform(self, node, inputs, outputs):
        y0, params = inputs

        self._solver.user_data.changeable_params[:] = params

        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
        except SolverError:
            self._y_out[:] = np.nan

        outputs[0][0] = self._y_out.copy()

    def grad(self, inputs, g):
        g, = g

        y0, params = inputs
        backward = SolveODEAdjointBackward(self._solver, self._t0, self._tvals)
        lamda, gradient = backward(y0, params, g)
        if self._t0 == self._tvals[0]:
            lamda = lamda - g[0]
        #return [-lamda + g[0], gradient]
        return [-lamda, gradient]


class SolveODEAdjointBackward(tt.Op):
    itypes = [tt.dvector, tt.dvector, tt.dmatrix]
    otypes = [tt.dvector, tt.dvector]

    __props__ = ('_solver_id', '_t0', '_tvals_id')

    def __init__(self, solver, t0, tvals):
        self._solver = solver
        self._t0 = t0
        self._y_out, self._grad_out, self._lamda_out = solver.make_output_buffers(tvals)
        self._tvals = tvals
        self._solver_id = id(solver)
        self._tvals_id = id(self._tvals)

    def perform(self, node, inputs, outputs):
        y0, params, grads = inputs

        self._solver.user_data.changeable_params[:] = params

        # TODO We don't really need to do the forward solve if we make sure
        # that it was executed previously, but it isn't very expensive
        # compared with the backward pass anyway.
        try:
            self._solver.solve_forward(self._t0, self._tvals, y0, self._y_out)
            self._solver.solve_backward(self._tvals[-1], self._t0, self._tvals,
                                        grads, self._grad_out, self._lamda_out)
        except SolverError:
            self._lamda_out[:] = np.nan
            self._grad_out[:] = np.nan

        outputs[0][0] = self._lamda_out.copy()
        outputs[1][0] = self._grad_out.copy()
