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
