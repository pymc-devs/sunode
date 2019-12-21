def generate_sundials_rhs(rhs_pre, rhs_calc, user_dtype):
    N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
    N_VGetLength_Serial = lib.N_VGetLength_Serial

    user_ndtype = numba.from_dtype(user_dtype)
    user_ndtype_p = numba.types.CPointer(user_ndtype)
    func_type = numba.cffi_support.map_type(ffi.typeof('CVRhsFn'))
    func_type = func_type.return_type(*(func_type.args[:-1] + (user_ndtype_p,)))
    func_type

    @numba.cfunc(func_type)
    def rhs_wrapper(t, y_, out_, user_data_):
        y_ptr = N_VGetArrayPointer_Serial(y_)
        n_vars = N_VGetLength_Serial(y_)
        out_ptr = N_VGetArrayPointer_Serial(out_)
        y = numba.carray(y_ptr, (1, n_vars,))
        out = numba.carray(out_ptr, (1, n_vars,))
        
        user_data = numba.carray(user_data_, (1,), user_dtype)[0]
        fixed = user_data.fixed_params.reshape((1, -1))
        changeable = user_data.changeable_params.reshape((1, -1))
        
        pre = rhs_pre()
        rhs_calc(out, pre, t, y, fixed, changeable)
        return 0
    
    return rhs_wrapper


def generate_sundials_sens_rhs(sens_rhs, user_dtype):
    N_VGetArrayPointer_Serial = lib.N_VGetArrayPointer_Serial
    N_VGetLength_Serial = lib.N_VGetLength_Serial
    
    user_ndtype = numba.from_dtype(user_dtype)
    user_ndtype_p = numba.types.CPointer(user_ndtype)

    func_type = numba.cffi_support.map_type(ffi.typeof('CVSensRhsFn'))
    args = list(func_type.args)
    args[-3] = user_ndtype_p
    func_type = func_type.return_type(*args)

    @numba.cfunc(func_type)
    def sens_rhs_wrapper(n_params, t, y_, ydot_, yS_, out_, user_data_, tmp1_, tmp2_):
        n_vars = N_VGetLength_Serial(y_)
        y_ptr = N_VGetArrayPointer_Serial(y_)
        y = numba.carray(y_ptr, (1, n_vars,))

        user_data = numba.carray(user_data_, (1,), user_dtype)[0]
        fixed = user_data.fixed_params
        changeable = user_data.changeable_params
        
        tmp_sens_rhs = user_data.tmp_nparams_nvars
        tmp_sens_yS = user_data.tmp2_nparams_nvars

        for i in range(n_params):
            yS_i_ptr = N_VGetArrayPointer_Serial(yS_[i])
            yS_i = numba.carray(yS_i_ptr, (n_vars,))
            tmp_sens_yS[i, :] = yS_i

        sens_rhs(
            tmp_sens_rhs,
            t,
            y,
            tmp_sens_yS,
            fixed.reshape((1, -1)),
            changeable.reshape((1, -1))
        )

        for i in range(n_params):
            out_i_ptr = N_VGetArrayPointer_Serial(out_[i])
            out_i = numba.carray(out_i_ptr, (n_vars,))
            out_i[:] = tmp_sens_rhs[i][:]

        return 0

    return sens_rhs_wrapper



def generate_ode_callbacks_sundials(
    module_name,
    dydt,
    t,
    y,
    s,
    params,
    fixed,
    changeable,
    user_dtype,
    fastcompile=True,
    debug=False
):
    """
    Generate ode callbacks for sundials.

    Parameters
    ----------
    module_name: string
        name of the generated module
    dydt: sympy expression
        describes the ODE and returns sympy matrix
    paramset: ParamSet object
        contains all parameters for the model
    t: sympy.Symbol
        sympy representation of time
    y: sympy.MatrixSymbol
        array of state variables
    s: sympy.MatrixSymbol
        Array of sensetivities (n_params, n_vars)
    params: datalass of sympy.Symbol
        Symbolic representations of all parameters
    fixed: sympy.MatrixSymbol
        Symbolic representation of the vector of fixed parameters
    changeable: sympy.MatrixSymbol
        Symbolic representation of the vector of changeable parameters
    debug: bool
        Print the generated python code
    """
    if sunode is None:
        raise ImportError('Could not import sunode.')
    
    jac = dydt.jacobian(y)
    jacprot = (jac * s.T.as_explicit()).as_explicit()
    dsense = dydt.jacobian(changeable)
    sdiff = (jacprot + dsense).as_explicit()
    
    n_vars = int(dydt.shape[0])
    
    rhs_pre, rhs_calc = lambdify_consts(
        "_rhs",
        const_args=[],
        var_args=[t, y, fixed, changeable],
        expr=dydt.T,
        debug=debug,
    )

    jac_pre, jac_calc = lambdify_consts(
        "_jac",
        const_args=[],
        var_args=[t, y, fixed, changeable],
        expr=jac,
        debug=debug,
    )

    if fastcompile:
        dsens_pre, dsens_calc = lambdify_consts(
            "_sens",
            const_args=[],
            var_args=[t, y, fixed, changeable],
            expr=dsense.T,
            debug=debug,
        )

        @numba.njit
        def sens_rhs(out, t, y, s, fixed, changeable):
            pre = jac_pre()
            jac_out = np.empty((n_vars, n_vars))
            jac_calc(jac_out, pre, t, y, fixed, changeable)
            jacprod = np.dot(s, jac_out.T)
            pre_sens = dsens_pre()
            dsens_calc(out, pre_sens, t, y, fixed, changeable)
            out[:] += jacprod
    else:
        sens_rhs_pre, sens_rhs_calc = lambdify_consts(
            "_sens",
            const_args=[],
            var_args=[t, y, s, fixed, changeable],
            expr=sdiff.T,
            debug=debug,
        )

        @numba.njit
        def sens_rhs(out, t, y, s, fixed, changeable):
            pre = sens_rhs_pre()
            sens_rhs_calc(out, pre, t, y, s, fixed, changeable)
    
    rhs = generate_sundials_rhs(rhs_pre, rhs_calc, user_dtype)
    sens_rhs = generate_sundials_sens_rhs(sens_rhs, user_dtype)

    return rhs, sens_rhs
