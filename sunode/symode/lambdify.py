import sympy
from itertools import count, product
import ast
import textwrap
import importlib
import sys
import inspect

def split_constants(expr, variables, names):
    if (not isinstance(expr, sympy.Expr)
            or isinstance(expr, sympy.Symbol)
            or not expr.free_symbols):
        return [], expr

    consts = []
    if not any(x in expr.free_symbols for x in variables):
        const = next(names)
        consts.append((const, expr))
        return consts, const

    args = []
    for arg in expr.args:
        sub_consts, sub_expr = split_constants(arg, variables, names)
        consts.extend(sub_consts)
        args.append(sub_expr)
    return consts, expr.func(*args)


def cse_const(expr, args):
    var_exprs = []
    const_exprs = []
    variables = list(args)

    const_names = (sympy.Symbol(f'CONSTANT_DUMMY__{i}_') for i in count())
    cse_names = (sympy.Symbol(f'CSE_DUMMY__{i}_') for i in count())
    
    pre, (final,) = sympy.cse(expr, cse_names)
    
    for sym, expr in pre:
        consts, mod_expr = split_constants(expr, variables, const_names)
        const_exprs.extend(consts)
        free = mod_expr.free_symbols
        if any(x in free for x in variables):
            var_exprs.append((sym, mod_expr))
            variables.append(sym)
        else:
            const_exprs.append((sym, mod_expr))
    consts, final = split_constants(final, variables, const_names)
    const_exprs.extend(consts)
    return const_exprs, var_exprs, final


class LambdifyAST:
    def __init__(self, glob=None, locale=None):
        self._global = glob
        self._locale = locale
        self._body = []
    
    def add_imports(self):
        imports = ast.parse(textwrap.dedent(
            """
            from collections import namedtuple
            import numba
            from numpy import *
            
            @numba.njit(fastmath=True)
            def logaddexp(a, b):
                min_val = fmin(a, b)
                max_val = fmax(a, b)
                return max_val + log1p(exp(min_val - max_val))
            """
        ))
        self._body.extend(imports.body)
        
    def add_const_namedtuple(self, const_vars, vars):
        call = ast.Call(
            func=ast.Name(id='namedtuple', ctx=ast.Load()),
            args=[
                ast.Str(s='Constants'),
                ast.List(
                    elts=[ast.Str(s=str(name)) for name in vars]
                         + [ast.Str(s=str(name)) for name in const_vars],
                    ctx=ast.Load(),
                )
            ],
            keywords=[],
        )
        assign = ast.Assign(
            targets=[ast.Name(id='Constants', ctx=ast.Store())],
            value=call
        )
        self._body.append(assign)
    
    def add_const_function(self, const_vars, varlist):
        func = ast.parse(textwrap.dedent(
            """
            @numba.njit(fastmath=True)
            def precompute():
                pass
            """
        )).body[0]
        func.args.args = [ast.arg(arg=str(name), annotation=None)
                          for name in const_vars]
        body = self._varlist_as_assigns(varlist)
        retval = ast.Return(
            value=ast.Call(
                func=ast.Name(id='Constants', ctx=ast.Load()),
                args=[ast.Name(id=str(sym), ctx=ast.Load())
                      for sym, _ in varlist]
                     + [ast.Name(id=str(sym), ctx=ast.Load())
                        for sym in const_vars],
                keywords=[],
            )
        )
        body.append(retval)
        func.body = body
        self._body.append(func)
    
    def add_var_function(self, const_vars, const_args, final_vars, varlist, retval):
        func = ast.parse(textwrap.dedent(
            """
            @numba.njit(fastmath=True)
            def compute(out, constants):
                pass
            """
        )).body[0]
        func.args.args.extend(
            ast.arg(arg=str(name), annotation=None)
            for name in final_vars
        )

        func.body = [
            ast.Assign(
                targets=[ast.Name(id=str(name), ctx=ast.Store())],
                value=ast.Attribute(
                    value=ast.Name(id='constants', ctx=ast.Load()),
                    attr=str(name),
                    ctx=ast.Load(),
                )
            )
            for name in list(const_vars) + list(const_args)
        ]

        func.body.extend(self._varlist_as_assigns(varlist))

        zero_out = ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id='out', ctx=ast.Load()),
                slice=ast.Slice(lower=None, upper=None, step=None),
                ctx=ast.Store()
            )],
            value=ast.Num(n=0)
        )
        func.body.append(zero_out)

        retval = sympy.Array(retval)
        shape = retval.shape
        assigns = []
        for idxs in product(*[range(n) for n in shape]):
            value = retval[idxs]
            if value == 0:
                continue
            assign = ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id='out', ctx=ast.Load()),
                        slice=ast.Index(
                            value=ast.Tuple(
                                elts=[ast.Num(n=i) for i in idxs],
                                ctx=ast.Load()
                            )
                        ),
                        ctx=ast.Store(),
                    )
                ],
                value=self._sympy_as_expr(value),
            )
            assigns.append(assign)
        func.body.extend(assigns)
        self._body.append(func)
    
    def _sympy_as_expr(self, expr):
        args = list(expr.free_symbols)
        func = sympy.lambdify(args, expr, 'numpy')
        source = inspect.getsource(func)
        module = ast.parse(source)
        ret = module.body[0].body[-1]
        return ret.value
    
    def _varlist_as_assigns(self, varlist):
        assigns = []
        for symb, val in varlist:
            value_ast = self._sympy_as_expr(val)
            assign = ast.Assign(
                targets=[ast.Name(id=str(symb), ctx=ast.Store())],
                value=value_ast,
            )
            assigns.append(assign)
        return assigns
    
    def as_module(self):
        mod = ast.Module(body=self._body)
        return ast.fix_missing_locations(mod)
    
    def as_string(self):
        import astor
        return astor.to_source(self.as_module())

    
class AstLoader(importlib.abc.InspectLoader):
    def __init__(self, asts):
        self._asts = asts

    def get_source(self, fullname):
        if fullname not in self._asts:            
            raise ImportError()
        return None
    
    def get_code(self, fullname):
        if fullname not in self._asts:
            raise ImportError()
        return self.source_to_code(self._asts[fullname])

    
def lambdify_consts(module_name, const_args, var_args, expr, debug=False):
    """Compile a sympy expression using numba.
    
    Split the expression into two functions: one, that precomputes
    partial results that only depend on `const_args`, and a second
    function that does the final computation and depends on `var_args`
    and the result of the first function.
    
    Parameters
    ----------
    module_name : str
        A unique name for the module that is generated.
    const_args : list of sympy.Symbol
        A list of symbols in `expr` that is used as input for
        the first returned function in that order.
    var_args : list of sympy.Symbol
        List of the remaining symbols in `expr`. Also order
        of arguments for the second returned function.
    expr : sympy.Expr
        A sympy expression that should be evaluated.
    debug : bool
        If true, print the generated python code. This requires
        `astor` to be installed.
    
    Returns
    -------
    precompute : function
        A function that precomputes partial expressions of `expr`.
        The arguments are the values for `const_args`.
    compute : function
        `compute` expects the output of `precompute` as first
        argument, and the values for `var_vargs` after that.
        It returns the final value of the expression.
    
    Examples
    --------
    >>> x1, x2, c1, c2 = sympy.symbols('x1, x2, c1, c2', real=True)
    >>> z = (sympy.exp(-sympy.sin(x1 + x2)
    >>>                * sympy.exp(c1 ** 2 + c1 ** 3 + c2 ** 4))
    >>>      + sympy.sin(x1 + x2)
    >>>      + sympy.exp(c1**2 + c2 ** 4)
    >>>      + x1 + c2)
    >>> precompute, compute = lambdify_consts('ode_func', [c1, c2], [x1, x2], z)
    >>> partial = precompute(-0.2, 0.2)
    >>> compute(partial, 1., 3.)
    3.672964361117696
    >>> z.n({})
    >>> z.n(subs={c1: -0.2, c2: 0.2, x1: 1., x2: 3.})
    3.67296436111770
    """
    consts, vars, final = cse_const(expr, var_args)
    pre_vars = [sym for sym, _ in consts]

    lam = LambdifyAST()
    lam.add_imports()
    lam.add_const_namedtuple(const_args, pre_vars)
    lam.add_const_function(const_args, consts)
    lam.add_var_function(pre_vars, const_args, var_args, vars, final)
    mod = lam.as_module()
    if debug:
        print(lam.as_string())
    
    loader = AstLoader({module_name: mod})
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module.precompute, module.compute

import sympy as sy
import sympy.codegen.rewriting

class logaddexp(sympy.Function):
    nargs = 2
    
    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return sy.exp(self.args[argindex - 1]) / (sy.exp(self.args[0]) + sy.exp(self.args[1]))
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_is_real(self):
        return self.args[0].is_real and self.args[1].is_real

    def _eval_is_finite(self):
        return self.args[0].is_finite and self.args[1].is_finite


logsumexp_2terms_opt = sympy.codegen.rewriting.ReplaceOptim(
    lambda l: (isinstance(l, sy.log)
               and l.args[0].is_Add
               and len(l.args[0].args) == 2
               and all(isinstance(t, sy.exp) for t in l.args[0].args)),
    lambda l: logaddexp(l.args[0].args[0].args[0], l.args[0].args[1].args[0])
)


def is_exp_sum(expr):
    if isinstance(expr, sy.exp):
        return True
    if not isinstance(expr, sy.Add):
        return False
    return all(isinstance(e, sy.exp) for e in expr.args) and len(expr.args) == 2

# assert is_exp_sum(sy.exp(c1))
# assert is_exp_sum(sy.exp(c1) + sy.exp(c2))


def is_exp_sum_pow(expr):
    if is_exp_sum(expr):
        return True
    return isinstance(expr, sy.Pow) and is_exp_sum(expr.args[0])

# assert is_exp_sum_pow(sy.exp(c1))
# assert is_exp_sum_pow(sy.exp(c1) + sy.exp(c2))
# assert is_exp_sum_pow(1/(sy.exp(c1) + sy.exp(c2)))


def is_exp_sum_pow_mult(expr):
    if is_exp_sum_pow(expr):
        return True
    return isinstance(expr, sy.Mul) and any(is_exp_sum_pow(e) for e in expr.args)


def is_multiple_exp_sum_pow_mult(expr):
    count = sum(is_exp_sum_pow_mult(e) for e in expr.args)
    return isinstance(expr, sy.Mul) and count > 1


# assert not is_multiple_exp_sum_pow_mult(sy.exp(c1))
# assert not is_multiple_exp_sum_pow_mult(1/(sy.exp(c1) + sy.exp(c2)))
# assert is_multiple_exp_sum_pow_mult(sy.exp(c2)/(sy.exp(c1) + sy.exp(c2)))
# assert is_multiple_exp_sum_pow_mult(sy.exp(c2)/(sy.exp(c1) + sy.exp(c2)) / 2)
# assert is_multiple_exp_sum_pow_mult(sy.exp(c2)/2/(sy.exp(c1) + sy.exp(c2)) ** 2)

from sympy.assumptions import Q, ask

def simplify_multiple_exp_sum(expr, do_simplify=False, optims=None):
    if optims is None:
#         optims = sympy.codegen.rewriting.optims_c99 + (logsumexp_2terms_opt,)
        optims = (
            sympy.codegen.rewriting.log1p_opt,
            logsumexp_2terms_opt,
        )
    print('in simplify_multiple_exp_sum')
    #assert expr.is_positive or expr.is_negative
    assert ask(Q.positive(expr)) or ask(Q.negative(expr))
    #sign = 1 if expr.is_positive else -1
    sign = 1 if ask(Q.positive(expr)) else -1
    # expand log so that the resulting expression is a sum 
    # given it is a multiplication/division before
    # expand_log apparently is not aware of assumptions given by a context manager
    # Therefore, use the force optin for now.
    log_expr = sy.expand_log(sy.log(sign * expr), force=True)
    log_expr = sympy.codegen.rewriting.optimize(log_expr, optims)
    val = sign * sy.exp(log_expr, evaluate=False)
    return val


explog_opt = sympy.codegen.rewriting.ReplaceOptim(
    #lambda l: ((l.is_positive or l.is_negative)
    lambda l: ((ask(Q.positive(l)) or ask(Q.negative(l)))
               and is_multiple_exp_sum_pow_mult(l)),
    simplify_multiple_exp_sum,
)
