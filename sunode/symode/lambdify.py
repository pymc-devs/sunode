from itertools import count, product
import ast
import textwrap
import importlib
import sys
import inspect
from functools import partial

from sympy.printing.pycode import SciPyPrinter
import sympy
import numpy as np


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
    def __init__(self, glob=None, locale=None, printer=None):
        self._global = glob
        self._locale = locale
        self._body = []
        if printer is None:
            printer = SciPyPrinter({
                'fully_qualified_modules': True,
                'inline': True,
                'allow_unknown_functions': True,
                'user_functions': {}
            })
        self._printer = printer
    
    def add_imports(self):
        imports = ast.parse(textwrap.dedent(
            """
            import numba
            import numpy
            import scipy
            
            @numba.njit(fastmath=True)
            def logaddexp(a, b):
                min_val = fmin(a, b)
                max_val = fmax(a, b)
                return max_val + log1p(exp(min_val - max_val))

            @numba.njit(fastmath=True)
            def CardinalBSpline(degree, t):
                if degree != 4:
                    return numpy.nan
                return ((((1/24)*t**4) if (t >= 0) and (t <= 1) else (t*(t*(t*(5/6 - 1/6*t) - 5/4) + 5/6) - 5/24) if (t >= 1) and (t <= 2) else (t*(t*(t*((1/4)*t - 5/2) + 35/4) - 25/2) + 155/24) if (t >= 2) and (t <= 3) else (t*(t*(t*(5/2 - 1/6*t) - 55/4) + 65/2) - 655/24) if (t >= 3) and (t <= 4) else (t*(t*(t*((1/24)*t - 5/6) + 25/4) - 125/6) + 625/24) if (t >= 4) and (t <= 5) else (0)))
            """
        ))
        self._body.extend(imports.body)
        
    def add_var_function(self, outname, argnames, varmap, assigns, expr):
        if outname in argnames:
            raise ValueError('Invalid variable name: %s' % outname)

        func = ast.parse(textwrap.dedent(
            """
            @numba.njit(fastmath=True, error_model='numpy')
            def compute(out):
                pass
            """
        )).body[0]
        func.args.args[0] = ast.arg(arg=outname, annotation=None)
        func.args.args.extend(
            ast.arg(arg=str(name), annotation=None)
            for name in argnames
        )

        body = []
        func.body = body

        # Write zeros to output
        body.append(ast.Assign(
            targets=[ast.Subscript(
                value=ast.Name(id=outname, ctx=ast.Load()),
                slice=ast.Slice(lower=None, upper=None, step=None),
                ctx=ast.Store(),
            )],
            value=ast.Num(n=0)
        ))

        # Prepare local variables from cse
        for name, value in assigns:
            if name in varmap or name in argnames or name == outname:
                raise ValueError('Invalid variable name: %s' % name)
            body.append(self._as_assign(value, name))

        # Write result in output
        for idxs in product(*[range(n) for n in expr.shape]):
            value = expr[idxs]
            if value == 0:
                continue
            body.append(self._as_assign(value, outname, idxs))

        path_as_ast = self._path_as_ast
        class Trafo(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id in varmap and isinstance(node.ctx, ast.Load):
                    return path_as_ast(*varmap[node.id])
                else:
                    return node

        func = Trafo().visit(func)

        self._body.append(func)

    def _sympy_as_ast(self, expr):
        module = ast.parse(self._printer.doprint(expr))
        return module.body[0].value

    def _path_as_ast(self, varname, *path, as_store=False):
        current = ast.Name(id=varname, ctx=ast.Load())
        for var in path:
            if isinstance(var, str):
                outer = ast.Attribute(value=current, attr=var, ctx=ast.Load())
            elif isinstance(var, tuple):
                tup = ast.Tuple(elts=[ast.Num(n=int(i)) for i in var], ctx=ast.Load())
                outer = ast.Subscript(value=current, slice=ast.Index(value=tup), ctx=ast.Load())
            else:
                raise ValueError('Invalid path item: %s' % var)
            current = outer
        if as_store:
            current.ctx = ast.Store()
        return current

    def _as_assign(self, expr, leftname, *leftpath):
        value = self._sympy_as_ast(expr)
        return ast.Assign(
            targets=[self._path_as_ast(leftname, *leftpath, as_store=True)],
            value=value,
        )

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

    
def lambdify_consts(module_name, argnames, expr, varmap, debug=False):
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
    cse_names = (sympy.Symbol(f'CSE_DUMMY__{i}_') for i in count())
    assigns, final = sympy.cse(list(expr.ravel()), cse_names)
    expr = np.array(final).reshape(expr.shape)
    assigns = [(var.name, e) for (var, e) in assigns]

    lam = LambdifyAST()
    lam.add_imports()
    lam.add_var_function('_out', argnames, varmap, assigns, expr)
    mod = lam.as_module()
    if debug:
        print(lam.as_string())
    
    loader = AstLoader({module_name: mod})
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module.compute

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


class CardinalBSpline(sy.Function):
    nargs = 2

    def as_sympy_expr(self):
        degree, x = self.args
        #knots = [sy.Integer(i) for i in range(n_knots)]
        knots = [sy.Integer(i) for i in range(degree + 2)]
        basis = sy.functions.special.bsplines.bspline_basis(degree, tuple(knots), 0, x)
        args = basis.args
        args_horner = []
        for expr, cond in args:
            args_horner.append((sym.horner(expr), cond))
        return sy.Piecewise(*args_horner)


def interpolate_spline(x, vals, lower, upper, degree, as_pure=False):
    n_vals = len(vals)
    n_knots = degree + n_vals + 1
    basis = partial(CardinalBSpline, degree)
    x = (x - lower) / (upper - lower)
    x = degree + x * (n_knots - 2 * (degree) - 1)
    basis_vecs = [basis(x - i) for i in range(len(vals))]
    if as_pure:
        basis_vecs = [b.as_sympy_expr() for b in basis_vecs]
    return sum(val * b for val, b in zip(vals, basis_vecs))
    

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
