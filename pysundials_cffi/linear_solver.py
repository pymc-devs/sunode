#import sys
import types
import pydoc
import logging

#import numpy as np
#from scipy import sparse
#import numba
#import numba.cffi_support

from pysundials_cffi import _cvodes
from pysundials_cffi.basic import data_dtype, index_dtype, notnull
from pysundials_cffi import basic

__all__ = ['make_linear_solver']

logger = logging.getLogger('pysundials_cffi.linear_solver')

lib = _cvodes.lib
ffi = _cvodes.ffi


LINEAR_SOLVERS = {
    'direct': lib.SUNLINEARSOLVER_DIRECT,
}


def make_linear_solver(kind='dense'):
    pass

def bind(obj, func):
    method = types.MethodType(func, obj)
    setattr(obj, func.__name__, method)


class Builder:
    def __init__(self, finalize, required=None, optional=None):
        if required is None:
            required = []
        if optional is None:
            optional = []
        self._required = required
        self._optional = optional
        self._finalize = finalize

        for func in self._required:
            bind(self, func)
        for func in self._optional:
            bind(self, func)
        self.__doc__ = self._make_docstring()

    def finalize(self):
        if self._required:
            raise ValueError(
                'Not all required methods were called. Missing %s'
                % [f.__name__ for f in self._required])
        return self._finalize(self)

    def options(self):
        print(self.__doc__)

    def _make_docstring(self):
        sections = []
        if self._required:
            sec = '\n'.join(pydoc.plaintext.document(func) for func in self._required)
            sec = pydoc.plaintext.section('Required methods', sec)
            sections.append(sec)
        if self._optional:
            sec = '\n'.join(pydoc.plaintext.document(func) for func in self._optional)
            sec = pydoc.plaintext.section('Optional methods', sec)
            sections.append(sec)
        return '\n'.join(sections)

    def _modify(self, remove=None, required=None, optional=None):
        if remove is not None:
            for name in remove:
                required_names = [f.__name__ for f in self._required]
                optional_names = [f.__name__ for f in self._optional]
                if name in required_names:
                    self._required.pop(required_names.index(name))
                elif name in optional_names:
                    self._optional.pop(optional_names.index(name))
                else:
                    raise ValueError('Unknown function %s' % name)
                delattr(self, name)

        if required is not None:
            for func in required:
                bind(self, func)
            req = required.copy()
            req.extend(self._required)
            self._required = req

        if optional is not None:
            for func in optional:
                bind(self, func)
            opt = optional.copy()
            opt.extend(self._optional)
            self._optional = opt

        self.__doc__ = self._make_docstring()
        return self


def make_direct_solver_function(builder, y, kind):
    if not hasattr(builder, '_jac_template'):
        builder = builder.jacobian('dense')
    jac = builder._jac_template
    if not isinstance(jac, basic.DenseMatrix):
        raise ValueError('Dense solver needs a dense jacobian.')

    def make_solver():
        if kind == 'dense':
            ptr = lib.SUNLinSol_Dense(y.c_ptr, jac.c_ptr)
        elif kind == 'lapack-dense':
            ptr = lib.SUNLinSol_LapackDense(y.c_ptr, jac.c_ptr)
        else:
            raise NotImplementedError()

        if ptr == ffi.NULL:
            raise ValueError('Could not create linear solver. Matrix and '
                             'vector backends are probabibly incompatible.')

        solver = LinearSolver(ptr)
        solver.borrow(y)
        solver.borrow(jac)
        return solver

    return make_solver


def make_sparse_solver_function(builder, y, kind):
    if not hasattr(builder, '_jac_template'):
        raise ValueError('Sparse solvers need a sparse jacobian, but '
                         'that has not been configured.')
    jac = builder._jac_template
    if not isinstance(jac, basic.SparseMatrix):
        raise ValueError('Sparse solvers need a sparse jacobian.')

    if kind == 'klu':
        def make_solver():
            ptr = lib.SUNLinSol_KLU(y.c_ptr, jac.c_ptr)
            solver = LinearSolver(ptr)
            solver.borrow(y)
            solver.borrow(jac)
            return solver

        return make_solver
    else:
        num_threads = 1
        superlu_ordering = 3

        def superlu_threads(builder, num):
            """Choose the number of threads that SuperLU will use."""
            nonlocal num_threads
            num_threads = num
            return builder._modify(['superlu_threads'])

        def superlu_ordering(builder, ordering):
            """Choose which ordering method SuperLU should use.
            
            Options are:
            - 'natural' for a natural ordering.
            - 'AtA' for a minimal degree ordering on :math:`A^TA`
            - 'At+A' for a minimal degree ordering on :math:`A^T + A`
            - 'colamd' (default) for a COLAMD ordering for unsymmetric matrices.
            """
            num = {
                'natural': 0,
                'AtA': 1,
                'At+A': 2,
                'colamd': 3
            }
            nonlocal superlu_ordering
            superlu_ordering = num[ordering]
            return builder._modify(['superlu_ordering'])

        builder._modify(optional=[superlu_threads, superlu_ordering])
        
        def make_solver():
            ptr = lib.SUNLinSol_SuperLUMT(y, jac, num_threads)
            notnull(ptr, 'Could not create linear solver superlu.')
            ret = SUNLinSol_SuperLUMTSetOrdering(ptr, ordering)
            if ret != lib.SUNLS_SUCCESS:
                raise ValueError('Could not set ordering.')
            solver = LinearSolver(ptr)
            solver.borrow(y)
            solver.borrow(jac)
            return solver

        return make_solver


def linear_solver(builder, kind):
    """Choose the linear solver.
    
    `kind` must be one of 'dense' (the default) and 'lapack'.
    """
    # provides builder._make_linear_solver

    if not hasattr(builder, '_y_template'):
        builder = builder.vector_backend('serial')
    y = builder._y_template

    if kind in ['dense', 'band', 'lapack-dense', 'lapack-band']:
        builder._make_linear_solver = make_direct_solver_function(builder, y, kind)
    elif kind in ['klu', 'superlu']:
        builder._make_linear_solver = make_sparse_solver_function(builder, y, kind)
    else:
        raise NotImplementedError()

    return builder._modify(['linear_solver'])


class Borrows:
    def __init__(self):
        self._borrowed = []

    def borrow(self, arg):
        self._borrowed.append(arg)

    def release_borrowed(self):
        self._borrowed = []


class LinearSolver(Borrows):
    def __init__(self, c_ptr):
        super().__init__()
        notnull(c_ptr, 'Linear solver cpointer is NULL.')
        self.c_ptr = c_ptr

    def initialize(self):
        ret = lib.SUNLinSolInitialize(self.c_ptr)
        if ret != 0:
            raise ValueError('Could not initialize linear solver.')

    def __del__(self):
        logger.debug('Freeing LinearSolver')
        lib.SUNLinSolFree(self.c_ptr)
        self.release_borrowed()
