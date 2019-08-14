import logging
import types
from types import MethodType
import pydoc
from typing import Callable


logger = logging.getLogger('pysundials_cffi.builder')


def bind(obj: Builder, func: Callable):
    method = MethodType(func, obj)
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

    # We tell mypy that there are dynamic methods
    def __getattr__(self, name: str) -> Callable:
        raise AttributeError()
