from __future__ import annotations

import logging
import pydoc
from dataclasses import dataclass
from typing import (
    Any,
    Type,
    Dict,
    cast,
    Set,
    Iterable,
    TypeVar,
    Optional,
    overload,
    Generic,
    Union,
)
from typing_extensions import Protocol, Literal

from sunode import solver, linear_solver
from sunode.basic import (
    DenseMatrix,
    SparseMatrix,
    Vector,
    Matrix,
    LinearSolver,
)


logger = logging.getLogger("sunode.builder")


@dataclass
class OptionData:
    vector_backend: Optional[str] = None
    jacobian: Optional[str] = None
    superlu_threads: Optional[int] = None
    klu_ordering: Optional[str] = None


@dataclass
class BuildData:
    y_template: Optional[Vector] = None
    jac_template: Optional[Matrix] = None
    linear_solver: Optional[LinearSolver] = None


T = TypeVar("T", bound="Option")


class Option:
    def __init__(self) -> None:
        self._builder: Optional[Builder] = None

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def builder(self) -> Builder:
        if self._builder is None:
            raise ValueError("Option can only be called through the Builder.")
        return self._builder

    def _take_builder(self, builder: Builder) -> None:
        self._builder = builder

    def _release_builder(self) -> None:
        self._builder = None

    @overload
    def __get__(self: T, builder: Builder, type: Any = None) -> BoundOption[T]:
        ...

    @overload
    def __get__(self: T, builder: Any, type: Any = None) -> T:
        ...

    def __get__(self: T, builder: Any, type: Any = None) -> Union[T, BoundOption[T]]:
        if isinstance(builder, Builder):
            return BoundOption(builder, self)
        return self


class BoundOption(Generic[T]):
    def __init__(self, builder: Builder, option: T):
        self._builder = builder
        self._option = option

    def __call__(self, *args: Any, **kwargs: Any) -> Builder:
        self._option._take_builder(self._builder)
        try:
            self._option(*args, **kwargs)
        finally:
            self._option._release_builder()
        return self._builder._remove([self._option.name])


class DirMeta(type):
    def __dir__(cls):
        #return list(cls._all_options) + type.__dir__(cls)
        return type.__dir__(cls)


class Builder(metaclass=DirMeta):
    _all_options: Dict[str, Type[Option]] = {}

    @classmethod
    def _option(cls: Type[Builder], option_class: Type[Option]) -> Type[Option]:
        name = option_class.__name__
        cls._all_options[name] = option_class
        return option_class

    def __init__(self) -> None:
        self._options: Dict[str, Option] = {}
        self._required: Set[str] = set()
        self._recent: Set[str] = set()

        self._add_initial_options()

    def _update_docstring(self) -> None:
        self.__doc__ = self._make_docstring(subset="all")

    def _add(self, options: Iterable[Option]) -> Builder:
        for option in options:
            if option.name in self._options:
                raise ValueError("Option with name %s exists." % option.name)
            self._options[option.name] = option
        self._recent = set(option.name for option in options)
        self._update_docstring()
        return self

    def _make_required(self, names: Iterable[str]) -> Builder:
        for name in names:
            if not name in self._options:
                raise KeyError("Unknown option: %s" % name)
            self._required.add(name)
        self._update_docstring()
        return self

    def _remove(self, names: Iterable[str]) -> Builder:
        for name in names:
            if not name in self._options:
                raise KeyError("Unknown option: %s" % name)
            del self._options[name]
            if name in self._required:
                self._required.remove(name)
        return self

    def help(self, subset: Literal["all", "possible"] = "possible") -> None:
        print(self._make_docstring(subset=subset))

    def _make_docstring(self, subset: Literal["all", "possible"] = "all") -> str:
        plaintext = getattr(cast(Any, pydoc), "plaintext")  # work around typing problem
        if subset == "all":
            methods = "\n".join(
                plaintext.document(option.__call__).replace("__call__", name)
                for name, option in type(self)._all_options.items()
            )
            return cast(str, plaintext.section("All possible options", methods))
        elif subset == "possible":
            optional = set(self._options.keys()) - self._required
            sections = []
            if self._required:
                sec = "\n".join(
                    plaintext.document(self._options[opt].__class__.__call__).replace(
                        "__call__", opt
                    )
                    for opt in self._required
                )
                sec = plaintext.section("Required options", sec)
                sections.append(sec)
            if optional:
                sec = "\n".join(
                    plaintext.document(self._options[opt].__class__.__call__).replace(
                        "__call__", opt
                    )
                    for opt in optional
                )
                sec = plaintext.section("Optional options", sec)
                sections.append(sec)
            return "\n".join(sections)
        raise ValueError(
            'Invalid subset: %s. Must be one of "all" or "possible"' % subset
        )

    def __getattr__(self, name: str) -> Option:
        return self._options[name]

    def __dir__(self) -> List[str]:
        return list(self._options) + list(super(Builder, self).__dir__())

    def _add_initial_options(self) -> None:
        self._build_data = BuildData()
        self._option_data = OptionData()
        self._add([linear_solver.klu_ordering()])

    def solve(self) -> solver.Solver:
        raise NotImplementedError()
    
    
@Builder._option
class vector_backend(Option):
    def __call__(self, kind: str) -> None:
        assert kind in ["serial"]
        self.builder._option_data.vector_backend = kind

    def build(self) -> None:
        ndim = self.builder._problem.n_states
        kind = self.builder._option_data.vector_backend
        if kind is None:
            kind = "serial"
        vector = empty_vector(ndim, kind=kind)
        self.builder._build_data.y_template = vector


@Builder._option
class jacobian(Option):
    def __call__(self, kind: str) -> None:
        """Hallo"""
        assert kind in ["dense", "sparse"]
        self.builder._option_data.jacobian = kind

    def build(self) -> None:
        ndim = self.builder._problem.n_states
        kind = self.builder._option_data.jacobian
        if kind is None:
            kind = "dense"
        matfunc = self.builder._problem.request_jac_func(kind)
        if matfunc is None:
            raise ValueError("Problem does not support jacobian")
        self.builder._opt
