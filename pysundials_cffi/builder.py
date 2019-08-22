import logging
import pydoc
from typing import Any, Type, Dict, cast, Set, Iterable, TypeVar
from typing_extensions import Protocol, Literal

from pysundials_cffi.problem import OdeProblem


logger = logging.getLogger("pysundials_cffi.builder")


T = TypeVar("T", bound="Builder", contravariant=True)


class BuilderOption(Protocol[T]):
    def __init__(self) -> None:
        ...

    def build(self) -> None:
        ...

    @property
    def name(self) -> str:
        ...

    def take_builder(self, builder: T) -> None:
        ...

    def release_builder(self) -> None:
        ...


# TODO Sort out those type vars
T2 = TypeVar("T2", bound="Builder", covariant=True)


class Builder:
    _all_options: Dict[str, Type[BuilderOption[Any]]] = {}

    @classmethod
    def _option(
        cls: Type[T], option_class: Type[BuilderOption[T]]
    ) -> Type[BuilderOption[T]]:
        name = option_class.__name__
        cls._all_options[name] = option_class
        return option_class

    def __init__(self: T) -> None:
        self._options: Dict[str, BuilderOption[T]] = {}
        self._required: Set[str] = set()
        self._recent: Set[str] = set()

    def _update_docstring(self) -> None:
        self.__doc__ = self._make_docstring(subset="all")

    def _add(self: T2, options: Iterable[BuilderOption[T2]]) -> T2:
        for option in options:
            if option.name in self._options:
                raise ValueError("Option with name %s exists." % option.name)
            self._options[option.name] = option
        self._recent = set(option.name for option in options)
        self._update_docstring()
        return self

    def _make_required(self: T2, names: Iterable[str]) -> T2:
        for name in names:
            if not name in self._options:
                raise KeyError("Unknown option: %s" % name)
            self._required.add(name)
        self._update_docstring()
        return self

    def _remove(self: T2, names: Iterable[str]) -> T2:
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

    def __getattr__(self: T2, name: str) -> BuilderOption[T2]:
        option = self._options[name]

        def call(*args: Any, **kwargs: Any) -> T2:
            option.take_builder(self)
            try:
                builder = option(*args, **kwargs)
            finally:
                option.release_builder()
            return cast(T2, builder)._remove([option.name])

        return cast(BuilderOption[T2], call)
