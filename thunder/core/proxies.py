from __future__ import annotations

import copy
from enum import auto, Enum
from numbers import Number
from typing import Any
from collections.abc import Callable
from collections.abc import Sequence

from functools import reduce, partial
import operator
import builtins
import math

import torch

from thunder.core.compile_data import using_symbolic_values, using_jit
from thunder.core.interpreter import is_jitting, ProvenanceRecord, PseudoInst
from thunder.core.trace import (
    VariableInterface,
    get_tracectx,
    is_tracing,
    TraceCtx,
)
from thunder.core.baseutils import (
    ProxyInterface,
    NumberProxyInterface,
    TensorProxyInterface,
    TorchAutogradFunctionCtxProxyInterface,
    TagBase,
)
import thunder.core.baseutils as baseutils
from thunder.core.langctxs import resolve_method, get_langctx
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes

ShapeLike = Sequence[int]


# TODO Document this class
# Wraps a Proxy, and changes the hash
# and equality to be based the name of the proxy.
class Variable:
    def __init__(self, p: ProxyInterface):
        self.proxy = p

    def __hash__(self):
        return hash(self.proxy.name)

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.proxy.name == other.proxy.name

        return False

    def __repr__(self):
        return str(self.proxy)


def variableify(x: Any) -> Any:
    if isinstance(x, ProxyInterface):
        return Variable(x)

    return x


def unvariableify(x: Any) -> Any:
    if isinstance(x, Variable):
        return x.proxy

    return x


# Tries to register a proxy name and returns a boolean indicating success
def register_proxy_name(name: None | str = None):
    trc = get_tracectx()

    if name is not None and not trc.has_name(name):
        trc.add_name(name)
        return True

    return False


def is_proxy_name_available(name: None | str = None):
    trc = get_tracectx()

    if name is not None and not trc.has_name(name):
        return True

    return False


def make_proxy_name(*, name: None | str = None, prefix: None | str = None) -> str:
    if register_proxy_name(name):
        return name

    trc = get_tracectx()
    return trc.make_name(prefix=prefix)


class ProxyTag(TagBase):
    pass


# TODO Document this class
# TODO Support multiple histories
class Proxy(VariableInterface, ProxyInterface):
    def __init__(
        self,
        name: str | None = None,
        *,
        prefix: None | str = None,
        history: None | tuple = None,
        tags: set | None = None,
    ):
        # Determines the prefix
        if prefix is None:
            if isinstance(self, FloatProxy):
                prefix = "f"
            elif isinstance(self, ComplexProxy):
                prefix = "c"
            elif isinstance(self, IntegerProxy):
                if self.python_type is int:
                    prefix = "i"
                elif self.python_type is bool:
                    prefix = "b"
                else:
                    baseutils.check(False, lambda x=self: f"Unexpected python type for IntegerProxy {x.python_type}")
            elif isinstance(self, NumberProxy):
                prefix = "n"
            elif isinstance(self, StringProxy):
                prefix = "s"
            elif isinstance(self, TensorProxy):
                prefix = "t"
            elif isinstance(self, CollectionProxy):
                prefix = "C"
            elif isinstance(self, TupleProxy):
                prefix = "tup"
            elif isinstance(self, ListProxy):
                prefix = "lst"
            elif isinstance(self, DictProxy):
                prefix = "d"
            elif isinstance(self, TorchAutogradFunctionCtxProxy):
                prefix = "fc"
            else:
                prefix = "p"

        self._name = make_proxy_name(name=name, prefix=prefix)
        self._has_weak_name: bool = name is None
        self.history = history
        self._tags = set(tags) if tags is not None else set()

    @property
    def tags(self) -> set:
        return self._tags

    @property
    def name(self) -> str:
        return self._name

    def replace(self, **changes):
        r"""Return a copy of the Proxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``.
        Note that the copy will use the current (environment) tracectx."""
        if type(self) != Proxy:
            raise NotImplementedError(f"replace is not implemented for {type(self)}")
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return Proxy(**kwargs)

    def replace_name(self, name: str | None = None):
        """Return a copy of this proxy with the given name."""
        return self.replace(name=name)

    def __repr__(self) -> str:
        # All subclasses of Proxy will have `self.name`, so this generic implementation relies on that.
        # To have a specific repr for a subclass, override the implementation for that subclass.
        return f'<{type(self).__name__}(name="{self.name}")>'

    def type_string(self) -> str:
        return "Any"

    # Disables dunder methods

    #
    # Unary dunders
    #

    def __abs__(self):
        raise NotImplementedError(f"__abs__ is not implemented for {type(self)}")

    # See https://docs.python.org/3/reference/datamodel.html#object.__ceil__
    def __ceil__(self):
        raise NotImplementedError(f"__ceil__ is not implemented for {type(self)}")

    # See https://docs.python.org/3/reference/datamodel.html#object.__floor__
    def __floor__(self):
        raise NotImplementedError(f"__floor__ is not implemented for {type(self)}")

    def __invert__(self):
        raise NotImplementedError(f"__invert__ is not implemented for {type(self)}")

    def __neg__(self):
        raise NotImplementedError(f"__neg__ is not implemented for {type(self)}")

    def __pos__(self):
        raise NotImplementedError(f"__pos__ is not implemented for {type(self)}")

    # See https://docs.python.org/3/reference/datamodel.html#object.__round__
    def __round__(self):
        raise NotImplementedError(f"__round__ is not implemented for {type(self)}")

    # See https://docs.python.org/3/reference/datamodel.html#object.__trunc__
    def __trunc__(self):
        raise NotImplementedError(f"__trunc__ is not implemented for {type(self)}")

    #
    # Binary dunders
    #

    def __add__(self, other):
        raise NotImplementedError(f"__add__ is not implemented for {type(self)}")

    def __radd__(self, other):
        raise NotImplementedError(f"__radd__ is not implemented for {type(self)}")

    def __divmod__(self, other):
        raise NotImplementedError(f"__divmod__ is not implemented for {type(self)}")

    def __rdivmod__(self, other):
        raise NotImplementedError(f"__rdivmod__ is not implemented for {type(self)}")

    def __floordiv__(self, other):
        raise NotImplementedError(f"__floordiv__ is not implemented for {type(self)}")

    def __rfloordiv__(self, other):
        raise NotImplementedError(f"__rfloordiv__ is not implemented for {type(self)}")

    def __mod__(self, other):
        raise NotImplementedError(f"__mod__ is not implemented for {type(self)}")

    def __rmod__(self, other):
        raise NotImplementedError(f"__rmod__ is not implemented for {type(self)}")

    def __mul__(self, other):
        raise NotImplementedError(f"__mul__ is not implemented for {type(self)}")

    def __rmul__(self, other):
        raise NotImplementedError(f"__rmul__ is not implemented for {type(self)}")

    def __pow__(self, other):
        raise NotImplementedError(f"__pow__ is not implemented for {type(self)}")

    def __rpow__(self, other):
        raise NotImplementedError(f"__rpow__ is not implemented for {type(self)}")

    def __sub__(self, other):
        raise NotImplementedError(f"__sub__ is not implemented for {type(self)}")

    def __rsub__(self, other):
        raise NotImplementedError(f"__rsub__ is not implemented for {type(self)}")

    def __truediv__(self, other):
        raise NotImplementedError(f"__truediv__ is not implemented for {type(self)}")

    def __rtruediv__(self, other):
        raise NotImplementedError(f"__rtruediv__ is not implemented for {type(self)}")

    #
    # Logical operations
    #
    def __and__(self, other):
        raise NotImplementedError(f"__and__ is not implemented for {type(self)}")

    def __rand__(self, other):
        raise NotImplementedError(f"__rand__ is not implemented for {type(self)}")

    def __eq__(self, other):
        raise NotImplementedError(f"__eq__ is not implemented for {type(self)}")

    def __ge__(self, other):
        raise NotImplementedError(f"__ge__ is not implemented for {type(self)}")

    def __gt__(self, other):
        raise NotImplementedError(f"__gt__ is not implemented for {type(self)}")

    def __le__(self, other):
        raise NotImplementedError(f"__le__ is not implemented for {type(self)}")

    def __lt__(self, other):
        raise NotImplementedError(f"__lt__ is not implemented for {type(self)}")

    def __ne__(self, other):
        raise NotImplementedError(f"__ne__ is not implemented for {type(self)}")

    # NOTE This is a bitwise or triggered by the | operator
    # See https://docs.python.org/3/reference/datamodel.html#object.__or__
    def __or__(self, other):
        raise NotImplementedError(f"__or__ is not implemented for {type(self)}")

    def __ror__(self, other):
        raise NotImplementedError(f"__ror__ is not implemented for {type(self)}")

    # The ^ operator
    # See https://docs.python.org/3/reference/datamodel.html#object.__xor__
    def __xor__(self, other):
        raise NotImplementedError(f"__xor__ is not implemented for {type(self)}")

    def __rxor__(self, other):
        raise NotImplementedError(f"__rxor__ is not implemented for {type(self)}")

    #
    # Shift operations
    #

    def __lshift__(self, other):
        raise NotImplementedError(f"__lshift__ is not implemented for {type(self)}")

    def __rlshift__(self, other):
        raise NotImplementedError(f"__rlshift__ is not implemented for {type(self)}")

    def __rshift__(self, other):
        raise NotImplementedError(f"__rshift__ is not implemented for {type(self)}")

    def __rrshift__(self, other):
        raise NotImplementedError(f"__rrshift__ is not implemented for {type(self)}")

    #
    # Casts to Python numbers
    #

    def __complex__(self):
        raise NotImplementedError(f"__complex__ is not implemented for {type(self)}")

    def __float__(self):
        raise NotImplementedError(f"__float__ is not implemented for {type(self)}")

    def __int__(self):
        raise NotImplementedError(f"__int__ is not implemented for {type(self)}")

    def __bool__(self):
        raise NotImplementedError(f"__bool__ is not implemented for {type(self)}")

    #
    # Matmul operators (not implemented for numbers)
    #

    def __matmul__(self, other):
        raise NotImplementedError(f"__matmul__ is not implemented for {type(self)}")

    def __rmatmul__(self, other):
        raise NotImplementedError(f"__rmatmul__ is not implemented for {type(self)}")

    #
    # Inplace dunders
    #

    def __iadd__(self, other):
        raise RuntimeError("Inplace operators like __iadd__ are not supported.")

    def __iand__(self, other):
        raise RuntimeError("Inplace operators like __iand__ are not supported.")

    def __iconcat__(self, other):
        raise RuntimeError("Inplace operators like __iconcat__ are not supported.")

    def __ifloordiv__(self, other):
        raise RuntimeError("Inplace operators like __ifloordiv__ are not supported.")

    def __ilshift__(self, other):
        raise RuntimeError("Inplace operators like __ilshift__ are not supported.")

    def __imatmul__(self, other):
        raise RuntimeError("Inplace operators like __imatmul__ are not supported.")

    def __imod__(self, other):
        raise RuntimeError("Inplace operators like __imod__ are not supported.")

    def __imul__(self, other):
        raise RuntimeError("Inplace operators like __imul__ are not supported.")

    def __ior__(self, other):
        raise RuntimeError("Inplace operators like __ior__ are not supported.")

    def __ipow__(self, other):
        raise RuntimeError("Inplace operators like __ipow__ are not supported.")

    def __irshift__(self, other):
        raise RuntimeError("Inplace operators like __irshift__ are not supported.")

    def __isub__(self, other):
        raise RuntimeError("Inplace operators like __isub__ are not supported.")

    def __itruediv__(self, other):
        raise RuntimeError("Inplace operators like __itruediv__ are not supported.")

    def __ixor__(self, other):
        raise RuntimeError("Inplace operators like __ixor__ are not supported.")


# A generic "anything" proxy
# Unlike many other proxies, this does not mimic the type of the object it wraps
# TODO RC1 Rename ._o to ._value for consistency
class AnyProxy(Proxy):
    def __init__(self, o: Any, /, *, name: str | None = None, history: None | tuple = None, tags: set | None = None):
        super().__init__(name=name, history=history, tags=tags)
        self._o = o

    def __repr__(self) -> str:
        return f"<AnyProxy '{self._o}>'"

    def type_string(self) -> str:
        return str(type(self._o))

    def replace(self, **changes):
        r"""returns a copy replacing \**changes. Note that the copy will use the current tracectx"""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return AnyProxy(self._o, **kwargs)


class StringProxy(Proxy, str):
    def __new__(cls, s: str, /, *, name: str | None = None, history: None | tuple = None, tags: set | None = None):
        return str.__new__(cls, s)

    def __init__(self, s: str, /, *, name: str | None = None, history: None | tuple = None, tags: set | None = None):
        Proxy.__init__(self, name=name, history=history, tags=tags)
        self.value: str = s

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"<StringProxy '{self.value}'>"

    def replace(self, **changes):
        r"""Return a copy of the StringProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return StringProxy(self.value, **kwargs)

    def type_string(self) -> str:
        return "str"

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, str):
            # important: *StringProxies* are isintance str
            return False
        return str(self) == str(other)


#
# Collection proxies
#


# The following class is DEPRECATED, and is only preserved her for experimental feature development
#   that relies upon it.
class CollectionProxy(Proxy):
    def __init__(self, coll: Any, *, name: str | None = None, tags: set | None = None):
        Proxy.__init__(self, name=name, tags=tags)
        self.coll = coll

    def collection(self) -> Any:
        return self.coll

    def type_string(self) -> str:
        return "Collection"


class TupleProxy(Proxy, tuple):
    def __new__(cls, tup: tuple, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        return tuple.__new__(cls, tup)

    def __init__(self, tup: tuple, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        Proxy.__init__(self, name=name, history=history, tags=tags)
        self._value = tup

    def type_string(self) -> str:
        return "tuple"

    def replace(self, **changes):
        r"""Return a copy of the TupleProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return TupleProxy(self._value, **kwargs)

    def __add__(self, other):
        if not isinstance(other, tuple):
            raise TypeError(f"can only concatenate tuple (not '{type(other)}') to tuple")

        return self._value + other

    def __bool__(self, /) -> bool:
        return bool(self._value)

    def __eq__(self, other, /) -> bool:
        return self._value == other

    def __setitem__(self, *args):
        raise TypeError("'tuple' object does not support item assignment")


class ListProxy(Proxy, list):
    def __new__(cls, lst: list, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        l = list.__new__(cls, lst)

        # NOTE This intentionally does not call the ListProxy.extend() method
        list.extend(l, lst)
        return l

    def __init__(self, lst: list, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        Proxy.__init__(self, name=name, history=history, tags=tags)
        self._value = lst

    def type_string(self, /) -> str:
        return "list"

    def replace(self, **changes):
        r"""Return a copy of the ListProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return ListProxy(self._value, **kwargs)

    def __add__(self, other, /):
        if not isinstance(other, list):
            raise TypeError(f"can only concatenate list (not '{type(other)}') to list")

        return self._value + other

    def __bool__(self, /) -> bool:
        return bool(self._value)

    def __eq__(self, other, /) -> bool:
        return self._value == other

    def __setitem__(self, *args):
        raise NotImplementedError("Assigning to elements of an input list is not yet supported")

    def append(self, arg, /):
        raise NotImplementedError("Appending to an input list is not yet supported")

    def clear(self, /):
        raise NotImplementedError("Clearing an input list is not yet supported")

    def extend(self, arg, /):
        raise NotImplementedError("Extending an input list is not yet supported")

    def insert(self, *args):
        raise NotImplementedError("Inserting into an input list is not yet supported")

    def pop(self):
        raise NotImplementedError("Popping from an input list is not yet supported")

    def remove(self, arg, /):
        raise NotImplementedError("Removing from an input list is not yet supported")


class DictProxy(Proxy, dict):
    def __new__(cls, d: dict, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        nd = dict.__new__(cls, d)
        dict.update(nd, d)
        return nd

    def __init__(self, d: dict, *, name: None | str = None, history: None | tuple = None, tags: set | None = None):
        Proxy.__init__(self, name=name, history=history, tags=tags)
        self._value = d

    def type_string(self, /) -> str:
        return "dict"

    def replace(self, **changes):
        r"""Return a copy of the DictProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(changes)
        return DictProxy(self._value, **kwargs)

    def __add__(self, other, /):
        return self._value + other

    def __bool__(self, /) -> bool:
        return bool(self._value)

    def __delitem__(self, key: Any):
        raise NotImplementedError("Deleting keys of an input dict is not yet supported")

    def __eq__(self, other, /) -> bool:
        return self._value == other

    def __ior__(self, other, /):
        raise NotImplementedError("Modifying an input dict inplace is not yet supported")

    def __or__(self, other, /) -> bool:
        return self._value | other

    def __setitem__(self, *args):
        raise NotImplementedError("Assigning to keys of an input dict is not yet supported")

    def clear(self, /):
        raise NotImplementedError("Clearning an input dict is not yet supported")

    def items(self, /):
        return self._value.items()

    def pop(self, key, /):
        raise NotImplementedError("Popping an input dict is not yet supported")

    def popitem(self, /):
        raise NotImplementedError("Popping an input dict is not yet supported")

    def update(self, other, /):
        raise NotImplementedError("Updating an input dict is not yet supported")

    def setdefault(self, *args):
        raise NotImplementedError("Calling setdefault on an input dict is not yet supported")


# CONSTRAINT annotates NumberProxy as their get processed by interpreter.
# A NumberProxy can be at one of the status:
# - A DYNAMIC NumberProxy cannot be converted to a static number;
# - A CONSTRAINABLE NumberProxy is treated as DYNAMIC by default, but it could be converted to STATIC by interpreter;
# - A STATIC NumberProxy can be treated as a static number, but not necessarily so;
#   The protocol here is that, if a NumberProxy instance is converted to static, we'll insert a guard logic in prologue trace to ensure the NumberProxy doesn't change at runtime.
class CONSTRAINT(Enum):
    DYNAMIC = auto()
    CONSTRAINABLE = auto()
    STATIC = auto()


# NOTE NumberProxies are NOT Numbers
# TODO Maybe NumberProxies should be Numbers?
class NumberProxy(Proxy, NumberProxyInterface):
    def __init__(
        self,
        name: str | None = None,
        value: Number | None = None,
        *,
        python_type: type,
        history: None | tuple = None,
        constraint: None | CONSTRAINT = None,
        tags: set | None = None,
    ):
        self.value = value
        self.python_type = python_type
        if constraint is None:
            constraint = CONSTRAINT.DYNAMIC
        self.constraint = constraint

        Proxy.__init__(self, name, history=history, tags=tags)

    # NOTE: Python numbers hash to themselves, and this mimics that behavior
    def __hash__(self) -> int:
        return hash(self.value)

    def replace(self, **changes):
        r"""Return a copy of the NumberProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``value``, ``python_type``, ``constraint``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
            value=self.value,
            python_type=self.python_type,
            constraint=self.constraint,
            __class__=self.__class__,  # undocumented
        )
        kwargs.update(changes)
        cls = kwargs.pop("__class__")
        return cls(**kwargs)

    def known_value(self) -> bool:
        return self.value is not None

    def make_static_constrained(self):
        baseutils.check(self.constraint != CONSTRAINT.DYNAMIC, lambda: f"dynamic NumberProxy cannot be made static")
        baseutils.check(self.value is not None, lambda: f"static NumberProxy needs to have value")
        self.constraint = CONSTRAINT.STATIC

    def make_constrainable(self):
        self.constraint = CONSTRAINT.CONSTRAINABLE

    def is_static_constrained(self) -> bool:
        return self.constraint == CONSTRAINT.STATIC

    def is_dynamic(self) -> bool:
        return self.constraint == CONSTRAINT.DYNAMIC

    #
    # Elementwise unary operators
    #

    # name is the name of the operation in the number language context to perform
    # fn is the function to call if executing outside a language context
    @staticmethod
    def _elementwise_unary_helper(a, name, fn, type_promotion_kind=None):

        vala = pyval(a)

        trace: None | TraceCtx = get_tracectx()
        lang: None | LangCtx = None
        try:
            lang = get_langctx()
        except LookupError:
            pass
        if trace is None or lang is None:
            # Outside of a trace or language context, operations on NumberProxies are
            #   executed by the Python interpreter
            baseutils.check(
                vala is not None,
                lambda: f"Trying to {name} a number with an unknown value",
                exception_type=AssertionError,
            )
            return fn(vala)

        method = resolve_method(name, a)
        return method(a)

    def __abs__(self):
        return self._elementwise_unary_helper(self, "abs", builtins.abs)

    # See https://docs.python.org/3/reference/datamodel.html#object.__ceil__
    def __ceil__(self):
        return self._elementwise_unary_helper(self, "ceil", math.ceil)

    # See https://docs.python.org/3/reference/datamodel.html#object.__floor__
    def __floor__(self):
        return self._elementwise_unary_helper(self, "floor", math.floor)

    def __invert__(self):
        return self._elementwise_unary_helper(self, "bitwise_not", operator.inv)

    def __neg__(self):
        return self._elementwise_unary_helper(self, "neg", operator.neg)

    def __pos__(self):
        return self._elementwise_unary_helper(self, "pos", operator.pos)

    # See https://docs.python.org/3/reference/datamodel.html#object.__round__
    def __round__(self):
        return self._elementwise_unary_helper(self, "round", builtins.round)

    # See https://docs.python.org/3/reference/datamodel.html#object.__trunc__
    def __trunc__(self):
        return self._elementwise_unary_helper(self, "trunc", math.trunc)

    #
    # Elementwise binary operators
    #

    @staticmethod
    def _elementwise_binary_helper(a, b, name, fn, type_promotion_kind=None):
        baseutils.check_type(b, (Number, NumberProxy, TensorProxy))

        vala = pyval(a)
        valb = pyval(b) if isinstance(b, NumberProxy) else b

        trace: None | TraceCtx = get_tracectx()
        lang: None | LangCtx = None
        try:
            lang = get_langctx()
        except LookupError:
            pass
        if trace is None or lang is None:
            # Outside of a trace or language context, binary operations on NumberProxies are
            #   executed by the Python interpreter
            baseutils.check(
                vala is not None and valb is not None,
                lambda: f"Trying to {name} numbers with unknown values",
                exception_type=AssertionError,
            )
            return fn(vala, valb)

        if is_jitting():
            fn: Callable = resolve_method(name, a, b)
            return fn(a, b)

        method = resolve_method(name, a, b)
        return method(a, b)

    def __add__(self, other):
        return self._elementwise_binary_helper(self, other, "add", operator.add)

    def __radd__(self, other):
        return self._elementwise_binary_helper(other, self, "add", operator.add)

    def __divmod__(self, other):
        return self._elementwise_binary_helper(self, other, "divmod", builtins.divmod)

    def __rdivmod__(self, other):
        return self._elementwise_binary_helper(other, self, "divmod", builtins.divmod)

    def __floordiv__(self, other):
        return self._elementwise_binary_helper(self, other, "floor_divide", operator.floordiv)

    def __rfloordiv__(self, other):
        return self._elementwise_binary_helper(other, self, "floor_divide", operator.floordiv)

    def __mod__(self, other):
        return self._elementwise_binary_helper(self, other, "mod", operator.mod)

    def __rmod__(self, other):
        return self._elementwise_binary_helper(other, self, "mod", operator.mod)

    def __mul__(self, other):
        return self._elementwise_binary_helper(self, other, "mul", operator.mul)

    def __rmul__(self, other):
        return self._elementwise_binary_helper(other, self, "mul", operator.mul)

    def __pow__(self, other):
        return self._elementwise_binary_helper(self, other, "pow", operator.pow)

    def __rpow__(self, other):
        return self._elementwise_binary_helper(other, self, "pow", operator.pow)

    def __sub__(self, other):
        return self._elementwise_binary_helper(self, other, "sub", operator.sub)

    def __rsub__(self, other):
        return self._elementwise_binary_helper(other, self, "sub", operator.sub)

    def __truediv__(self, other):
        return self._elementwise_binary_helper(self, other, "true_divide", operator.truediv)

    def __rtruediv__(self, other):
        return self._elementwise_binary_helper(other, self, "true_divide", operator.truediv)

    #
    # Logical operations
    #

    # NOTE This is a bitwise and and triggered by the & operator
    # See https://docs.python.org/3/reference/datamodel.html#object.__and__
    def __and__(self, other):
        return self._elementwise_binary_helper(self, other, "bitwise_and", operator.and_)

    def __rand__(self, other):
        return self._elementwise_binary_helper(other, self, "bitwise_and", operator.and_)

    def __eq__(self, other):
        # NOTE This short-circuit allows queries like a == (), which is a valid comparison
        #   for a number in Python
        if not isinstance(other, (Number, NumberProxy)):
            return False

        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "eq", operator.eq, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    def __ge__(self, other):
        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "ge", operator.ge, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    def __gt__(self, other):
        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "gt", operator.gt, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    def __le__(self, other):
        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "le", operator.le, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    def __lt__(self, other):
        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "lt", operator.lt, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    def __ne__(self, other):
        # NOTE This short-circuit allows queries like a != (), which is a valid comparison
        #   for a number in Python
        if not isinstance(other, (Number, NumberProxy)):
            return True

        from thunder.core.utils import ELEMENTWISE_TYPE_PROMOTION_KIND

        return self._elementwise_binary_helper(
            self, other, "ne", operator.ne, ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
        )

    # NOTE This is a bitwise or triggered by the | operator
    # See https://docs.python.org/3/reference/datamodel.html#object.__or__
    def __or__(self, other):
        return self._elementwise_binary_helper(self, other, "bitwise_or", operator.or_)

    def __ror__(self, other):
        return self._elementwise_binary_helper(other, self, "bitwise_or", operator.or_)

    # The ^ operator
    # See https://docs.python.org/3/reference/datamodel.html#object.__xor__
    def __xor__(self, other):
        return self._elementwise_binary_helper(self, other, "bitwise_xor", operator.xor)

    def __rxor__(self, other):
        return self._elementwise_binary_helper(other, self, "bitwise_xor", operator.xor)

    #
    # Shift operations
    #
    # Issue "Implement logical and arithmetic left and right shifts"
    #   tracks implementing these

    def __lshift__(self, other):
        raise NotImplementedError

    def __rlshift__(self, other):
        raise NotImplementedError

    def __rshift__(self, other):
        raise NotImplementedError

    def __rrshift__(self, other):
        raise NotImplementedError

    #
    # Casts to Python numbers
    #
    # NOTE These casts must return actual Python numbers, Python itself does not
    #   permit returning a subclass like an IntegerProxy.

    def __complex__(self):
        return complex(self.value)

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)

    def __bool__(self):
        if is_jitting():
            method = resolve_method("check_bool_conversion", self)
            method(self, bool(self.value))
        return bool(self.value)

    #
    # Matmul operators (not implemented for numbers)
    #

    def __matmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError

    # Inplace ops return NotImplemented instructing Python to use the out of place to do *= etc.

    def __iadd__(self, other):
        return NotImplemented

    def __iand__(self, other):
        return NotImplemented

    def __iconcat__(self, other):
        return NotImplemented

    def __ifloordiv__(self, other):
        return NotImplemented

    def __ilshift__(self, other):
        return NotImplemented

    def __imatmul__(self, other):
        return NotImplemented

    def __imod__(self, other):
        return NotImplemented

    def __imul__(self, other):
        return NotImplemented

    def __ior__(self, other):
        return NotImplemented

    def __ipow__(self, other):
        return NotImplemented

    def __irshift__(self, other):
        return NotImplemented

    def __isub__(self, other):
        return NotImplemented

    def __itruediv__(self, other):
        return NotImplemented

    def __ixor__(self, other):
        return NotImplemented


NumberLike = Number | NumberProxy


def pyval(x: NumberLike | str | AnyProxy) -> Number | str | any:
    baseutils.check_type(x, (NumberProxy, Number, str, AnyProxy))

    if isinstance(x, AnyProxy):
        return x._o

    if isinstance(x, (NumberProxy, StringProxy)):
        return x.value

    return x


def pytype(x: Proxy) -> type | None:
    if isinstance(x, AnyProxy):
        return type(x._o)

    if isinstance(x, (complex, ComplexProxy)):
        return complex
    if isinstance(x, (float, FloatProxy)):
        return float
    if isinstance(x, bool):
        return bool
    if isinstance(x, IntegerProxy) and x.python_type is bool:
        return bool
    if isinstance(x, (int, IntegerProxy)):
        return int
    if isinstance(x, str):
        return str
    if isinstance(x, tuple):
        return tuple
    if isinstance(x, list):
        return list
    if isinstance(x, dict):
        return dict


# TODO RC1 Update Proxy number inits to be value, /, *, name, history
class ComplexProxy(NumberProxy):
    def __init__(
        self,
        name=None,
        value=None,
        history: None | tuple = None,
        constraint: None | CONSTRAINT = None,
        tags: set | None = None,
    ):
        NumberProxy.__init__(
            self, name=name, value=value, python_type=complex, history=history, constraint=constraint, tags=tags
        )

    def replace(self, **changes):
        r"""Return a copy of the ComplexProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``value``, ``constraint``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
            value=self.value,
            constraint=self.constraint,
            __class__=self.__class__,  # undocumented on purpose
        )
        kwargs.update(changes)
        cls = kwargs.pop("__class__")
        return cls(**kwargs)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"complex {value_str}"


# TODO Review dtype conversions
# TODO Review -9999 as the marker value for unknown values
class IntegerProxy(NumberProxy):
    def __init__(
        self,
        name: str | None = None,
        value=None,
        history: None | tuple = None,
        constraint: None | CONSTRAINT = None,
        tags: set | None = None,
    ):
        # NOTE bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(
            self, name=name, value=value, python_type=python_type, history=history, constraint=constraint, tags=tags
        )

    def replace(self, **changes):
        r"""Return a copy of the IntegerProxy with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``value``, ``constraint``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
            value=self.value,
            constraint=self.constraint,
            __class__=self.__class__,
        )
        kwargs.update(changes)
        cls = kwargs.pop("__class__")
        return cls(**kwargs)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        type_str = "int" if self.python_type is int else "bool"
        return f"{type_str} {value_str}"

    def __repr__(self):
        if self.python_type is bool:
            return f"[IntegerProxy (bool type) name={self.name}, value={self.value}, static={self.constraint}]"
        return f"[IntegerProxy name={self.name}, value={self.value}, static={self.constraint}]"

    def __index__(self):
        return self.value


# TODO Review dtype conversions
class FloatProxy(NumberProxy):
    def __init__(
        self,
        name=None,
        value=None,
        history: None | tuple = None,
        constraint: None | CONSTRAINT = None,
        tags: set | None = None,
    ):
        NumberProxy.__init__(
            self, name=name, value=value, python_type=float, history=history, constraint=constraint, tags=tags
        )

    def replace(self, **changes):
        r"""Return a copy of the FloatProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``value``, ``constraint``.
        Note that the copy will use the current (environment) tracectx."""
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
            value=self.value,
            constraint=self.constraint,
            __class__=self.__class__,  # undocumented on purpose
        )
        kwargs.update(changes)
        cls = kwargs.pop("__class__")
        return cls(**kwargs)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"float {value_str}"

    def __repr__(self):
        return f"[FloatProxy name={self.name}, value={self.value}, static={self.constraint}]"


class DistParallelType(Enum):
    NONE = auto()
    REPLICATED = auto()
    FULLY_SHARDED = auto()
    # Following two are for tensor parallelism
    COLUMN_WISE = auto()
    ROW_WISE = auto()


def _infer_tensor_properties(
    like: TensorProxy | FutureTensorProxy | None = None,
    shape: ShapeLike | None = None,
    device: devices.Device | None = None,
    dtype: dtypes.dtype | None = None,
    requires_grad: bool | None = None,
    grad: TensorProxy | None = None,
    distparallel_type: DistParallelType | None = None,
    thunder_fsdp_padding_size: int | None = None,
):
    _shape = None
    _device = None
    _dtype = None
    _requires_grad: None | bool = None
    _grad = None
    _dist_parallel_type = DistParallelType.NONE
    _thunder_fsdp_padding_size = None

    if like is not None:
        baseutils.check_type(like, (TensorProxy, FutureTensorProxy))
        _shape = tuple(like.shape)
        _device = like.device
        _dtype = like.true_dtype
        _requires_grad = like.requires_grad
        _grad = like.grad
        _dist_parallel_type = getattr(like, "distparallel_type", DistParallelType.NONE)

    if shape is not None:
        baseutils.check_valid_shape(shape)

    _shape = tuple(shape) if shape is not None else _shape
    _device = device if device is not None else _device
    _dtype = dtype if dtype is not None else _dtype
    _dtype = dtypes.numbertype_to_dtype(_dtype) if dtypes.is_numbertype(_dtype) else _dtype
    _requires_grad = requires_grad if requires_grad is not None else _requires_grad
    _requires_grad = False if not dtypes.is_inexact_dtype(_dtype) else _requires_grad
    _grad = grad if grad is not None else _grad
    _grad = None if not _requires_grad else _grad
    _dist_parallel_type = distparallel_type if distparallel_type is not None else _dist_parallel_type
    _thunder_fsdp_padding_size = (
        thunder_fsdp_padding_size if thunder_fsdp_padding_size is not None else _thunder_fsdp_padding_size
    )

    baseutils.check(_shape is not None, lambda: f"_shape cannot be None when creating TensorProxy")
    if not using_symbolic_values():
        _shape = tuple(pyval(x) for x in _shape)
        # Computes derived properties
        _numel = reduce(operator.mul, _shape, 1)
    else:
        # deferred computation of numel
        # TODO: similar to how `shape` is handled, this should be CSE or lifted for efficiency
        _numel = lambda *args: reduce(operator.mul, _shape, 1)

    # TODO Alias rank to ndim?
    _ndim = len(_shape)

    # Validates inputs
    baseutils.check_type(_device, devices.Device)
    baseutils.check_type(_dtype, dtypes.dtype)
    baseutils.check_type(_requires_grad, bool)
    baseutils.check_type(_dist_parallel_type, DistParallelType)
    if isinstance(_thunder_fsdp_padding_size, int):
        baseutils.check(
            _dist_parallel_type == DistParallelType.FULLY_SHARDED,
            lambda: f"{_dist_parallel_type = } and {_thunder_fsdp_padding_size = } do not work",
        )
        baseutils.check(
            _thunder_fsdp_padding_size > 0,
            lambda: f"{_thunder_fsdp_padding_size=} expected to be > 0 or `None`",
        )

    # NOTE for simplicity functions that want to reason about weak dtypes should explicitly request
    #   the true_dtype property
    _true_dtype = _dtype
    _dtype = dtypes.to_strong_dtype(_dtype)

    return (
        _shape,
        _device,
        _dtype,
        _true_dtype,
        _numel,
        _ndim,
        _requires_grad,
        _grad,
        _dist_parallel_type,
        _thunder_fsdp_padding_size,
    )


# NOTE A FutureTensorProxy is intentionally NOT a subclass of TensorProxy
class FutureTensorProxy(Proxy, TensorProxyInterface):
    def __init__(
        self,
        name: str | None = None,
        *,
        like: TensorProxy | FutureTensorProxy | None = None,
        shape: ShapeLike | None = None,
        device: devices.Device | None = None,
        dtype: dtypes.dtype | None = None,
        prefix: None | str = None,
        history: None | tuple = None,
        tags: set | None = None,
    ):
        super().__init__(name, prefix=prefix, history=history, tags=tags)

        # NOTE FutureTensorProxies never require grad
        (
            self._shape,
            self._device,
            self._dtype,
            self._true_dtype,
            self._numel,
            self._ndim,
            self._requires_grad,
            _,  # grad
            _,  # distparallel_type
            _,  # thunder_fsdp_padding_size
        ) = _infer_tensor_properties(
            like,
            shape,
            device,
            dtype,
            False,
        )

        trace = get_tracectx()
        if trace is not None:
            trace._any_future_tensors = True

    @property
    def shape(self):
        return self._shape

    @property
    def numel(self):
        return self._numel

    @property
    def ndim(self):
        return self._ndim

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def true_dtype(self):
        return self._true_dtype

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return None  # FutureTensorProxies never require grad

    def __repr__(self):
        return f'<{type(self).__name__}(name="{self.name}", dtype={self.dtype}, shape={self.shape})>'

    def type_string(self):
        return f"FUTURE {self.device} {self.dtype.shortname()}{list(self.shape)}"

    def wait(self) -> TensorProxy:
        from thunder.distributed.prims import wait

        return wait(self)

    def replace(self, **changes):
        r"""Return a copy of the FutureTensorProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``shape``, ``dtype``, ``device``.
        ``like`` is also a valid keyword and will take metadata from the tensor proxy argument
        in preference to the old values but overridable by keyword arguments.
        Note that the copy will use the current (environment) tracectx."""

        like = changes.get("like")
        (
            shape,
            device,
            dtype,
            true_dtype,
            numel,
            ndim,
            requires_grad,
            _,  # grad
            _,  # distparallel_type
            _,  # thunder_fsdp_padding_size
        ) = _infer_tensor_properties(
            like,
            changes.get("shape", self._shape if like is None else None),
            changes.get("device", self._device if like is None else None),
            changes.get("dtype", self._dtype if like is None else None),
            False,
        )
        name = changes.get("name", self.name)
        history = changes.get("history", self.history)
        tags = changes.get("tags", self.tags)
        return FutureTensorProxy(
            name=name,
            tags=tags,
            shape=shape,
            device=device,
            dtype=dtype,
            history=history,
        )


# TODO RC1 Review dunders -- any remaining?
class TensorProxy(Proxy, TensorProxyInterface):
    def __init__(
        self,
        name: str | None = None,
        *,
        like: TensorProxy | FutureTensorProxy | None = None,
        shape: ShapeLike | None = None,
        device: devices.Device | None = None,
        dtype: dtypes.dtype | None = None,
        requires_grad: bool | None = None,
        grad: TensorProxy | None = None,
        prefix: None | str = None,
        distparallel_type: DistParallelType | None = None,
        history: None | tuple = None,
        tags: set | None = None,
        thunder_fsdp_padding_size: int | None = None,
    ):
        super().__init__(name, prefix=prefix, history=history, tags=tags)

        (
            self._shape,
            self._device,
            self._dtype,
            self._true_dtype,
            self._numel,
            self._ndim,
            self._requires_grad,
            self._grad,
            self._distparallel_type,
            self._thunder_fsdp_padding_size,
        ) = _infer_tensor_properties(
            like,
            shape,
            device,
            dtype,
            requires_grad,
            grad,
            distparallel_type,
            thunder_fsdp_padding_size,
        )

    # NOTE The following properties DO NOT depend on the language context or record
    #   themselves into the trace, so they can be used when working with tensor proxies
    #   outside of a trace or language context
    @property
    def shape(self):
        if not using_symbolic_values() or not is_tracing():
            return self._shape
        else:
            from thunder.core.prims import shape

            return shape(self)

    @property
    def ndim(self):
        return self._ndim

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def true_dtype(self):
        return self._true_dtype

    @property
    def requires_grad(self):
        return self._requires_grad

    @property
    def grad(self):
        return self._grad

    @property
    def distparallel_type(self):
        return self._distparallel_type

    @property
    def thunder_fsdp_padding_size(self):
        return self._thunder_fsdp_padding_size

    # We need to implement `__len__` as
    # > In addition to bypassing any instance attributes in the
    # > interest of correctness, implicit special method lookup
    # > generally also bypasses the __getattribute__() method
    # > even of the object’s metaclass
    # Ref: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
    def __len__(self):
        fn = resolve_method("len", self)
        return fn(self)

    def replace(self, **changes):
        r"""Return a copy of the TensorProxy object with new values for the specified fields as given to the constructor as arguments.
        Valid keyword arguments are ``name``, ``history``, ``shape``, ``dtype``, ``device``, ``requires_grad``, ``distparallel_type``,  ``thunder_fsdp_padding_size``.
        ``like`` is also a valid keyword and will take metadata from the tensor proxy argument
        in preference to the old values but overridable by keyword arguments.
        Note that the copy will use the current (environment) tracectx."""

        like = changes.get("like")
        (
            shape,
            device,
            dtype,
            true_dtype,
            numel,
            ndim,
            requires_grad,
            grad,
            distparallel_type,
            thunder_fsdp_padding_size,
        ) = _infer_tensor_properties(
            like,
            changes.get("shape", self._shape if like is None else None),
            changes.get("device", self._device if like is None else None),
            changes.get("dtype", self._dtype if like is None else None),
            changes.get("requires_grad", self._requires_grad if like is None else None),
            changes.get("grad", self._grad if like is None else None),
            changes.get("distparallel_type", self._distparallel_type if like is None else None),
            changes.get("thunder_fsdp_padding_size", self._thunder_fsdp_padding_size if like is None else None),
        )
        name = changes.get("name", self.name)
        history = changes.get("history", self.history)
        tags = changes.get("tags", self.tags)
        return TensorProxy(
            name=name,
            tags=tags,
            shape=shape,
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
            distparallel_type=distparallel_type,
            thunder_fsdp_padding_size=thunder_fsdp_padding_size,
            history=history,
        )

    def __repr__(self):
        return f'<{type(self).__name__}(name="{self.name}", dtype={self.dtype}, shape={self._shape})>'

    def type_string(self):
        return f"{self.device.device_str()} {self.dtype.shortname()}{list(self._shape)}"

    # NOTE __getattr__ is overridden to support language-specific methods
    def __getattr__(self, attr: str, /):
        method_or_value: None | Callable | Any = resolve_method(attr, self)
        if method_or_value is None:
            method_or_value = self.get_default_attr(attr)
        baseutils.check(method_or_value is not None, lambda: f"Unknown attribute {attr}", exception_type=AttributeError)

        if callable(method_or_value):
            # TODO: This is a temporary fix to allow for the `numel` attribute
            # to be called without arguments. This is a workaround for the fact
            # that the `numel` was initially in Thunder introduced not as a
            # method but a property. Now a lot of code relies on it being a
            # property. But PyTorch uses it as a method. We need to converge on
            # one or the other.
            # https://github.com/Lightning-AI/lightning-thunder/issues/925
            class _Numel(int):
                def __new__(cls, value):
                    assert isinstance(value, int), f"Expected int, got {type(value)}"
                    return int.__new__(cls, value)

                def __call__(self):
                    return int(self)

            if attr == "numel":
                if isinstance(self._numel, int):
                    return _Numel(self._numel)
                return method_or_value(self)
            return partial(method_or_value, self)

        return method_or_value

    def __iter__(self):
        # NOTE: this implementation is equivalent to torch.Tensor.__iter__

        if self.ndim == 0:
            raise TypeError("iteration over a 0-dim tensor")

        unbound_tuple = self.unbind(0)
        return iter(unbound_tuple)

    #
    # Default attribute
    #
    def get_default_attr(self, attr: str, /) -> None | Any:
        if attr == "numel":
            return self._numel
        return None

    #
    # Datatype conversion shorthands
    #

    def float(self):
        method = resolve_method("float", self)
        return method(self)

    #
    # Indexing operators
    #

    def __getitem__(self, key):
        method = resolve_method("getitem", self, key)
        return method(self, key)

    def __setitem__(self, key, value):
        method = resolve_method("setitem_", self, key, value)
        return method(self, key, value)

    #
    # Elementwise unary operators
    #

    def __abs__(self):
        method = resolve_method("abs", self)
        return method(self)

    def __ceil__(self):
        method = resolve_method("ceil", self)
        return method(self)

    def __floor__(self):
        method = resolve_method("floor", self)
        return method(self)

    def __invert__(self):
        method = resolve_method("bitwise_not", self)
        return method(self)

    def __neg__(self):
        method = resolve_method("neg", self)
        return method(self)

    def __pos__(self):
        method = resolve_method("pos", self)
        return method(self)

    def __round__(self):
        method = resolve_method("round", self)
        return method(self)

    def __trunc__(self):
        method = resolve_method("trunc", self)
        return method(self)

    #
    # dtype conversion operators
    #

    def __complex__(self):
        raise NotImplementedError

    def __float__(self):
        raise NotImplementedError

    def __int__(self):
        raise NotImplementedError

    def __bool__(self):
        raise NotImplementedError

    #
    # Elementwise binary operators
    #

    def __add__(self, other):
        method = resolve_method("add", self, other)
        return method(self, other)

    def __iadd__(self, other):
        method = resolve_method("add_", self, other)
        return method(self, other)

    def __radd__(self, other):
        method = resolve_method("add", other, self)
        return method(other, self)

    def __divmod__(self, other):
        method = resolve_method("divmod", self, other)
        return method(self, other)

    def __rdivmod__(self, other):
        method = resolve_method("divmod", other, self)
        return method(other, self)

    def __eq__(self, other):
        method = resolve_method("eq", self, other)
        return method(self, other)

    def __floordiv__(self, other):
        method = resolve_method("floor_divide", self, other)
        return method(self, other)

    def __rfloordiv__(self, other):
        method = resolve_method("floor_divide", other, self)
        return method(other, self)

    def __mod__(self, other):
        method = resolve_method("mod", self, other)
        return method(self, other)

    def __rmod__(self, other):
        method = resolve_method("mod", other, self)
        return method(other, self)

    def __mul__(self, other):
        method = resolve_method("mul", self, other)
        return method(self, other)

    def __imul__(self, other):
        method = resolve_method("mul_", self, other)
        return method(self, other)

    def __rmul__(self, other):
        method = resolve_method("mul", other, self)
        return method(other, self)

    def __pow__(self, other):
        method = resolve_method("pow", self, other)
        return method(self, other)

    def __ipow__(self, other):
        method = resolve_method("pow_", self, other)
        return method(self, other)

    def __rpow__(self, other):
        method = resolve_method("pow", other, self)
        return method(other, self)

    def __sub__(self, other):
        method = resolve_method("sub", self, other)
        return method(self, other)

    def __isub__(self, other):
        method = resolve_method("sub_", self, other)
        return method(self, other)

    def __rsub__(self, other):
        method = resolve_method("sub", other, self)
        return method(other, self)

    def __truediv__(self, other):
        method = resolve_method("true_divide", self, other)
        return method(self, other)

    def __rtruediv__(self, other):
        method = resolve_method("true_divide", other, self)
        return method(other, self)

    def __itruediv__(self, other):
        method = resolve_method("div_", self, other, rounding_mode=None)
        return method(self, other)

    #
    # Logical operations
    #

    # TODO Review logical vs bitwise dispatch
    def __and__(self, other):
        method = resolve_method("bitwise_and", self, other)
        return method(self, other)

    def __rand__(self, other):
        method = resolve_method("bitwise_and", other, self)
        return method(other, self)

    def __ge__(self, other):
        method = resolve_method("ge", self, other)
        return method(self, other)

    def __gt__(self, other):
        method = resolve_method("gt", self, other)
        return method(self, other)

    def __le__(self, other):
        method = resolve_method("le", self, other)
        return method(self, other)

    def __lt__(self, other):
        method = resolve_method("lt", self, other)
        return method(self, other)

    def __ne__(self, other):
        method = resolve_method("ne", self, other)
        return method(self, other)

    # TODO Review logical vs bitwise dispatch
    def __or__(self, other):
        method = resolve_method("bitwise_or", self, other)
        return method(self, other)

    def __ror__(self, other):
        method = resolve_method("bitwise_or", other, self)
        return method(other, self)

    def __xor__(self, other):
        method = resolve_method("bitwise_xor", self, other)
        return method(self, other)

    def __rxor__(self, other):
        method = resolve_method("bitwise_xor", other, self)
        return method(other, self)

    #
    # Shift operations
    #

    def __lshift__(self, other):
        method = resolve_method("lshift", self, other)
        return method(self, other)

    def __rlshift__(self, other):
        method = resolve_method("lshift", other, self)
        return method(other, self)

    def __rshift__(self, other):
        method = resolve_method("rshift", self, other)
        return method(self, other)

    def __rrshift__(self, other):
        method = resolve_method("rshift", other, self)
        return method(other, self)

    #
    # Matmul
    #

    def __matmul__(self, other):
        method = resolve_method("matmul", self, other)
        return method(self, other)

    def __rmatmul__(self, other):
        method = resolve_method("matmul", other, self)
        return method(other, self)

    #
    # Transposes
    #

    @property
    def T(self):
        method = resolve_method("T", self)
        return method(self)

    @property
    def mT(self):
        method = resolve_method("mT", self)
        return method(self)

    #
    # Real
    #

    @property
    def real(self):
        method = resolve_method("real", self)
        return method(self)


class TorchAutogradFunctionCtxProxy(Proxy, TorchAutogradFunctionCtxProxyInterface):
    def __init__(
        self,
        ctx: torch.autograd.function.FunctionCtx,
        /,
        *,
        name: str | None = None,
        history: tuple[Any, ...] | None = None,
        tags: set[ProxyTag, ...] | None = None,
    ):
        self._ctx = ctx
        self._tensors: list[TensorProxy] = []
        self._const_for_backward: dict[str, Any] = {}
        super().__init__(name=name, history=history, tags=tags)

    def type_string(self) -> str:
        return "TorchAutogradFunctionCtxProxy"

    def __repr__(self) -> str:
        return f"<TorchAutogradFunctionCtxProxy '{self.name}', '{self.saved_tensors=}', '{self._const_for_backward=}'>"

    def replace(self, **changes):
        kwargs = dict(
            name=self.name,
            history=self.history,
            tags=self.tags,
        )
        kwargs.update(**changes)
        return TorchAutogradFunctionCtxProxy(
            self._ctx,
            **kwargs,
        )

    @property
    def saved_tensors(self) -> tuple[TensorProxy, ...]:
        return tuple(self._tensors)

    @property
    def saved_consts(self) -> tuple[Any, ...]:
        return tuple(self._const_for_backward.values())

    def save_for_backward(self, *tensors):
        self._tensors.extend(tensors)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(self, "_const_for_backward") and name not in {
            "_name",
            "_has_weak_name",
            "history",
            "_tags",
            "_const_for_backward",
            "_tensors",
        }:
            self._const_for_backward[name] = value


#
# Helpers for creating and working with proxies
#

_cls_to_number_proxy_map = {
    float: FloatProxy,
    int: IntegerProxy,
    bool: IntegerProxy,
    complex: ComplexProxy,
}


# TODO: move this function to jit_ext.py
def tensorproxy(t: torch.Tensor, /, *, name: None | str, history: None | tuple = None) -> TensorProxy:
    from thunder.core.interpreter import ProvenanceRecord, PseudoInst, wrap_const

    if hasattr(t, "_thunder_device"):
        torch_device = t._thunder_device
    else:
        torch_device = t.device
    device = devices.to_device(torch_device)
    dtype = dtypes.to_dtype(t.dtype)

    grad = None
    if t.is_leaf and t.grad is not None:
        grad_pr = None
        if history is not None:
            attr_pr = ProvenanceRecord(inst=PseudoInst.CONSTANT, inputs=[], value="grad")
            grad_pr = ProvenanceRecord(PseudoInst.LOAD_ATTR, inputs=[history, attr_pr])
        grad = tensorproxy(t.grad, name=f"{name}_grad", history=grad_pr)

    # See Note [DistributedDataParallel and distparallel_type]
    distparallel_type = getattr(t, "distparallel_type", None)
    _thunder_fsdp_padding_size = getattr(t, "_thunder_fsdp_padding_size", None)
    if using_symbolic_values():
        shape_attr = ProvenanceRecord(PseudoInst.LOAD_ATTR, inputs=[copy.copy(history), wrap_const("shape").provenance])
        shape = tuple(
            IntegerProxy(
                None,
                s,
                history=ProvenanceRecord(PseudoInst.BINARY_SUBSCR, inputs=[shape_attr, wrap_const(idx).provenance]),
                constraint=CONSTRAINT.CONSTRAINABLE,
            )
            for idx, s in enumerate(t.shape)
        )
    else:
        # NOTE Without tuple(t.shape) then the shape would be a torch.Size object
        shape = tuple(t.shape)
    return TensorProxy(
        name,
        shape=tuple(shape),
        device=device,
        dtype=dtype,
        requires_grad=t.requires_grad,
        grad=grad,
        distparallel_type=distparallel_type,
        history=history,
        thunder_fsdp_padding_size=_thunder_fsdp_padding_size,
    )


def futuretensorproxy(
    t: torch.Tensor | TensorProxy | FutureTensorProxy, /, *, name: None | str, history: None | tuple = None
) -> FutureTensorProxy:
    if hasattr(t, "_thunder_device"):
        torch_device = t._thunder_device
    else:
        torch_device = t.device
    device = devices.to_device(torch_device)
    dtype = dtypes.to_dtype(t.dtype)
    # NOTE Without tuple(t.shape) then the shape would be a torch.Size object
    return FutureTensorProxy(
        name,
        shape=tuple(t.shape),
        device=device,
        dtype=dtype,
        history=history,
    )


def numberproxy(cls: type, value: Number | None, constraint: None | CONSTRAINT = None) -> NumberProxy:
    pcls = _cls_to_number_proxy_map[cls]
    return pcls(value=value, constraint=constraint)


# TODO RC1 Remove this function
def is_proxyable(x: Any, /) -> bool:
    if isinstance(x, Proxy):
        return False

    return isinstance(x, (Number, torch.Tensor))


def proxy(x: Any, *, name: str | None = None, history: None | tuple = None) -> Any | Proxy:
    if x is None:
        return AnyProxy(None, name=name, history=history)
    if type(x) is slice:
        return AnyProxy(x, name=name, history=history)
    if x is ...:
        return AnyProxy(x, name=name, history=history)

    if isinstance(x, torch.Tensor):
        return tensorproxy(x, name=name, history=history)

    if isinstance(x, str):
        return StringProxy(x, name=name, history=history)

    if isinstance(x, Number):
        if isinstance(x, complex):
            return ComplexProxy(name=name, value=x, history=history)
        if isinstance(x, float):
            return FloatProxy(name=name, value=x, history=history)
        if isinstance(x, int):
            return IntegerProxy(name=name, value=x, history=history)

        raise NotImplementedError

    if isinstance(x, tuple):
        return TupleProxy(x, name=name, history=history)
    if isinstance(x, list):
        return ListProxy(x, name=name, history=history)
    if isinstance(x, dict):
        return DictProxy(x, name=name, history=history)

    if isinstance(x, torch.dtype):
        return AnyProxy(x, name=name, history=history)
    if isinstance(x, torch.device):
        return AnyProxy(x, name=name, history=history)
    if isinstance(x, torch.autograd.function.FunctionCtx):
        return TorchAutogradFunctionCtxProxy(x, name=name, history=history)
    if isinstance(x, torch.memory_format):
        return AnyProxy(x, name=name, history=history)

    return x
