from __future__ import annotations

from enum import auto, Enum
from numbers import Number
from typing import Type, Optional, Any, Tuple, List, Union
from collections.abc import Callable
from collections.abc import Sequence
from functools import reduce, partial
import operator
import builtins
import math

from thunder.core.trace import VariableInterface, get_tracectx, TraceCtx
from thunder.core.baseutils import ProxyInterface, NumberProxyInterface, TensorProxyInterface
import thunder.core.baseutils as baseutils
from thunder.core.langctxs import resolve_method, get_langctx, LanguageContext
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


def make_proxy_name(*, name: None | str = None, prefix: None | str = None) -> str:
    trc = get_tracectx()

    if name is not None and not trc.has_name(name):
        trc.add_name(name)
        return name

    return trc.make_name(prefix=prefix)


# TODO Document this class
# TODO Support multiple histories
class Proxy(VariableInterface, ProxyInterface):
    def __init__(self, name: str | None = None, *, prefix: None | str = None, history: None | tuple = None):
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
            elif isinstance(self, TensorProxy):
                prefix = "t"
            elif isinstance(self, CollectionProxy):
                prefix = "C"
            else:
                prefix = "p"

        self._name = make_proxy_name(name=name, prefix=prefix)
        self._has_weak_name: bool = name is None
        self.history = history

    @property
    def name(self) -> str:
        return self._name

    def replace_name(self, name: str | None = None):
        """Return a copy of this proxy with the given name."""
        return self.__class__(name=name)

    def __repr__(self) -> str:
        return f"{self._name}"

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


# This is a proxy for anything not covered by more type-specific proxies.
# We do need this within the litjit to track history but differs from the other
# proxies in that we do not try as much to mimick the proxied object as
# for the other proxies.
class AnyProxy(Proxy):
    def __init__(self, o: Any, /, *, name: str | None = None, history: None | tuple = None):
        super().__init__(name=name, history=history)
        self._o = o


class StringProxy(Proxy, str):
    def __new__(cls, s: str, /, *, name: str | None = None, history: None | tuple = None):
        return str.__new__(cls, s)

    def __init__(self, s: str, /, *, name: str | None = None, history: None | tuple = None):
        Proxy.__init__(self, name=name, history=history)
        self._s: str = s

    def __str__(self) -> str:
        return self._s

    def type_string(self) -> str:
        return "str"


class CollectionProxy(Proxy):
    def __init__(self, coll: Any, *, name: str | None = None):
        Proxy.__init__(self, name=name)
        self.coll = coll

    def collection(self) -> Any:
        return self.coll

    def type_string(self) -> str:
        return "Collection"


# NOTE NumberProxies are NOT Numbers
# TODO Maybe NumberProxies should be Numbers?
class NumberProxy(Proxy, NumberProxyInterface):
    def __init__(
        self, name: str | None = None, value: Number | None = None, *, python_type: type, history: None | tuple = None
    ):
        self.value = value
        self.python_type = python_type

        Proxy.__init__(self, name, history=history)

    # NOTE: Python numbers hash to themselves, and this mimics that behavior
    def __hash__(self) -> int:
        return hash(self.value)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return self.__class__(name=name, value=self.value, python_type=self.python_type)

    def known_value(self) -> bool:
        return self.value is not None

    #
    # Elementwise unary operators
    #

    # name is the name of the operation in the number language context to perform
    # fn is the function to call if executing outside a language context
    @staticmethod
    def _elementwise_unary_helper(a, name, fn):
        trace: None | TraceCtx = get_tracectx()

        langctx: None | LanguageContext
        try:
            langctx = get_langctx()
        except LookupError as le:
            langctx = None

        vala = pyval(a)

        if trace is None:
            # Outside of a trace context, operations on NumberProxies are executed by the
            #   Python interpreter

            baseutils.check(
                vala is not None,
                lambda: f"Trying to {name} a number with an unknown value",
                exception_type=AssertionError,
            )
            return fn(vala)

        method: Callable

        has_prologue: bool = trace.prologue is not None
        if has_prologue:
            method: Callable = resolve_method(name, a)
            return method(a)
        else:
            # Outside the interpreter this tries to find a method, and if none exists the
            #   operation silently fallsback to the Python interpreter
            try:
                method = resolve_method(name, a)
            except Exception as e:
                return fn(vala)

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
    def _elementwise_binary_helper(a, b, name, fn):
        baseutils.check_type(b, (Number, TensorProxy))

        trace: None | TraceCtx = get_tracectx()
        langctx: None | LanguageContext
        try:
            langctx = get_langctx()
        except LookupError as le:
            langctx = None

        vala = pyval(a)
        valb = pyval(b) if isinstance(b, NumberProxy) else b
        if trace is None:
            # Outside of a trace or language context, binary operations on NumberProxies are
            #   executed by the Python interpreter

            return fn(vala, valb)

        has_prologue: bool = trace.prologue is not None
        if has_prologue:
            fn: None | Callable = resolve_method(name, a, b)

            if fn is None:
                langctx: LanguageContext = get_langctx()
                raise ValueError(
                    f"Could not find a binary operator called {name} on numbers in the {langctx.name} language context"
                )

            return fn(a, b)

        if isinstance(b, TensorProxy):
            tensor_fn = resolve_method(name, a, b)
            return tensor_fn(a, b)

        # TODO GTC Record binary operations outside the using_interpreter mode
        return fn(vala, valb)

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
        if not isinstance(other, Number):
            return False

        return self._elementwise_binary_helper(self, other, "eq", operator.eq)

    def __ge__(self, other):
        return self._elementwise_binary_helper(self, other, "ge", operator.ge)

    def __gt__(self, other):
        return self._elementwise_binary_helper(self, other, "gt", operator.gt)

    def __le__(self, other):
        return self._elementwise_binary_helper(self, other, "le", operator.le)

    def __lt__(self, other):
        return self._elementwise_binary_helper(self, other, "lt", operator.lt)

    def __ne__(self, other):
        # NOTE This short-circuit allows queries like a != (), which is a valid comparison
        #   for a number in Python
        if not isinstance(other, Number):
            return True

        return self._elementwise_binary_helper(self, other, "ne", operator.ne)

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
    # Issue https://github.com/Lightning-AI/lightning-thunder/issues/594
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


def pyval(x: NumberProxy | Number) -> Number:
    baseutils.check_type(x, (NumberProxy, Number))

    # NOTE This has to query NumberProxy, not Number, because NumberProxies are Numbers
    #   (but not all Numbers are NumberProxies)
    if isinstance(x, NumberProxy):
        return x.value

    return x


def pytype(x: NumberProxy | Number) -> type:
    baseutils.check_type(x, (NumberProxy, Number))

    if isinstance(x, NumberProxy):
        return x.python_type

    return type(x)


# TODO GTC Update Proxy number inits to be value, /, *, name, history
class ComplexProxy(NumberProxy, complex):
    def __new__(cls, *, name=None, value, history: None | tuple = None):
        if value is None:
            value = complex(float("nan"), float("nan"))

        return complex.__new__(cls, value)

    def __init__(self, name=None, value=None, history: None | tuple = None):
        NumberProxy.__init__(self, name=name, value=value, python_type=complex, history=history)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return ComplexProxy(name=name, value=self.value)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"complex {value_str}"


# TODO Review dtype conversions
# TODO Review -9999 as the marker value for unknown values
class IntegerProxy(NumberProxy, int):
    def __new__(cls, *, name: str | None = None, value: Number, history: None | tuple = None):
        if value is None:
            value = -9999

        return int.__new__(cls, value)

    def __init__(self, name: str | None = None, value=None, history: None | tuple = None):
        # NOTE bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(self, name=name, value=value, python_type=python_type, history=history)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return IntegerProxy(name=name, value=self.value)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        type_str = "int" if self.python_type is int else "bool"
        return f"{type_str} {value_str}"

    def __repr__(self):
        if self.python_type is bool:
            return f"[IntegerProxy (bool type) name={self.name}, value={self.value}]"
        return f"[IntegerProxy name={self.name}, value={self.value}]"


# TODO Review dtype conversions
class FloatProxy(NumberProxy, float):
    def __new__(cls, *, name=None, value, history: None | tuple = None):
        if value is None:
            value = float("nan")

        return float.__new__(cls, value)

    def __init__(self, name=None, value=None, history: None | tuple = None):
        NumberProxy.__init__(self, name=name, value=value, python_type=float, history=history)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return FloatProxy(name=name, value=self.value)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"float {value_str}"

    def __repr__(self):
        return f"[FloatProxy name={self.name}, value={self.value}]"


class DDPType(Enum):
    NONE = auto()
    REPLICATED = auto()
    FULLY_SHARDED = auto()


def _infer_tensor_properties(
    like: TensorProxy | FutureTensorProxy | None = None,
    shape: ShapeLike | None = None,
    device: devices.Device | None = None,
    dtype: dtypes.dtype | None = None,
    requires_grad: bool | None = None,
    ddp_type: DDPType | None = None,
):
    _shape = None
    _device = None
    _dtype = None
    _requires_grad: None | bool = None
    _ddp_type = DDPType.NONE

    if like is not None:
        baseutils.check_type(like, (TensorProxy, FutureTensorProxy))
        _shape = tuple(like.shape)
        _device = like.device
        _dtype = like.true_dtype
        _requires_grad = like.requires_grad
        _ddp_type = getattr(like, "ddp_type", DDPType.NONE)

    if shape is not None:
        baseutils.check_valid_shape(shape)

    _shape = tuple(shape) if shape is not None else _shape
    _device = device if device is not None else _device
    _dtype = dtype if dtype is not None else _dtype
    _dtype = dtypes.numbertype_to_dtype(_dtype) if dtypes.is_numbertype(_dtype) else _dtype
    _requires_grad = requires_grad if requires_grad is not None else _requires_grad
    _requires_grad = False if not dtypes.is_inexact_dtype(_dtype) else _requires_grad
    _ddp_type = ddp_type if ddp_type is not None else _ddp_type

    # Extracts actual values for shape
    # TODO This will need to be revisited when we add support for dynamic constraints
    _shape = tuple(pyval(x) for x in _shape)

    # Computes derived properties
    _numel = reduce(operator.mul, _shape, 1)

    # TODO Alias rank to ndim?
    _ndim = len(_shape)

    # Validates inputs
    baseutils.check_type(_device, devices.Device)
    baseutils.check_type(_dtype, dtypes.dtype)
    baseutils.check_type(_requires_grad, bool)
    baseutils.check_type(_ddp_type, DDPType)

    # NOTE for simplicity functions that want to reason about weak dtypes should explicitly request
    #   the true_dtype property
    _true_dtype = _dtype
    _dtype = dtypes.to_strong_dtype(_dtype)

    return _shape, _device, _dtype, _true_dtype, _numel, _ndim, _requires_grad, _ddp_type


# NOTE A FutureTensorProxy is intentionally NOT a subclass of TensorProxy
class FutureTensorProxy(Proxy):
    def __init__(
        self,
        name: str | None = None,
        *,
        like: TensorProxy | FutureTensorProxy | None = None,
        shape: ShapeLike | None = None,
        device: devices.Device | None = None,
        dtype: dtypes.dtype | None = None,
    ):
        super().__init__(name)

        # NOTE FutureTensorProxies never require grad
        (
            self._shape,
            self._device,
            self._dtype,
            self._true_dtype,
            self._numel,
            self._ndim,
            self._requires_grad,
            _,
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

    def type_string(self):
        return f"FUTURE {self.device} {self.dtype.shortname()}{list(self.shape)}"

    @property
    def size(self, /) -> Any:
        fn = resolve_method("size", self)
        return fn(self)

    def wait(self) -> TensorProxy:
        from thunder.distributed.prims import wait

        return wait(self)

    # TODO GTC Update hasing on tensor types (this is just here because some distributed code directly hashes FutureTensorProxies)
    def __hash__(self) -> int:
        return hash(self.name)


# TODO GTC Review dunders -- any remaining?
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
        prefix: None | str = None,
        ddp_type: DDPType | None = None,
        history: None | tuple = None,
    ):
        super().__init__(name, prefix=prefix, history=history)

        (
            self._shape,
            self._device,
            self._dtype,
            self._true_dtype,
            self._numel,
            self._ndim,
            self._requires_grad,
            self._ddp_type,
        ) = _infer_tensor_properties(like, shape, device, dtype, requires_grad, ddp_type)

    # NOTE The following properties DO NOT depend on the language context or record
    #   themselves into the trace, so they can be used when working with tensor proxies
    #   outside of a trace or language context
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
    def ddp_type(self):
        return self._ddp_type

    @property
    def size(self, /) -> Any:
        fn = resolve_method("size", self)
        return fn(self)

    # We need to implement `__len__` as
    # > In addition to bypassing any instance attributes in the
    # > interest of correctness, implicit special method lookup
    # > generally also bypasses the __getattribute__() method
    # > even of the object’s metaclass
    # Ref: https://docs.python.org/3/reference/datamodel.html#special-method-lookup
    def __len__(self):
        fn = resolve_method("len", self)
        return fn(self)

    # TODO GTC Review this with other changes
    def replace_name(self, name: str):
        """Return a copy of this proxy with the given name."""
        return tensorproxy(self, name=name)

    def type_string(self):
        return f"{self.device} {self.dtype.shortname()}{list(self.shape)}"

    # NOTE __getattr__ is overridden to support language-specific methods
    def __getattr__(self, attr: str, /):
        method: None | Callable = resolve_method(attr, self)
        baseutils.check(method is not None, lambda: f"Unknown attribute {attr}", exception_type=AttributeError)
        return partial(method, self)

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

    def __rmul__(self, other):
        method = resolve_method("mul", other, self)
        return method(other, self)

    def __pow__(self, other):
        method = resolve_method("pow", self, other)
        return method(self, other)

    def __rpow__(self, other):
        method = resolve_method("pow", other, self)
        return method(other, self)

    def __sub__(self, other):
        method = resolve_method("sub", self, other)
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


#
# Helpers for creating and working with proxies
#

_cls_to_number_proxy_map = {
    float: FloatProxy,
    int: IntegerProxy,
    bool: IntegerProxy,
    complex: ComplexProxy,
}

import torch


def tensorproxy(t: torch.Tensor, /, *, name: None | str, history: None | tuple = None) -> TensorProxy:
    device = devices.device_from_string(str(t.device))
    dtype = dtypes.to_dtype(t.dtype)
    # See Note [DistributedDataParallel and ddp_type]
    ddp_type = getattr(t, "ddp_type", None)
    # NOTE Without tuple(t.shape) then the shape would be a torch.Size object
    return TensorProxy(
        name,
        shape=tuple(t.shape),
        device=device,
        dtype=dtype,
        requires_grad=t.requires_grad,
        ddp_type=ddp_type,
        history=history,
    )


def numberproxy(cls: type, value: Number | None) -> NumberProxy:
    pcls = _cls_to_number_proxy_map[cls]
    return pcls(value=value)


# TODO GTC Remove this function
def is_proxyable(x: Any, /) -> bool:
    if isinstance(x, Proxy):
        return False

    return isinstance(x, (Number, torch.Tensor))


def proxy(x: Any, *, name: str | None = None, history: None | tuple = None) -> Any | Proxy:
    if isinstance(x, torch.Tensor):
        return tensorproxy(x, name=name, history=history)

    if isinstance(x, str):
        return StringProxy(x, name=name, history=history)

    if isinstance(x, Number):
        if isinstance(x, complex):
            return ComplexProxy(name=name, value=x)
        if isinstance(x, float):
            return FloatProxy(name=name, value=x)
        if isinstance(x, int):
            return IntegerProxy(name=name, value=x, history=history)

        raise NotImplementedError

    return x
