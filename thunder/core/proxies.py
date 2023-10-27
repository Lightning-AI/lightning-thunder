from __future__ import annotations

from enum import auto, Enum
from numbers import Number
from typing import Type, Optional, Any, Tuple, List, Union
from collections.abc import Sequence
from functools import reduce, partial
import operator
import builtins
import math

from thunder.core.trace import VariableInterface, get_tracectx
from thunder.core.baseutils import ProxyInterface, NumberProxyInterface, TensorProxyInterface
import thunder.core.baseutils as baseutils
from thunder.core.langctx import langctx_for, get_langctx, get_numberctx
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


# TODO Document this class
class Proxy(VariableInterface, ProxyInterface):
    def __init__(self, name: str | None = None):
        trace = get_tracectx()
        if name is None:
            prefix: str | None = None
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

            name = trace.make_name(prefix=prefix)
        else:
            trace.add_name(name)

        self._name = name

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

    #
    # Inplace Operations
    #

    def __iadd__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __iand__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __iconcat__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __ifloordiv__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __ilshift__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __imatmul__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __imod__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __imul__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __ior__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __ipow__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __irshift__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __isub__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __itruediv__(self, other):
        raise RuntimeError("Inplace operators are not supported.")

    def __ixor__(self, other):
        raise RuntimeError("Inplace operators are not supported.")


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
    def __init__(self, name: str | None = None, value: Number | None = None, *, python_type: type):
        self.value = value
        self.python_type = python_type

        Proxy.__init__(self, name)

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
        vala = pyval(a)
        baseutils.check(
            vala is not None, lambda: f"Trying to {name} a number with an unknown value", exception_type=AssertionError
        )

        # TODO Make it so failing to find the operation is a failure
        # Records the operation (on a number) if in a language context
        langctx = get_langctx()
        if langctx is not None:
            numberctx = get_numberctx()
            fn = getattr(numberctx, name, None)
            if fn is not None:
                return fn(a)

        # NOTE langctx is None
        # In this case the operation is (conceptually) run eagerly
        return fn(vala)

    def __abs__(self):
        return self._elementwise_unary_helper(self, "py_abs", builtins.abs)

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
        baseutils.check_type(b, (TensorProxy, Number))

        vala = pyval(a)
        baseutils.check(
            vala is not None, lambda: f"Trying to {name} a number with an unknown value", exception_type=AssertionError
        )

        langctx = get_langctx()
        if isinstance(b, TensorProxy):
            baseutils.check(
                langctx is not None,
                lambda: f"Cannot use an operator to {name} a number and a tensor outside of a language context",
            )
            tensor_fn = getattr(langctx, name)
            return tensor_fn(a, b)

        # NOTE isinstance(b, Number)
        valb = pyval(b)
        baseutils.check(
            valb is not None, lambda: f"Trying to {name} a number with an unknown value", exception_type=AssertionError
        )

        # TODO Enable this
        # Records the operation (on two numbers) if in a language context
        # if langctx is not None:
        #     fn = getattr(langctx, name)
        #     return fn(a, b)

        # NOTE langctx is None
        # In this case the operation is (conceptually) run eagerly
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


class ComplexProxy(NumberProxy, complex):
    def __new__(cls, *, name=None, value):
        if value is None:
            value = complex(float("nan"), float("nan"))

        return complex.__new__(cls, value)

    def __init__(self, name=None, value=None):
        NumberProxy.__init__(self, name=name, value=value, python_type=complex)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return ComplexProxy(name=name, value=self.value)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"complex {value_str}"


# TODO Review dtype conversions
# TODO Review -9999 as the marker value for unknown values
class IntegerProxy(NumberProxy, int):
    def __new__(cls, *, name: str | None = None, value: Number):
        if value is None:
            value = -9999

        return int.__new__(cls, value)

    def __init__(self, name: str | None = None, value=None):
        # NOTE bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(self, name=name, value=value, python_type=python_type)

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
    def __new__(cls, *, name=None, value):
        if value is None:
            value = float("nan")

        return float.__new__(cls, value)

    def __init__(self, name=None, value=None):
        NumberProxy.__init__(self, name=name, value=value, python_type=float)

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
    # FULLY_SHARDED = auto()


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
    def size(self):
        langctx = get_langctx()
        return langctx.size(self)

    def wait(self) -> TensorProxy:
        from thunder.distributed.prims import wait

        return wait(self)


# TODO Review dunders -- any remaining?
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
        ddp_type: DDPType | None = None,
    ):
        super().__init__(name)

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
    def size(self):
        langctx = get_langctx()
        return langctx.size(self)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        langctx = get_langctx()
        return langctx.tensorproxy(name, self)

    def type_string(self):
        return f"{self.device} {self.dtype.shortname()}{list(self.shape)}"

    # NOTE __getattr__ is overridden to support language-specific methods
    def __getattr__(self, attr):
        langctx = get_langctx()
        method = langctx.method_lookup(attr)

        baseutils.check(method is not None, lambda: f"Unknown attribute {attr}", exception_type=AttributeError)

        return partial(method, self)

    #
    # Datatype conversion shorthands
    #
    # NOTE This don't map the names directly because we want to avoid defining
    #   functions named "float" that clobber the builtin "float" wherever possible
    #   That's why these functions are defined directly (instead of automatically
    #   being translated using the language context)
    # TODO Implement additional shorthands

    def float(self):
        langctx = get_langctx()
        return langctx.to_float(self)

    #
    # Indexing operators
    #

    def __getitem__(self, key):
        ctx = get_langctx()
        return ctx.get_item(self, key)

    #
    # Elementwise unary operators
    #

    def __abs__(self):
        langctx = get_langctx()
        return langctx.abs(self)

    def __ceil__(self):
        langctx = get_langctx()
        return langctx.ceil(self)

    def __floor__(self):
        langctx = get_langctx()
        return langctx.floor(self)

    def __invert__(self):
        langctx = get_langctx()
        return langctx.bitwise_not(self)

    def __neg__(self):
        langctx = get_langctx()
        return langctx.neg(self)

    def __pos__(self):
        return self

    def __round__(self):
        langctx = get_langctx()
        return langctx.round(self)

    def __trunc__(self):
        langctx = get_langctx()
        return langctx.trunc(self)

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
        langctx = get_langctx()
        return langctx.add(self, other)

    def __radd__(self, other):
        langctx = get_langctx()
        return langctx.add(other, self)

    def __divmod__(self, other):
        langctx = get_langctx()
        return langctx.divmod(self, other)

    def __rdivmod__(self, other):
        langctx = get_langctx()
        return langctx.divmod(other, self)

    def __eq__(self, other):
        langctx = get_langctx()
        return langctx.eq(self, other)

    def __floordiv__(self, other):
        langctx = get_langctx()
        return langctx.floor_divide(self, other)

    def __rfloordiv__(self, other):
        langctx = get_langctx()
        return langctx.floor_divide(other, self)

    def __mod__(self, other):
        langctx = get_langctx()
        return langctx.mod(self, other)

    def __rmod__(self, other):
        langctx = get_langctx()
        return langctx.mod(other, self)

    def __mul__(self, other):
        langctx = get_langctx()
        return langctx.mul(self, other)

    def __rmul__(self, other):
        langctx = get_langctx()
        return langctx.mul(other, self)

    def __pow__(self, other):
        langctx = get_langctx()
        return langctx.pow(self, other)

    def __rpow__(self, other):
        langctx = get_langctx()
        return langctx.pow(other, self)

    def __sub__(self, other):
        langctx = get_langctx()
        return langctx.sub(self, other)

    def __rsub__(self, other):
        langctx = get_langctx()
        return langctx.sub(other, self)

    def __truediv__(self, other):
        langctx = get_langctx()
        return langctx.true_divide(self, other)

    def __rtruediv__(self, other):
        langctx = get_langctx()
        return langctx.true_divide(other, self)

    #
    # Logical operations
    #

    # TODO Review logical vs bitwise dispatch
    def __and__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_and(self, other)

    def __rand__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_and(other, self)

    def __ge__(self, other):
        langctx = get_langctx()
        return langctx.ge(self, other)

    def __gt__(self, other):
        langctx = get_langctx()
        return langctx.gt(self, other)

    def __le__(self, other):
        langctx = get_langctx()
        return langctx.le(self, other)

    def __lt__(self, other):
        langctx = get_langctx()
        return langctx.lt(self, other)

    def __ne__(self, other):
        langctx = get_langctx()
        return langctx.ne(self, other)

    # TODO Review logical vs bitwise dispatch
    def __or__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_or(self, other)

    def __ror__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_or(other, self)

    def __xor__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_xor(self, other)

    def __rxor__(self, other):
        langctx = get_langctx()
        return langctx.bitwise_xor(other, self)

    #
    # Shift operations
    #

    def __lshift__(self, other):
        langctx = get_langctx()
        return langctx.lshift(self, other)

    def __rlshift__(self, other):
        langctx = get_langctx()
        return langctx.lshift(other, self)

    def __rshift__(self, other):
        langctx = get_langctx()
        return langctx.rshift(self, other)

    def __rrshift__(self, other):
        langctx = get_langctx()
        return langctx.rshift(other, self)

    #
    # Matmul
    #

    def __matmul__(self, other):
        langctx = get_langctx()
        return langctx.matmul(self, other)

    def __rmatmul__(self, other):
        langctx = get_langctx()
        return langctx.matmul(other, self)

    #
    # Transpose
    #

    @property
    def mT(self):
        langctx = get_langctx()
        return langctx.matrix_transpose(self)

    #
    # Real
    #

    @property
    def real(self):
        langctx = get_langctx()
        return langctx.real(self)


#
# Helpers for creating and working with proxies
#

_cls_to_number_proxy_map = {
    float: FloatProxy,
    int: IntegerProxy,
    bool: IntegerProxy,
    complex: ComplexProxy,
}


def numberproxy(cls: type, value: Number | None) -> NumberProxy:
    pcls = _cls_to_number_proxy_map[cls]
    return pcls(value=value)


def is_proxyable(x: Any) -> bool:
    if isinstance(x, Number) and not isinstance(x, NumberProxy):
        return True

    # NOTE The langctx may not have defined the tensor_cls attribute
    #   (the core language context has no associated tensor_cls)
    langctx = langctx_for(x)
    try:
        tensor_cls = langctx.tensor_cls
        return isinstance(x, tensor_cls)
    except AttributeError:
        return False


# TODO Improve type annotation to return type of X or Proxy
# TODO defer to langctx for tensor type -- consider all possible langctxs
# TODO maybe consider what happens when a proxy is passed to this
# TODO handle complex number type
def proxy(x: Any, *, name: str | None = None) -> Any:
    langctx = langctx_for(x)

    tensor_cls = langctx.tensor_cls
    if isinstance(x, tensor_cls):
        return langctx.tensorproxy(name, x)

    if isinstance(x, Number):
        if isinstance(x, complex):
            return ComplexProxy(name=name, value=x)
        if isinstance(x, float):
            return FloatProxy(name=name, value=x)
        if isinstance(x, int):
            return IntegerProxy(name=name, value=x)

        raise NotImplementedError

    return x
