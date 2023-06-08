from numbers import Number
from typing import Type, Optional, Any, Tuple, List, Union
from collections.abc import Sequence
from functools import reduce, partial
import operator
import builtins

from thunder.core.trace import VariableInterface, get_tracectx
from thunder.core.baseutils import ProxyInterface, NumberProxyInterface, TensorProxyInterface
import thunder.core.baseutils as baseutils
from thunder.core.langctx import langctx_for, get_langctx
import thunder.core.devices as devices
import thunder.core.dtypes as dtypes


# TODO Document this class
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
    def __init__(self, name=None):
        trace = get_tracectx()
        if name is None:
            name = trace.make_name()
        else:
            trace.add_name(name)

        self._name = name

    @property
    def name(self):
        return self._name

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return self.__class__(name=name)

    def __repr__(self):
        return f"{self._name}"

    def type_string(self):
        return "Any"


# NOTE NumberProxies are NOT Numbers
# TODO Maybe NumberProxies should be Numbers?
class NumberProxy(Proxy, NumberProxyInterface):
    def __init__(self, name=None, value=None, *, python_type):
        super().__init__(name)
        self.value = value
        self.python_type = python_type

    # NOTE: Python numbers hash to themselves, and this mimics that behavior
    def __hash__(self):
        return hash(self.value)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return self.__class__(name=name, value=self.value, python_type=self.python_type)

    def known_value(self):
        return self.value is not None

    #
    # Elementwise unary operators
    #
    # TODO Track these elementwise operations by adding explicit calls to
    #   the appropriate prims

    def __abs__(self):
        return pyval(self).__abs__()

    def __ceil__(self):
        return pyval(self).__ceil__()

    def __floor__(self):
        return pyval(self).__floor__()

    def __invert__(self):
        return pyval(self).__invert__()

    def __neg__(self):
        return pyval(self).__neg__()

    def __pos__(self):
        return pyval(self).__pos__()

    def __round__(self):
        return pyval(self).__round__()

    def __trunc__(self):
        return pyval(self).__trunc__()

    #
    # Elementwise binary operators
    #

    @staticmethod
    def _elementwise_binary_helper(a, b, name, fn):
        baseutils.check_type(b, (TensorProxy, Number))

        if isinstance(b, TensorProxy):
            langctx = get_langctx()
            tensor_fn = getattr(langctx, name)
            return tensor_fn(a, b)

        a, b = pyval(a), pyval(b)
        baseutils.check(a is not None, lambda: f"Trying to {name} an unknown int", exception_type=AssertionError)
        baseutils.check(
            b is not None, lambda: f"Trying to {name} with an unknown number", exception_type=AssertionError
        )

        return fn(a, b)

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


def pyval(x: Union[NumberProxy, Number]) -> Number:
    baseutils.check_type(x, (NumberProxy, Number))

    # NOTE This has to query NumberProxy, not Number, because NumberProxies are Numbers
    #   (but not all Numbers are NumberProxies)
    if isinstance(x, NumberProxy):
        return x.value

    return x


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

    #
    # dtype conversion operators
    #

    def __complex__(self):
        return self.value

    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value.__int__()

    def __bool__(self):
        raise self.value.__bool__()

    #
    # Shift operations
    #

    def __lshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__lshift__(pyval(other))

        return langctx.lshift(self, other)

    def __rlshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rlshift__(pyval(other))

        return langctx.lshift(other, self)

    def __rshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rshift__(pyval(other))

        return langctx.rshift(self, other)

    def __rrshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rrshift__(pyval(other))

        return langctx.rshift(other, self)

    #
    # Matmul
    #

    def __matmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError


# TODO Review dtype conversions
# TODO Review -9999 as the marker value for unknown values
class IntegerProxy(NumberProxy, int):
    def __new__(cls, *, name=None, value):
        if value is None:
            value = -9999

        return int.__new__(cls, value)

    def __init__(self, name=None, value=None):
        # NOTE bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(self, name=name, value=value, python_type=python_type)

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return IntegerProxy(name=name, value=self.value)

    def type_string(self):
        value_str = f"{self.value}" if self.value is not None else "?"
        return f"int {value_str}"

    def __repr__(self):
        return f"[IntegerProxy name={self.name}, value={self.value}]"

    #
    # dtype conversion operators
    #

    def __complex__(self):
        return self.value.__complex__()

    def __float__(self):
        return self.value.__float__()

    def __int__(self):
        return self.value

    def __bool__(self):
        return self.value.__bool__()

    #
    # Shift operations
    #

    def __lshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return self._elementwise_binary_no_ctx("__lshift__", int.__lshift__, self.value, pyval(other))

        return langctx.lshift(self, other)

    def __rlshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return self._elementwise_binary_no_ctx("__rlshift__", int.__rlshift__, self.value, pyval(other))

        return langctx.lshift(other, self)

    def __rshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return self._elementwise_binary_no_ctx("__rshift__", int.__rshift__, self.value, pyval(other))

        return langctx.rshift(self, other)

    def __rrshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return self._elementwise_binary_no_ctx("__rrshift__", int.__rrshift__, self.value, pyval(other))

        return langctx.rshift(other, self)

    #
    # Matmul
    #

    def __matmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError


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

    #
    # dtype conversion operators
    #

    def __complex__(self):
        return self.value.__complex__()

    def __float__(self):
        return self.value

    def __int__(self):
        return self.value.__int__()

    def __bool__(self):
        return self.value.__bool__()

    #
    # Shift operations
    #

    def __lshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__lshift__(pyval(other))

        return langctx.lshift(self, other)

    def __rlshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rlshift__(pyval(other))

        return langctx.lshift(other, self)

    def __rshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rshift__(pyval(other))

        return langctx.rshift(self, other)

    def __rrshift__(self, other):
        langctx = get_langctx()

        if langctx is None:
            return pyval(self).__rrshift__(pyval(other))

        return langctx.rshift(other, self)

    #
    # Matmul
    #

    def __matmul__(self, other):
        raise NotImplementedError

    def __rmatmul__(self, other):
        raise NotImplementedError


# TODO Add remaining dunders
class TensorProxy(Proxy, TensorProxyInterface):
    def __init__(
        self,
        name=None,
        *,
        like=None,
        shape: Optional[Union[tuple[int, ...], list[int]]] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(name)

        self._device = None
        self._dtype = None
        self._shape = None

        if like is not None:
            baseutils.check_type(like, TensorProxy)
            self._shape = tuple(like.shape)
            self._device = like.device
            self._dtype = like.true_dtype

        self._shape = shape if shape is not None else self._shape
        self._device = device if device is not None else self._device
        self._dtype = dtype if dtype is not None else self._dtype
        self._dtype = dtypes.numbertype_to_dtype(self._dtype) if dtypes.is_numbertype(self._dtype) else self._dtype

        # Computes derived properties
        self.numel = reduce(operator.mul, self.shape, 1)

        # TODO Alias rank to ndim?
        self.ndim = len(self.shape)

        # Validates inputs
        baseutils.check_valid_shape(self._shape)
        baseutils.check_type(self._device, devices.Device)
        baseutils.check_type(self._dtype, dtypes.dtype)

        # NOTE for simplicity functions that want to reason about weak dtypes should explicitly request
        #   the true_dtype property
        self.true_dtype = self._dtype
        self._dtype = dtypes.to_strong_dtype(self._dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def replace_name(self, name):
        """Return a copy of this proxy with the given name."""
        return TensorProxy(name, like=self)

    def type_string(self):
        return f"{self.device} {self.dtype.shortname()}{list(self.shape)}"

    @property
    def size(self):
        langctx = get_langctx()
        return langctx.size(self)

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
        return langctx.invert(self)

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
# Helpers for creating and working with proxies
#

_cls_to_number_proxy_map = {
    float: FloatProxy,
    int: IntegerProxy,
    bool: IntegerProxy,
}


def numberproxy(cls: type, value: Optional[Number]) -> NumberProxy:
    pcls = _cls_to_number_proxy_map[cls]
    return pcls(value=value)


def is_proxyable(x: Any) -> bool:
    if isinstance(x, Number):
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
def proxy(x: Any, *, name=None) -> Any:
    langctx = langctx_for(x)

    try:
        tensor_cls = langctx.tensor_cls
        if isinstance(x, tensor_cls):
            return langctx.tensorproxy(name, x)
    except AttributeError:
        pass

    if isinstance(x, Number):
        if isinstance(x, complex):
            return ComplexProxy(name=name, value=x)
        if isinstance(x, float):
            return FloatProxy(name=name, value=x)
        if isinstance(x, int):
            return IntegerProxy(name=name, value=x)

        raise NotImplementedError

    return x
