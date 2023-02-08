import operator
import string
from collections import deque
from enum import auto, Enum
from functools import partial, reduce
from numbers import Number

import thunder.core.dtypes as dtypes

from .trace import get_language_context

# This file defines Thunder's most basic proxies, stand-ins for other Python objects that
#   record Python interactions for the tracing context.

# This file depends on trace.py and the dtypes submodule.

__all__ = [
    # Proxies
    "Proxy",
    "NumberProxy",
    "FloatProxy",
    "IntegerProxy",
    "TensorProxy",
    # Proxy helpers and types
    "proxy",
    "make_proxy_name",
]


class Proxy:
    def __init__(self, name):
        self.name = name

    # NOTE: hashing on name (or something like it) is important
    #   The number proxies are subclasses of Python number types,
    #   so they will inherit that __hash__ if another isn't defined.
    #   The hash value for a Python number is its value, and Python
    #   dicts don't distinguish the values 3 and 3.0. This means
    #   that proxies with the same value, even with different types
    #   would hash to the same key, preventing them from being
    #   stored in a dict together.
    def __hash__(self):
        return object.__hash__(self)


class NumberProxy(Proxy):
    def __init__(self, *, name, value, python_type):
        super().__init__(name)
        self.python_type = python_type
        self.value = value

    def __hash__(self):
        return Proxy.__hash__(self)

    # TODO: update these comparisons to support Number x Tensor
    def __eq__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value == other_value

    def __ge__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value >= other_value

    def __gt__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value > other_value

    def __le__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value <= other_value

    def __lt__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value < other_value

    def __ne__(self, other):
        other_value = other.value if isinstance(other, NumberProxy) else other
        return self.value != other_value


# NOTE: Why no bool proxy? Because bool cannot be subclassed. There are no bool
#   instances, just True and False. Further, isinstance(True, int) is True in Python!
#   So bools get handled by IntegerProxy.

# TODO: implement more methods
#   See https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
class IntegerProxy(NumberProxy, int):
    """A proxy integer."""

    def __new__(cls, *, name, value):
        return int.__new__(cls, value)

    def __init__(self, *, name, value):
        # NOTE: bools are also integers in Python
        python_type = bool if isinstance(value, bool) else int
        NumberProxy.__init__(self, name=name, value=value, python_type=python_type)

    def __repr__(self):
        return f"[IntegerProxy name={self.name} value={self.value}]"

    def __hash__(self):
        return Proxy.__hash__(self)

    # NOTE: it'd be nice to define dunders to preserve proxies
    #   across calls to int() and float(), but returning "strict subclasses" of
    #   numbers is deprecated.

    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)


class FloatProxy(NumberProxy, float):
    def __new__(cls, *, name, value):
        return float.__new__(cls, value)

    def __init__(self, *, name, value):
        NumberProxy.__init__(self, name=name, value=value, python_type=float)

    def __repr__(self):
        return f"[FloatProxy name={self.name} value={self.value}]"

    def __hash__(self):
        return Proxy.__hash__(self)

    # NOTE: it'd be nice to define dunders to preserve proxies
    #   across calls to int() and float(), but returning "strict subclasses" of
    #   numbers is deprecated.

    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)


class ComplexProxy(NumberProxy, complex):
    def __new__(cls, *, name, value):
        return complex.__new__(cls, value)

    def __init__(self, *, name, value):
        NumberProxy.__init__(self, name=name, value=value, python_type=complex)

    def __repr__(self):
        return f"[ComplexProxy name={self.name} value={self.value}]"

    def __hash__(self):
        return Proxy.__hash__(self)


# TODO: want this to pass isinstance(p, torch.Tensor) and isinstance(p, np.array) depending on
#   language context
# TODO: maybe change "tensor" param to "like" to be clearer
class TensorProxy(Proxy):
    """A proxy tensor."""

    def __init__(
        self,
        *,
        name,
        tensor=None,
        shape=None,
        device=None,
        dtype=None,
        strides=None,
    ):
        super().__init__(name)

        if dtype is not None:
            assert dtypes.is_dtype(dtype), f"Unknown valuetype={dtype}!"

        if tensor is not None:
            # Pulls metadata from the tensor, but explicit kwargs take precedence
            assert isinstance(tensor, TensorProxy)
            self.shape = tuple(tensor.shape if shape is None else shape)
            self._dtype = tensor.dtype if dtype is None else dtype
            self.device = tensor.device if device is None else device
            self.strides = tensor.strides if strides is None else strides
        else:
            # Requires all metadata, except strides, be specified explicitly
            assert shape is not None
            assert device is not None
            assert dtype is not None

            # Shape is a tuple of integer proxies
            self.shape = tuple(shape)
            self.device = device
            self._dtype = dtype
            self.strides = strides

        self.ndim = len(self.shape)

        # Canonicalizes numbertypes to datatypes
        if dtypes.is_numbertype(self._dtype):
            self._dtype = dtypes.numbertype_to_dtype(self._dtype)

    def __repr__(self):
        return f"[TensorProxy, name={self.name}, shape={self.shape}, dtype={self.dtype}, has_weak_dtype={dtypes.is_weak_dtype(self._dtype)}]"

    def __hash__(self):
        return Proxy.__hash__(self)

    # .dtype, registered using __getattr__
    def _get_dtype(self):
        """Returns the strong variant of the tensor's dtype.

        To acquire the actual dtype use "true_dtype"
        """
        return dtypes.to_strong_dtype(self._dtype)

    def numel(self):
        return reduce(operator.mul, self.shape, 1)

    # +
    def __add__(self, other):
        ctx = get_language_context()
        return ctx.add(self, other)

    # +
    def __radd__(self, other):
        ctx = get_language_context()
        return ctx.add(other, self)

    # ==
    def __eq__(self, other):
        ctx = get_language_context()
        if ctx is not None:
            return ctx.eq(other, self)

        # TODO: consider this out of context behavior
        return super().__eq__(other)

    # <
    def __lt__(self, other):
        ctx = get_language_context()
        return ctx.lt(self, other)

    # *
    def __mul__(self, other):
        ctx = get_language_context()
        return ctx.mul(self, other)

    # *
    def __rmul__(self, other):
        ctx = get_language_context()
        return ctx.mul(other, self)

    # -
    def __sub__(self, other):
        ctx = get_language_context()
        return ctx.sub(self, other)

    # /
    def __truediv__(self, other):
        ctx = get_language_context()
        return ctx.true_divide(self, other)

    # @
    def __matmul__(self, other):
        ctx = get_language_context()
        return ctx.matmul(self, other)

    def __getitem__(self, key):
        ctx = get_language_context()
        return ctx.get_item(self, key)

    # NOTE: If an attribute wasn't found, this assumes the attribute is a method defined
    #  by the language context. Just returning that method wouldn't work, however,
    #  since the TensorProxy, passed as self here, wouldn't be passed through to the
    #  actual method. That's why this partials the returned method.
    def __getattr__(self, name):
        # Handles properties
        if name == "dtype":
            return super().__getattribute__("_get_dtype")()
        if name == "true_dtype":
            return super().__getattribute__("_dtype")
        if name == "strides":
            return super().__getattribute__("strides")

        ctx = get_language_context()
        return partial(getattr(ctx, name), self)


def proxy(x, *, name):
    """Creates a proxy object."""

    if isinstance(x, complex):
        return ComplexProxy(name=name, value=x)
    if isinstance(x, int):
        return IntegerProxy(name=name, value=x)
    if isinstance(x, float):
        return FloatProxy(name=name, value=x)

    raise ValueError(f"Can't proxy unknown type {type(x)}")
