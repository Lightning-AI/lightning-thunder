from numbers import Number
from typing import Any
from collections.abc import Callable

import numpy as np

from thunder.core.langctx import langctx, Languages
from thunder.numpy.langctx import register_method

from thunder.core.proxies import TensorProxy
from thunder.core.symbol import Symbol
import thunder.clang as clang


#
# NumPy operator definitions
#
# NOTE NumPy language support is demonstrative. PRs extending it are welcome!


# Decorator that sets the language context and constructs a Symbol for each function
class npsymbol:
    def __init__(self, *, method_name: None | str = None):
        self.method_name: None | str = method_name

    def __call__(self, fn: Callable) -> Symbol:
        _fn = langctx(Languages.NUMPY)(fn)
        # TODO: register _fn as opaque with the interpreter or do this in jit_ext?
        sym = Symbol(name=fn.__name__, meta=_fn)

        if self.method_name is not None:
            register_method(self.method_name, _fn)

        return sym


#
# Tensor properties
#


# NOTE Named `compute_len` so that it doesn't conflict with built-in `len`
def compute_len(a: TensorProxy, /) -> int:
    return a.shape[0]


register_method("len", compute_len)


def size(a: TensorProxy, /) -> int:
    return a.numel


register_method("size", size)


#
# Elementwise binary operators
#


# TODO Create a factory that adds ufunc support to elementwise operations
npsymbol(method_name="add")


def add(a: Number | TensorProxy, b: Number | TensorProxy, /, *, where: None | Number | TensorProxy = None):
    result = clang.add(a, b)
    if where is not None:
        return clang.where(where, result, a)
    return result
