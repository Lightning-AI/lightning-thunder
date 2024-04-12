import enum
import sys
from typing import Any
from collections.abc import Callable
from collections.abc import Hashable
from types import ModuleType
import warnings
from functools import cache, partial

import torch.cuda

from thunder.core.utils import check
from thunder.core.symbol import Symbol, BoundSymbol, default_python_printer
from thunder.core.trace import TraceCtx
from thunder.core.proxies import Proxy
from thunder.core.baseutils import run_once

__all__ = [
    "register_executor",
    "deregister_executor",
    "get_all_executors",
    "get_default_executors",
    "get_always_executors",
    "get_executor",
    "set_default_executors",
    "set_always_executors",
    "add_default_executor",
    "add_always_executor",
    "remove_default_executor",
    "remove_always_executor",
]


class ImplInfo:
    def __init__(
        self,
        *,
        symbol: Symbol | None = None,
        checker: Callable | None = None,
        execution_transform: Callable | None = None,
        grad_transform: Callable | None = None,
    ):
        self.symbol: Symbol | None = symbol
        # None implies that the symbol is always executable.
        self.checker: Callable | None = checker
        # None implies that the symbol is called as-is.
        self.execution_transform: Callable | None = execution_transform
        # None implies that the symbol has no grad, or that the grad is in _grad_fn_map.
        self.grad_transform: Callable | None = grad_transform


class Executor:
    def __init__(self, name: Hashable, *, version: None | Any = None):
        self._name: Hashable = name
        self._version = version

        self._implmap: dict[Hashable, ImplInfo] = {}
        self._lookasides: dict[Callable, Callable] = {}

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def version(self) -> Any:
        return self._version

    @property
    def implmap(self) -> dict[Hashable, ImplInfo]:
        return self._implmap

    def __repr__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        return isinstance(other, Executor) and other.name == self.name

    def can_execute_or_fuse(self, bsym: BoundSymbol) -> bool:
        return self.can_execute(bsym) or self.can_fuse(bsym)

    def can_execute(self, bsym: BoundSymbol) -> bool:
        sym = bsym.sym
        impl: None | ImplInfo = self.implmap.get(sym.id, None)

        if impl is None:
            return False

        if impl.checker is None:
            return True

        return impl.checker(*bsym.args, **bsym.kwargs)

    # Returns True when nvFuser can fuse every operation the bound symbol calls, False otherwise
    def can_fuse(self, bsym: BoundSymbol) -> bool:
        sym = bsym.sym

        if sym.is_fusion:
            return False

        if self.can_execute(bsym):
            return True

        if len(bsym.subsymbols) == 0:
            return False

        # Checks if all the operations this calls are executable
        for ssym in bsym.subsymbols:
            if not self.can_fuse(ssym):
                return False

        return True

    def get_execution_transform(self, sym: Symbol) -> None | Callable:
        impl: None | ImplInfo = self.implmap.get(sym.id, None)

        if impl is None:
            return None

        return impl.execution_transform

    def get_grad_transform(self, sym: Symbol) -> None | Callable:
        impl: None | ImplInfo = self.implmap.get(sym.id, None)

        if impl is None:
            return None

        return impl.grad_transform


class FUEL_LEVEL(enum.Enum):
    UNLIMITED = enum.auto()


# To implement a FusionExecutor, create a subclass of it that implements the fusion_pass method
class FusionExecutor(Executor):
    def __init__(self, name: Hashable, *, version: None | Any = None):
        super().__init__(name, version=version)

    # Optimization fuel is a concept introduced by David B.
    # Whalley (https://dl.acm.org/doi/pdf/10.1145/186025.186103) to isolate
    # compiler bugs.
    #
    # A FusionExecutor keeps track of its own optimization fuel as the number
    # of remaining optimizations it can do. Each fusion pass can call
    # `get_fuel` to acquire a certain amount of optimization fuel, and, only if
    # it returns true, perform the actual optimization. `set_fuel` is used by
    # the test or the user to fuel the executor to a certain level.
    #
    # I used this to isolate
    # https://github.com/NVIDIA/Fuser/issues/1667 from a numbers-mismatch error
    # happened in test_grad.py. I binary-searched for the smallest optimization
    # fuel that reproduced the error, and the last fusion generated in that run
    # was the culprit. See test_nvfuser.py::test_optimization_fuel for an example.
    def get_fuel(self, amount: int = 1, /) -> bool:
        raise NotImplementedError

    def set_fuel(self, value: int | FUEL_LEVEL):
        raise NotImplementedError

    def fusion_pass(self, trace: TraceCtx) -> TraceCtx:
        raise NotImplementedError

    def fuse(self, region: "Region", fusion_counter: int) -> BoundSymbol:  # type: ignore (circular import)
        raise NotImplementedError

    def register_supported(
        self,
        sym_or_id: Symbol | Hashable,
        checker: None | Callable = None,
        *,
        execution_transform: None | Callable = None,
        grad_transform: None | Callable = None,
    ):
        impl = ImplInfo(checker=checker, execution_transform=execution_transform, grad_transform=grad_transform)

        _id = sym_or_id.id if isinstance(sym_or_id, Symbol) else sym_or_id
        self.implmap[_id] = impl

    def register_temporary_operation(
        self, name: str, fn: Callable, *, inputs: list[Proxy], outputs: list[Proxy], bsyms: list[BoundSymbol]
    ) -> BoundSymbol:
        def _meta(*args):
            return tuple(outputs)

        def _bind_postprocess(bsym: BoundSymbol) -> None:
            bsym.subsymbols = tuple(bsyms)
            bsym._call_ctx = {name: fn}

        sym = Symbol(name=name, meta=_meta, is_fusion=True, _bind_postprocess=_bind_postprocess, executor=self)
        return sym.bind(*inputs, output=outputs)


class OperatorExecutor(Executor):
    def __init__(self, name: Hashable, *, version: None | Any = None):
        super().__init__(name, version=version)

        self._opmap: dict[str, Symbol] = {}

    @property
    def opmap(self) -> dict[str, Symbol]:
        return self._opmap

    # TODO Document this operation
    # TODO Wrap meta in prim context?
    # TODO Document how to avoid name collisions
    def register_operator(
        self,
        name: str,
        *,
        like: None | Callable = None,
        meta: None | Callable = None,
        tags: None | list[Any] = None,
        module: None | type | ModuleType = None,
        fn: None | Callable = None,
        bind_postprocess: None | Callable = None,
        replaces: None | Callable = None,
        python_printer: Callable = default_python_printer,
    ) -> Symbol:
        ln = like is None
        mn = meta is None
        assert (
            ln ^ mn
        ), f"Expected one and only one of 'like' and 'meta' to be specified. {'Neither' if ln and mn else 'Both'} were specified."
        assert (module is not None) + (
            fn is not None
        ) <= 2, f"Expected one and only one of 'module' or 'fn' to be specified. Module: {module}, Fn: {fn}"

        # NOTE Directly specifying a meta function makes the operation a prim
        is_prim = meta is not None
        # Set tags to be the same as 'like' if 'tags' is not specified
        tags = like.tags if (tags is None and like is not None and hasattr(like, "tags")) else tags
        meta = meta if meta is not None else like
        call_ctx: None | dict[str, Callable] = None if fn is None else {name: fn}

        def _bind_postprocess(bsym: BoundSymbol) -> None:
            bsym._call_ctx = call_ctx
            if bind_postprocess is not None:
                bind_postprocess(bsym)

        sym = Symbol(
            name=name,
            id=name,
            meta=meta,
            is_prim=is_prim,
            _module=module,
            executor=self,
            _bind_postprocess=_bind_postprocess,
            python_printer=python_printer,
            tags=tags,
        )
        self.opmap[name] = sym

        if replaces is not None:
            self._lookasides[replaces] = sym

        return sym

    def register_implementation(
        self,
        sym_or_id: Symbol | Hashable,
        op: None | Symbol = None,
        *,
        checker: None | Callable = None,
        execution_transform: None | Callable = None,
        grad_transform: None | Callable = None,
    ):
        if execution_transform is None:
            assert op is not None

        impl = ImplInfo(
            symbol=op, checker=checker, execution_transform=execution_transform, grad_transform=grad_transform
        )

        _id = sym_or_id.id if isinstance(sym_or_id, Symbol) else sym_or_id
        self.implmap[_id] = impl


def single_op_executor(
    exc_name: Hashable,
    op_name: str,
    *,
    version: None | Any = None,
    replaces: Callable | None = None,
    like: None | Callable = None,
    meta: None | Callable = None,
    module: None | type | ModuleType = None,
    fn: None | Callable = None,
    checker: None | Callable = None,
    execution_transform: None | Callable = None,
    grad_transform: None | Callable = None,
) -> OperatorExecutor:
    """
    Creates a new OperatorExecutor, registers it with thunder, and registers a new operator with it,
    implemented with fn. Also registers the implementation of the operator with the executor.
    """
    exc = OperatorExecutor(exc_name, version=version)
    register_executor(exc)

    if replaces is None:
        replaces = fn

    sym = exc.register_operator(
        op_name,
        replaces=replaces,
        like=like,
        meta=meta,
        module=module,
        fn=fn,
    )

    exc.register_implementation(
        sym, op=sym, checker=checker, execution_transform=execution_transform, grad_transform=grad_transform
    )

    return exc


# Creates common datastructures
_executor_map: dict[Hashable, Executor] = {}
_default_executors: list[Executor] = []
_always_executors: list[Executor] = []


# Registers a new executor with thunder
# Either accepts one of the executor classes above, or the components necessary to create one
def register_executor(ex: Executor) -> Executor:
    _executor_map[ex.name] = ex
    return ex


def get_all_executors() -> tuple[Executor, ...]:
    # manually import all native executors to let them register themselves
    from thunder.executors import (
        apex_entropyex,
        cudnn_layernormex,
        cudnnex,
        nvfuserex,
        pythonex,
        sdpaex,
        torch_compile,
        torchex,
        transformer_engineex,
        triton_crossentropy,
    )

    return tuple(_executor_map.values())


def get_default_executors() -> tuple[Executor, ...]:
    return tuple(_default_executors)


def get_always_executors() -> tuple[Executor, ...]:
    return tuple(_always_executors)


def get_executor(name: Any) -> None | Executor:
    ex: Executor
    for ex in get_all_executors():
        if ex.name == name:
            return ex

    return None


def set_default_executors(defaults: list[Executor]) -> list[Executor]:
    global _default_executors

    _default_executors = defaults
    return _default_executors


def set_always_executors(always: list[Executor]) -> list[Executor]:
    global _always_executors

    _always_executors = always
    return _always_executors


def add_default_executor(ex: Executor) -> list[Executor]:
    global _default_executors

    _default_executors = [ex] + _default_executors
    return _default_executors


def add_always_executor(ex: Executor) -> list[Executor]:
    global _always_executors

    _always_executors = [ex] + _always_executors
    return _always_executors


def remove_default_executor(ex: Hashable | Executor) -> list[Executor]:
    global _default_executors

    _id = ex.name if isinstance(ex, Executor) else ex
    _default_executors = list([x for x in _default_executors if x.name != _id])
    return _default_executors


def remove_always_executor(ex: Hashable | Executor) -> list[Executor]:
    global _always_executors

    _id = ex.name if isinstance(ex, Executor) else ex
    _always_executors = list([x for x in _always_executors if x.name != _id])
    return _always_executors


# Deregisters an executor -- removing it from default and always lists
def deregister_executor(ex: Hashable | Executor) -> None:
    _id: Hashable = ex.name if isinstance(ex, Executor) else ex

    if _id in _executor_map:
        del _executor_map[_id]

    remove_always_executor(_id)
    remove_default_executor(_id)
