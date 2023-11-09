from typing import Any, Callable, Hashable
from types import ModuleType

from thunder.core.utils import check
from thunder.core.symbol import Symbol, BoundSymbol
from thunder.core.trace import TraceCtx
from thunder.core.proxies import Proxy

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
        symbol: None | Symbol = None,
        checker: Callable,
        execution_transform: None | Callable = None,
        grad_transform: None | Callable = None,
    ):
        self.symbol: Symbol = symbol
        self.checker: Callable = checker
        self.execution_transform: None | Callable = execution_transform
        self.grad_transform: None | Callable = grad_transform


class Executor:
    def __init__(self, name: Hashable, *, version: None | Any = None):
        self._name: Hashable = name
        self._version = version

        self._implmap: dict[Hashable, ImplInfo] = {}

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
        return self.name

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

        return impl.checker(*bsym.args, **bsym.kwargs)

    # Returns True when nvFuser can fuse every operation the bound symbol calls, False otherwise
    def can_fuse(self, bsym: BoundSymbol) -> bool:
        sym = bsym.sym

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


# To implement a FusionExecutor, create a subclass of it that implements the fusion_pass method
class FusionExecutor(Executor):
    def __init__(self, name: Hashable, *, version: None | Any = None):
        super().__init__(name, version=version)

    def fusion_pass(trace: TraceCtx) -> TraceCtx:
        raise NotImplementedError

    def register_supported(
        self,
        sym_or_id: Symbol | Hashable,
        checker=Callable,
        *,
        execution_transform: None | Callable = None,
        grad_transform: None | Callable = None,
    ):
        impl = ImplInfo(checker=checker, execution_transform=execution_transform, grad_transform=grad_transform)

        id = sym_or_id.id if isinstance(sym_or_id, Symbol) else sym_or_id
        self.implmap[id] = impl

    def register_temporary_operation(
        self, name: str, fn: Callable, *, inputs: list[Proxy], outputs: list[Proxy], bsyms: list[BoundSymbol]
    ) -> BoundSymbol:
        def _meta(*args):
            return tuple(outputs)

        def _bind_postprocess(bsym: BoundSymbol) -> None:
            bsym.subsymbols = tuple(bsyms)

        ctx: dict[str, Callable] = {name: fn}
        sym = Symbol(
            name=name, meta=_meta, is_fusion=True, call_ctx=ctx, _bind_postprocess=_bind_postprocess, executor=self
        )
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
        module: None | type | ModuleType = None,
        fn: None | Callable = None,
    ) -> Symbol:
        assert (like is None) ^ (meta is None), "Expected one and only one of 'like' and 'meta' to be specified"
        assert (module is not None) + (
            fn is not None
        ) <= 2, "Expected one and only one of 'module' or 'fn' to be specified"

        # NOTE Directly specifying a meta function makes the operation a prim
        is_prim = meta is not None
        meta = meta if meta is not None else like
        call_ctx: None | dict[str, Callable] = None if fn is None else {name: fn}

        sym = Symbol(name=name, id=name, meta=meta, is_prim=is_prim, _module=module, call_ctx=call_ctx, executor=self)
        self.opmap[name] = sym

        return sym

    def register_implementation(
        self,
        sym_or_id: Symbol | Hashable,
        op: None | Symbol = None,
        *,
        checker=Callable,
        execution_transform: None | Callable = None,
        grad_transform: None | Callable = None,
    ):
        if execution_transform is None:
            assert op is not None

        impl = ImplInfo(
            symbol=op, checker=checker, execution_transform=execution_transform, grad_transform=grad_transform
        )

        id = sym_or_id.id if isinstance(sym_or_id, Symbol) else sym_or_id
        self.implmap[id] = impl


# Creates common datastructures
_executor_map: dict[Hashable, Executor] = {}
_default_executors: list[Executor] = []
_always_executors: list[Executor] = []


# Registers a new executor with thunder
# Either accepts one of the executor classes above, or the components necessary to create one
def register_executor(
    ex: Hashable | Executor, *, opmap: None | dict = None, fusion_pass: None | Callable = None
) -> Executor:
    ex_: Executor
    if isinstance(ex, Executor):
        assert opmap is None and fusion_pass is None
        ex_ = ex
    else:
        # NOTE isinstance(ex, Executor) is False
        # Assumes ex is an id describing an executor
        assert opmap is not None or fusion_pass is not None

        if fusion_pass is not None:
            ex_ = FusionExecutor(ex, opmap=opmap, fusion_pass=fusion_pass)
        else:
            # NOTE opmap is not None
            ex_ = OperatorExecutor(ex, opmap=opmap)

    _executor_map[ex_.name] = ex_
    return ex_


def get_all_executors() -> list[Executor]:
    return list(_executor_map.values())


def get_default_executors() -> list[Executor]:
    return _default_executors


def get_always_executors() -> list[Executor]:
    return _always_executors


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

    id = ex.name if isinstance(ex, Executor) else ex
    _default_executors = list([x for x in _default_executors if x.name != id])
    return _default_executors


def remove_always_executor(ex: Hashable | Executor) -> list[Executor]:
    global _always_executors

    id = ex.name if isinstance(ex, Executor) else ex
    _always_executors = list([x for x in _always_executors if x.name != id])
    return _always_executors


# Deregisters an executor -- removing it from default and always lists
def deregister_executor(ex: Hashable | Executor) -> None:
    id: Hashable = ex.name if isinstance(ex, Executor) else ex

    if id in _executor_map:
        del _executor_map[id]

    remove_always_executor(id)
    remove_default_executor(id)
