from collections.abc import Callable, Sequence, Hashable
import enum
import itertools
import os
import sys
from types import ModuleType
from typing import Any

import torch.cuda

from thunder.core.devices import to_torch_device
from thunder.core.dtypes import to_torch_dtype
from thunder.core.proxies import Proxy, TensorProxy, proxy
from thunder.core.pytree import tree_map
from thunder.core.symbol import Symbol, BoundSymbol, default_python_printer
from thunder.core.trace import TraceCtx
from thunder.core.vjp_utils import disable_caching_split_forward_and_backward

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
        self._opmap: dict[str, Symbol] = {}

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def version(self) -> Any:
        return self._version

    @property
    def implmap(self) -> dict[Hashable, ImplInfo]:
        return self._implmap

    @property
    def opmap(self) -> dict[str, Symbol]:
        return self._opmap

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__name__}('{str(self.name)}')"

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

    def get_execution_transform(self, bsym: BoundSymbol) -> None | Callable:
        sym = bsym.sym

        impl: None | ImplInfo = self.implmap.get(sym.id, None)

        if impl is None:
            return None

        if not (impl.checker is None or impl.checker(*bsym.args, **bsym.kwargs)):
            return None

        return impl.execution_transform

    def get_grad_transform(self, sym: Symbol) -> None | Callable:
        impl: None | ImplInfo = self.implmap.get(sym.id, None)

        if impl is None:
            return None

        return impl.grad_transform

    # TODO Document this operation
    # TODO Wrap meta in prim context?
    # TODO Document how to avoid name collisions
    def register_operator(
        self,
        name: str,
        *,
        like: None | Symbol = None,
        meta: None | Callable = None,
        tags: None | list[Any] = None,
        module: None | type | ModuleType = None,
        fn: None | Callable = None,
        bind_postprocess: None | Callable = None,
        replaces: None | Callable | tuple[Callable, ...] = None,
        python_printer: Callable = default_python_printer,
    ) -> Symbol:
        ln = like is None
        mn = meta is None
        assert ln ^ mn, (
            f"Expected one and only one of 'like' and 'meta' to be specified. {'Neither' if ln and mn else 'Both'} were specified."
        )
        assert (module is not None) + (fn is not None) <= 2, (
            f"Expected one and only one of 'module' or 'fn' to be specified. Module: {module}, Fn: {fn}"
        )

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

        if callable(replaces):
            replaces = (replaces,)

        self._lookasides.update({replace: sym for replace in replaces or []})

        return sym


class FUEL_LEVEL(enum.Enum):
    UNLIMITED = enum.auto()


# To implement a FusionExecutor, create a subclass of it that implements the fusion_pass method
class FusionExecutor(Executor):
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
            bsym._call_ctx = {name: fn}

        sym = Symbol(name=name, meta=_meta, is_fusion=True, _bind_postprocess=_bind_postprocess, executor=self)
        return sym.bind(*inputs, output=outputs, subsymbols=tuple(bsyms))


class OperatorExecutor(Executor):
    def __init__(self, name: Hashable, *, version: None | Any = None):
        super().__init__(name, version=version)

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


class StatefulExecutor(OperatorExecutor):
    """
    A specialized executor for operators that need to keep a state for as long as the trace lifetime.

    This executor is able to register operators and bind them to their state class.
    It uses a counter to keep operator names unique.

    Key Features:
        - Creates and binds state object to operators.
        - Gives unique naming to the operators by using a counter.
        - Disables cacheing of it's symbols' `grad_transform`.

    Example:
        >>> executor = StatefulExecutor()
        >>> op = executor.register_operator(
        ...     name="temp_add",
        ...     StateClass,
        ...     [...same arguments that would have been passed to register_operator...]
        ... )

    For more examples check out transformer engine executor.
    """

    def __init__(self, name: Hashable, *, version: Any | None = None):
        super().__init__(name, version=version)
        self._counter: int = 0

    def register_stateful_operator(self, name: str, state_class, *, meta) -> Callable[..., Any]:
        """
        Sets up a function that creates and registers new symbol for the given operator name and state class.
        The output function also binds an instance of the state class to the bound symbol call context.

        Args:
            name: The name of the operator. Note that it is going to be combined with the counter number to generate a unique name.
            state_class: The name of a user defined class that implements a __call__ method that returns the
                operator output (i.e. the value of the state).
            meta: The meta function for the operator.

        Returns:
            out: Instance of `create_new_symbol_and_call` function that register the operator and binds the state object upon call.
        """

        def create_new_symbol_and_call(*args, **kwargs):
            op_name = f"{name}_{self._counter}"

            def bind_state(bsym):
                bsym._call_ctx = {op_name: state_class()}

            sym: Symbol = self.register_operator(op_name, meta=meta, bind_postprocess=bind_state)

            self._counter += 1
            return sym(*args, **kwargs)

        return create_new_symbol_and_call

    def get_grad_transform(self, sym: Symbol):
        """
        Since every call to the symbol is unique, this function disables cacheing on the symbol's grad transform.

        Args:
            sym: Symbol queried for grad transform.

        Returns:
            Callable | None: a callable object implementing grad transform if it exists.
        """
        grad_transform: Any | None = super().get_grad_transform(sym)
        if grad_transform:
            # Always disable cache for stateful grad transform
            return disable_caching_split_forward_and_backward(grad_transform)
        return grad_transform


class TemporaryExecutor(OperatorExecutor):
    """
    A specialized executor for managing temporary operator registrations at runtime.

    This executor generates unique identifiers for each registered operator by combining
    the operator name with instance-specific identifiers. It's designed for scenarios
    requiring dynamic operator registration without conflicting with existing operations.

    Key Features:
        - Creates unique operator names using instance ID and counter
        - Supports runtime registration of operators
        - Handles opaque function registration
        - Maintains isolation between different temporary registrations

    Example:
        >>> executor = TemporaryExecutor()
        >>> op = executor.register_operator(
        ...     name="temp_add",
        ...     like=thunder.torch.add,
        ...     fn=lambda x, y: x + y
        ... )

    Note:
        Operators registered through this executor are intended for temporary use
        and should not be relied upon for permanent implementations.
    """

    def __init__(self):
        super().__init__(f"__ad_hoc_executor_{id(self)}")
        self.counter = 0  # a counter to disambiguate names

    def register_operator(
        self,
        name: str,
        *,
        like: None | Symbol = None,
        meta: None | Callable = None,
        tags: None | list[Any] = None,
        module: None | type | ModuleType = None,
        fn: None | Callable = None,
        bind_postprocess: None | Callable = None,
        replaces: None | Callable = None,
        python_printer: Callable = default_python_printer,
    ) -> Symbol:
        op = super().register_operator(
            f"{name}_{id(self)}_{self.counter}",
            like=like,
            meta=meta,
            tags=tags,
            module=module,
            fn=fn,
            bind_postprocess=bind_postprocess,
            replaces=replaces,
            python_printer=python_printer,
        )
        self.counter += 1
        return op

    def register_operator_for_opaque_function(self, fn):
        def get_name(fn):
            mod = sys.modules[fn.__module__]
            if getattr(mod, fn.__name__) == fn:
                return f"{fn.__module__}.{fn.__name__}"
            raise RuntimeError("could not find name")

        def auto_meta(fn):
            def meta(*args, **kwargs):
                concrete_args = []
                for a in args:
                    if isinstance(a, TensorProxy):
                        dtype = to_torch_dtype(a.dtype)
                        if dtype.is_floating_point:
                            t = torch.randn(
                                a.shape,
                                dtype=dtype,
                                device=to_torch_device(a.device),
                                requires_grad=a.requires_grad,
                            )
                        else:
                            t = torch.ones(
                                a.shape,
                                dtype=dtype,
                                device=to_torch_device(a.device),
                                requires_grad=a.requires_grad,
                            )
                        concrete_args.append(t)
                    else:
                        concrete_args.append(a)
                results = fn(*concrete_args, **kwargs)
                meta_results = tree_map(lambda r: proxy(r) if isinstance(r, torch.Tensor) else r, results)
                return meta_results

            return meta

        name = get_name(fn).replace(".", "_")
        meta = auto_meta(fn)
        symbol = self.register_operator(name, fn=fn, meta=meta, replaces=fn)
        return symbol

    def __repr__(self) -> str:
        return f"<thunder.extend.TemporaryExecutor object {id(self)}>"


def single_op_executor(
    exc_name: Hashable,
    op_name: str,
    fn: Callable,
    meta: Callable,
    *,
    version: None | Any = None,
    replaces: Callable | None = None,
    tags: None | list[Any] = None,
    checker: None | Callable = None,
    execution_transform: None | Callable = None,
    grad_transform: None | Callable = None,
) -> OperatorExecutor:
    """
    Creates a new OperatorExecutor, registers it with thunder, and registers a new operator with it,
    implemented with fn. Also registers the implementation of the operator with the executor.

    If the operator needs a backward, you should provide a `grad_transform` function.

    Args:
        exc_name: The name of the executor.
        op_name: The name of the operator created in the executor.
        fn: The function that implements the operator.
        meta: The meta function for the operator. Meta functions are the functions that the interpreter
              uses to trace with. The meta function takes the same arguments as `fn`, with the exception
              that the objects passed to it are proxied, and it should return proxy (`TensorProxy`) objects
              with the appropriate metadata, identical to the way that `fn` would return them for those given inputs.

        version: The version of the executor. Defaults to None.
        replaces: The function that you call in your code, which `fn` replaces. For example, if you have a
                  custom sdpa kernel, you could replace torch.nn.functional.scaled_dot_product_attention
                  with it to run your benchmarks without code changes.
        checker: The checker function for the operator. If you're using a cuda kernel for example, you have
                 the option to assert that all input tensors are on the same cuda device.
        execution_transform: The execution transform function of the operator.
        grad_transform: The grad transform function of the operator.
    """
    exc = OperatorExecutor(exc_name, version=version)
    register_executor(exc)

    if replaces is None:
        replaces = fn

    sym = exc.register_operator(
        op_name,
        fn=fn,
        replaces=replaces,
        meta=meta,
        tags=tags,
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
        apexex,
        cudnn_layernormex,
        cudnnex,
        nvfuserex,
        pythonex,
        sdpaex,
        fa3ex,
        torch_compile,
        torchex,
        transformer_engineex,
        transformer_engine_v2ex,
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


def resolve_executors(executors: None | Sequence[Executor | str]) -> tuple[Executor, ...]:
    """
    Look up registered executors by name. If executors is None, return the default executors.
    """
    if executors is None:
        return get_default_executors()

    failed_executors: list[str] = []
    resolved_executors: list[Executor] = []
    for e in executors:
        if isinstance(e, str):
            ex = get_executor(e)
            if not ex:
                failed_executors.append(e)
                continue
            else:
                resolved_executors.append(ex)
        elif not isinstance(e, Executor):
            raise ValueError(f"An object of type {type(e)} is not a valid Executor. Executor list: {executors}")
        else:
            resolved_executors.append(e)

    if failed_executors:
        raise ValueError(
            f"Expected an Executor or the name of a registered Executor, instead got: {failed_executors[0] if len(failed_executors) == 1 else failed_executors}"
            + os.linesep
            + f"Registered executors: {get_all_executors()}"
        )

    return tuple(resolved_executors)


def add_executor_lists(
    exc_list: None | Sequence[Executor | str], other_exc_list: None | Sequence[Executor | str]
) -> Sequence[Executor]:
    new_exc_list = []
    exc_list = resolve_executors(exc_list)
    other_exc_list = resolve_executors(other_exc_list)
    for exc in itertools.chain(exc_list, other_exc_list):
        if exc not in new_exc_list:
            new_exc_list.append(exc)

    return new_exc_list
