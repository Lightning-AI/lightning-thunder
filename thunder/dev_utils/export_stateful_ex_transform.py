import weakref
from collections.abc import Callable
from typing import List, Tuple

from thunder.core.transform_common import (
    Transform,
)
from thunder.core.trace import TraceCtx as Trace
from thunder.core.module import ThunderModule


class ExportStatefulExecutorsStats:
    def __init__(self, tm: ThunderModule, resolver_fn: Callable):
        """Lightweight accessor attached to a `ThunderModule`.

        Args:
            tm: The `ThunderModule` instance this accessor belongs to.
            resolver_fn: A callable that knows how to resolve the recorded
                references on `tm` and return real values.
        """
        self.tm = tm
        self.resolver_fn = resolver_fn


class ExportStatefulExecutorsTransform(Transform):
    """Register references and resolve runtime state lazily.

    What this transform does:
    - Singleton registry to plug per-executor exporters
    - At module transform time, installs a lightweight accessor on the module
      (e.g., `module.te_fp8_states`) that can resolve values on demand
    - At post-optimization time, calls registered reference callbacks to record
      only where values will materialize (holders + attribute paths). No data
      are copied or materialized in this step
    - When code calls the accessor (e.g., `module.te_fp8_states()`), the resolve
      callback reads the recorded references and returns the latest values


    API overview:
    - register_ref_callback(name, register_cb, resolve_cb, instance_cls):
      name: attribute name to attach on the module
      register_cb(trace, module): store references from the trace/python_ctx
      resolve_cb(module): materialize and return values using the stored refs
      instance_cls: a small class constructed as instance_cls(module, resolve_cb)
        and attached as `setattr(module, name, instance)`; it typically stores
        containers for references and implements __call__(...) to resolve

    Usage:
    1) Register once at import/init time. For example, for TransformerEngine:
       ExportStatefulExecutorsTransform.register_ref_callback(
           "te_fp8_states", register_cb, resolve_cb, StatsClass
       )
    2) Enable at compile time:
       thunder.jit(model, executors=[...], transforms=[..., ExportStatefulExecutorsTransform()])
    3) After each run, call `module.te_fp8_states()` to resolve and return the latest values.

    Notes:
    - Supports multiple ThunderModule instances (e.g., subgraphs)
    - Callback errors are swallowed to avoid interfering with execution
    """

    _register_callbacks: dict[str, Callable] = {}
    _callback_attributes: List[Tuple[str, type[ExportStatefulExecutorsStats], Callable]] = []

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure singleton instance across repeated transform construction."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize internal weakrefs registry.

        ThunderCompiler and other compilation flows may create multiple
        `ThunderModule` instances for subgraphs; we keep weak references
        to update all of them during post-optimization registration.
        """
        self.tm_refs = []

    @classmethod
    def register_ref_callback(
        cls, name: str, callback: Callable, resolve_cb: Callable, instance: type[ExportStatefulExecutorsStats]
    ) -> None:
        """Register per-executor reference and resolver callbacks.

        Installs a module attribute named `name` by constructing `instance` with
        the resolver function. The `callback` will be invoked during
        post-optimization to record reference locations on the module.

        Args:
            name: Module attribute to attach (e.g., "te_fp8_states").
            callback: Function `(trace, module) -> None` that records refs.
            resolve_cb: Function `(module) -> Any` that resolves values on demand.
            instance: A class (must be a subclass of ExportStatefulExecutorsStats) constructed as `instance(module, resolve_cb)`.
        """
        if not issubclass(instance, ExportStatefulExecutorsStats):
            raise TypeError(f"Provided instance {instance} must be a subclass of ExportStatefulExecutorsStats")
        cls._register_callbacks[name] = callback
        cls._callback_attributes.append((name, instance, resolve_cb))

    def transform_module(self, model) -> None:
        assert model is not None
        # Cache a weakref to the ThunderModule for later runtime export
        self.tm_refs.append(weakref.ref(model))
        # Initialize attributes on model
        for name, instance, resolve_cb in self._callback_attributes:
            setattr(model, name, instance(model, resolve_cb))

    def transform_trace_post_optimization(self, computation_trace: Trace, **kwargs):
        for tm_ref in self.tm_refs:
            # Resolve ThunderModule from weakref; if unavailable, skip
            tm = tm_ref() if tm_ref is not None else None
            if tm is None:
                continue

            # Invoke all registered callbacks to register reference locations
            for _, cb in self._register_callbacks.items():
                try:
                    cb(computation_trace, tm)
                except Exception:
                    pass
        return computation_trace
