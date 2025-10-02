import weakref
from typing import Callable, Dict

from thunder.core.transform_common import (
    Transform,
)
from thunder.core.trace import TraceCtx as Trace


class ExportStatefulExecutorsTransform(Transform):
    """Export runtime state from stateful executors after a trace executes.

    - Singleton transform with a registry of export callbacks
    - Register via `register_export_callback(name, callback)`
    - Callbacks receive `(computation_trace, thunder_module)` and may attach
      serialized state to the module (e.g., `module.te_fp8_stats`)
    - Safe: callback errors are swallowed; export never blocks execution

    Example (TransformerEngine): a callback collects FP8 amax/scale and
    quantizer metadata from `python_ctx` and records them under
    `module.te_fp8_stats = {"forward": [...], "backward": [...]}`.

    Usage:
    1) Register once at import/init time:
       ExportStatefulExecutorsTransform.register_export_callback("my_exec", my_cb)
    2) Enable at compile time:
       thunder.jit(model, executors=[...], transforms=[..., ExportStatefulExecutorsTransform()])
    3) Read exported fields from the compiled module in tests/tools.
    """

    _instance = None
    _callbacks: Dict[str, Callable] = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.tm_ref = None

    @classmethod
    def register_export_callback(cls, name: str, callback: Callable) -> None:
        cls._callbacks[name] = callback

    def transform_module(self, model) -> None:
        # Cache a weakref to the ThunderModule for later runtime export
        self.tm_ref = weakref.ref(model)

    def transform_trace_post_execution(self, computation_trace: Trace, **kwargs):
        # Resolve ThunderModule from weakref; if unavailable, skip
        tm = self.tm_ref() if self.tm_ref is not None else None
        if tm is None:
            return computation_trace

        # Invoke all registered export callbacks.
        for _, cb in self._callbacks.items():
            try:
                cb(computation_trace, tm)
            except Exception:
                # Swallow errors from individual exporters to avoid breaking execution.
                pass

        return computation_trace
