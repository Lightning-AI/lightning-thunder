import warnings

from collections.abc import Callable

from lightning_utilities.core.imports import package_available

from thunder import Transform
from thunder.extend import StatefulExecutor
from thunder.core.trace import TraceCtx

__all__ = ["transformer_engine_ex", "TransformerEngineTransform", "_te_activation_checkpointing_transform"]

transformer_engine_ex: None | StatefulExecutor = None
TransformerEngineTransform: None | Transform = None
_te_activation_checkpointing_transform: None | Callable[[TraceCtx], TraceCtx] = None


if package_available("transformer_engine"):
    import thunder.executors.transformer_engineex_impl as impl

    transformer_engine_ex = impl.transformer_engine_ex
    TransformerEngineTransform = impl.TransformerEngineTransform
    _te_activation_checkpointing_transform = impl._te_activation_checkpointing_transform

else:
    warnings.warn("transformer_engine module not found!")
