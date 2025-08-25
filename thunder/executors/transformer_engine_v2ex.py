import warnings

from collections.abc import Callable

from lightning_utilities.core.imports import package_available

from thunder import Transform
from thunder.extend import StatefulExecutor
from thunder.core.trace import TraceCtx

__all__ = ["transformer_engine_v2_ex", "TransformerEngineTransformV2", "_te_activation_checkpointing_transform"]

transformer_engine_v2_ex: None | StatefulExecutor = None
TransformerEngineTransformV2: None | Transform = None
_te_activation_checkpointing_transform: None | Callable[[TraceCtx], TraceCtx] = None

if package_available("transformer_engine"):
    import thunder.executors.transformer_engine_v2ex_impl as impl

    transformer_engine_v2_ex = impl.transformer_engine_v2_ex
    TransformerEngineTransformV2 = impl.TransformerEngineTransformV2
    _te_activation_checkpointing_transform = impl._te_activation_checkpointing_transform

else:
    warnings.warn("transformer_engine module not found!")
