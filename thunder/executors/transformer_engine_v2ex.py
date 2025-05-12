import warnings

from lightning_utilities.core.imports import package_available

from thunder import Transform
from thunder.extend import StatefulExecutor

__all__ = ["transformer_engine_v2_ex", "TransformerEngineTransform"]

transformer_engine_v2_ex: None | StatefulExecutor = None
TransformerEngineTransform: None | Transform = None

if package_available("transformer_engine"):
    import thunder.executors.transformer_engine_v2ex_impl as impl

    transformer_engine_v2_ex = impl.functional_te_ex
    TransformerEngineTransform = impl.TransformerEngineTransform

else:
    warnings.warn("transformer_engine module not found!")
