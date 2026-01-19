import warnings

from lightning_utilities.core.imports import package_available

from thunder import Transform
from thunder.extend import OperatorExecutor

__all__ = ["tilegym_ex", "TileGymTransform"]


tilegym_ex: None | OperatorExecutor = None
TileGymTransform: None | Transform = None


if package_available("tilegym"):
    import thunder.executors.tilegymex_impl as impl

    tilegym_ex = impl.tilegym_ex
    TileGymTransform = impl.TileGymTransform
else:
    warnings.warn("tilegym module not found!")
