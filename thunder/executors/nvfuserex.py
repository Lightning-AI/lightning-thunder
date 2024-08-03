import torch
from looseversion import LooseVersion

from thunder.extend import FusionExecutor

__all__ = ["nvfuser_version", "required_nvfuser_version", "nvfuser_available", "nvfuserex"]


#
# Functions for detecting nvFuser and its version
#
def nvfuser_version() -> LooseVersion | None:
    # Short-circuits if CUDA isn't available
    if not torch.cuda.is_available():
        return None

    try:
        import nvfuser

        if hasattr(nvfuser, "version"):
            return LooseVersion(nvfuser.version())

        # NOTE: This import of nvFuser may or may not have version info
        return LooseVersion("0.0.0")
    except ImportError:
        pass

    # NOTE This occurs when nvFuser couldn't be imported
    return None


def required_nvfuser_version() -> LooseVersion:
    return LooseVersion("0.2.8")


def nvfuser_available() -> bool:
    v = nvfuser_version()
    if v is None:
        return False

    required = required_nvfuser_version()
    if v < required:
        import warnings

        msg = f"Your nvfuser installation is out of date. Thunder requires version {required}, but found version {v}."
        warnings.warn(msg)
        return False
    return True


nvfuserex: None | FusionExecutor = None
if nvfuser_available():
    import thunder.executors.nvfuserex_impl as impl

    nvfuserex = impl.ex
