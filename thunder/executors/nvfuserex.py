import torch
from looseversion import LooseVersion

from thunder.extend import FusionExecutor

__all__ = ["nvfuser_version", "required_nvfuser_version", "nvfuser_available" "nvfuserex"]


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
    return LooseVersion("0.0.1")


def nvfuser_available() -> bool:
    v = nvfuser_version()
    return v is not None and v >= required_nvfuser_version()


nvfuserex: None | FusionExecutor = None
if nvfuser_available():
    import thunder.executors.nvfuserex_impl as impl

    nvfuserex = impl.ex
