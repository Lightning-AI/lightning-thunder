from collections.abc import Callable

from lightning_utilities.core.imports import package_available
from looseversion import LooseVersion
from thunder.extend import OperatorExecutor

__all__ = [
    "cudnn_version",
    "required_cudnn_version",
    "cudnn_available",
    "cudnn_ex",
    "torch_to_cudnn_dtype",
    "_get_cudnn_handle",
]


#
# Functions for detecting cudnn and its version
#
def cudnn_version() -> LooseVersion | None:
    try:
        import cudnn

        if hasattr(cudnn, "__version__"):
            return LooseVersion(cudnn.__version__)

        # NOTE: This import of cudnn may or may not have version info
        return LooseVersion("0.0.0")
    except ImportError:
        pass

    # NOTE This occurs when cudnn couldn't be imported
    return None


def required_cudnn_version() -> LooseVersion:
    # History of versions:
    # Using 1.3.0+ because it works better with other libraries (e.g. torch) that also build on top of cudnn
    # Using 1.5.0+ because it handles exception with unsupported graphs better
    # Using 1.5.1 because of a compatibility fix
    # Using 1.5.2 to allow stride 0
    return LooseVersion("1.5.2")


def cudnn_available() -> bool:
    v = cudnn_version()
    if v is None:
        return False
    required = required_cudnn_version()
    if v < required:
        import warnings

        msg = f"Your cuDNN installation is out of date. Thunder requires version {required}, but found version {v}."
        warnings.warn(msg)
        return False
    return True


cudnn_ex: None | OperatorExecutor = None
torch_to_cudnn_dtype: None | Callable = None
cudnn = None
# Mapping from device to cudnn handles - shared across all cuDNN executors
_device_to_cudnn_handle = {}


def _get_cudnn_handle(query_device):
    """
    This function creates a new handle for the device that cudnn should
    run its kernels on. As the suggested approach by cudnn is to make a few handles
    as possible, this function caches these per-device handles.
    """

    if not cudnn_available():
        return None
    else:
        import torch
        import cudnn

        handle = _device_to_cudnn_handle.get(query_device, None)
        if handle is None:
            with torch.cuda.device(query_device):
                handle = cudnn.create_handle()
                _device_to_cudnn_handle[query_device] = handle

        # Make sure the user stream is set on the handle
        # Fetch the current user stream and pass the data pointer to set_stream API
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device=query_device).cuda_stream)

        return handle


if cudnn_available():
    import thunder.executors.cudnn_sdpa as sdpa_impl

    torch_to_cudnn_dtype = sdpa_impl.torch_to_cudnn_dtype
    cudnn_ex = sdpa_impl.cudnn_ex
