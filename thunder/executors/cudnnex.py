from typing import Any

from lightning_utilities.core.imports import package_available
from looseversion import LooseVersion
from thunder.extend import OperatorExecutor, register_executor

__all__ = ["cudnn_version", "required_cudnn_version", "cudnn_available", "cudnn_ex"]


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
cudnn_sdpa_fwd: Any = None
cudnn_sdpa_bwd: Any = None

cudnn = None

if cudnn_available():
    import cudnn
    import thunder.executors.cudnn_sdpa as sdpa_impl

    sdpa_impl.cudnn = cudnn

    cudnn_ex = OperatorExecutor("cudnn", version=cudnn.backend_version())
    register_executor(cudnn_ex)
    cudnn_sdpa_fwd = cudnn_ex.register_operator(
        "cudnn_sdpa_fwd",
        meta=sdpa_impl._cudnn_sdpa_forward_meta,
        fn=sdpa_impl._cudnn_sdpa_fwd_impl,
        tags=(sdpa_impl.OpTags.DONT_AUTO_RECOMPUTE_IN_BACKWARD,),
    )

    cudnn_sdpa_bwd = cudnn_ex.register_operator(
        "cudnn_sdpa_bwd",
        meta=sdpa_impl._cudnn_sdpa_bwd_meta,
        fn=sdpa_impl._cudnn_sdpa_bwd_impl,
    )

    sdpa_impl.cudnn_sdpa_fwd = cudnn_sdpa_fwd
    sdpa_impl.cudnn_sdpa_bwd = cudnn_sdpa_bwd

    cudnn_ex.register_implementation(
        sdpa_impl.ltorch.scaled_dot_product_attention,
        checker=sdpa_impl._cudnn_sdpa_checker,
        execution_transform=sdpa_impl._cudnn_sdpa_fwd_wrapper,
        grad_transform=sdpa_impl._cudnn_sdpa_bwd_wrapper,
    )
