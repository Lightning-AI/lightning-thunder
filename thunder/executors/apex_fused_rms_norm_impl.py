from collections.abc import Sequence
import math


from thunder.core.proxies import TensorProxy
from thunder.torch import TensorLike
from thunder.executors.apexex import apex_ex


APEX_FUSED_NORMS_AVAILABLE = True
try:
    # Fused layer norm is only importable if torch.distributed is available
    # https://github.com/NVIDIA/apex/issues/1853
    from torch.distributed import is_available

    if not is_available():
        raise ImportError
    import fused_layer_norm_cuda
    from apex.normalization.fused_layer_norm import FusedRMSNormAffineMixedDtypesFunction
except ImportError:
    APEX_FUSED_NORMS_AVAILABLE = False


def apex_fused_norms_available() -> bool:
    return APEX_FUSED_NORMS_AVAILABLE


def apex_fused_rms_norm_forward_affine_meta(
    input: TensorLike, normalized_shape: Sequence[int], weight: TensorLike, eps: float
):
    output_or_input = TensorProxy(like=input)
    weight = TensorProxy(like=input, shape=normalized_shape)
    unnormalized_dims = len(input.shape) - len(normalized_shape)
    invvar = TensorProxy(like=input, shape=(math.prod(input.shape[:unnormalized_dims]),))
    return TensorProxy(like=input), invvar


def apex_fused_rms_norm_backward_affine_meta(
    grad_output: TensorLike,
    invvar: TensorLike,
    input_or_output: TensorLike,
    normalized_shape: Sequence[int],
    weight_,
    eps: float,
    memory_efficient: bool,
):
    return TensorProxy(like=grad_output), TensorProxy(like=weight_)


# Create a new symbol and register lookaside only if import is available.
if apex_fused_norms_available():
    apex_fused_rms_norm_forward_affine = apex_ex.register_operator(
        "apex_fused_rms_norm_forward_affine",
        meta=apex_fused_rms_norm_forward_affine_meta,
        fn=fused_layer_norm_cuda.rms_forward_affine,
        replaces=fused_layer_norm_cuda.rms_forward_affine,
    )

    apex_fused_rms_norm_forward_affine_mixed_dtypes = apex_ex.register_operator(
        "apex_fused_rms_norm_forward_affine_mixed_dtypes",
        meta=apex_fused_rms_norm_forward_affine_meta,
        fn=fused_layer_norm_cuda.rms_forward_affine_mixed_dtypes,
        replaces=fused_layer_norm_cuda.rms_forward_affine_mixed_dtypes,
    )

    apex_fused_rms_norm_backward_affine = apex_ex.register_operator(
        "apex_fused_rms_norm_backward_affine",
        meta=apex_fused_rms_norm_backward_affine_meta,
        fn=fused_layer_norm_cuda.rms_backward_affine,
        replaces=fused_layer_norm_cuda.rms_backward_affine,
    )
