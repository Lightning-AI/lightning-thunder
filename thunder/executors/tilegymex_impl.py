from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from lightning_utilities.core.imports import package_available

import thunder.core.devices as devices
import thunder.core.dtypes as dtypes
from thunder.core.proxies import pyval
from thunder.extend import OperatorExecutor, register_executor
from thunder import Transform
import thunder.torch as ltorch

if TYPE_CHECKING:
    from thunder.torch import TensorLike


if not package_available("tilegym"):
    raise ImportError("tilegym is required for the tilegym executor")

import tilegym
from tilegym import ops as tg_ops


tilegym_ex: OperatorExecutor = OperatorExecutor("tilegym", version=getattr(tilegym, "__version__", None))
register_executor(tilegym_ex)


def _is_cuda_tensor(t: TensorLike) -> bool:
    return t.device.devicetype == devices.DeviceType.CUDA


def _pybool(x) -> bool:
    try:
        return bool(pyval(x))
    except Exception:
        return False


def _pyfloat_or_none(x) -> float | None:
    if x is None:
        return None
    try:
        return float(pyval(x))
    except Exception:
        return None


def _parse_min_cc(s: str) -> tuple[int, int] | None:
    # Accept "10.0", "10,0", or "100" (treated as "10.0").
    s = (s or "").strip()
    if not s:
        return None
    if "." in s:
        a, b = s.split(".", 1)
        return int(a), int(b)
    if "," in s:
        a, b = s.split(",", 1)
        return int(a), int(b)
    if s.isdigit():
        if len(s) >= 2:
            return int(s[:-1]), int(s[-1])
        return int(s), 0
    return None


def _tilegym_device_cc_ok(device_index: int) -> bool:
    # Default to Blackwell+ (SM100). Override via env vars:
    # - THUNDER_TILEGYM_ALLOW_ANY_CC=1  (bypass)
    # - THUNDER_TILEGYM_MIN_CC=10.0     (set minimum)
    if os.environ.get("THUNDER_TILEGYM_ALLOW_ANY_CC", "0").lower() in ("1", "true", "yes", "y", "on"):
        return True

    min_cc = _parse_min_cc(os.environ.get("THUNDER_TILEGYM_MIN_CC", "10.0"))
    if min_cc is None:
        min_cc = (10, 0)

    if not torch.cuda.is_available():
        return False
    try:
        cc = torch.cuda.get_device_capability(device_index)
    except Exception:
        return False

    return tuple(cc) >= tuple(min_cc)


def _tilegym_sdpa_checker(
    query: TensorLike,
    key: TensorLike,
    value: TensorLike,
    attn_mask: TensorLike | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> bool:
    # TileGym kernels are CUDA-only.
    if not (_is_cuda_tensor(query) and _is_cuda_tensor(key) and _is_cuda_tensor(value)):
        return False

    if not _tilegym_device_cc_ok(query.device.index):
        return False

    if key.device != query.device or value.device != query.device:
        return False

    # TileGym kernels currently don't support explicit masks or dropout.
    if attn_mask is not None:
        return False

    try:
        dropout_p_val = float(pyval(dropout_p))
    except Exception:
        return False
    if dropout_p_val != 0.0:
        return False

    is_causal_val = _pybool(is_causal)

    # TileGym attention kernels don't implement backward yet.
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False

    # Expected shapes: (B, H, S, D)
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False

    bq, hq, sq, dq = query.shape
    bk, hk, sk, dk = key.shape
    bv, hv, sv, dv = value.shape

    if bq != bk or bq != bv:
        return False
    if hq != hk or hq != hv:
        # Thunder/torch SDPA expects same number of heads
        return False
    if sk != sv:
        return False
    if dq != dk or dq != dv:
        # TileGym fmha expects Dq == Dk == Dv
        return False

    # TileGym decode kernel assumes non-causal semantics for q_len==1 and k_len>1.
    if sq == 1 and sk > 1 and is_causal_val:
        return False

    # TileGym prefill causal assumes query positions start at 0 and align with keys.
    if is_causal_val and sq != sk:
        return False

    # D requirements: TensorCore-friendly.
    if dq % 8 != 0:
        return False

    # Dtype requirements (TileGym kernels use MMA paths).
    if query.dtype not in (dtypes.float16, dtypes.bfloat16):
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False

    # If scale is symbolic/unknown, we can still run (TileGym defaults to 1/sqrt(D)).
    _ = _pyfloat_or_none(scale)

    return True


def _tilegym_sdpa_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    # Checker guarantees attn_mask is None and dropout_p == 0.0.
    if query.shape[2] == 1 and key.shape[2] > 1:
        # Decode kernel (non-causal semantics expected; checker enforces that)
        return tg_ops.fmha_decode(query, key, value, sm_scale=scale)
    return tg_ops.fmha(query, key, value, scaling=scale, is_causal=is_causal)


tilegym_sdpa = tilegym_ex.register_operator(
    "tilegym_scaled_dot_product_attention",
    like=ltorch.scaled_dot_product_attention,
    fn=_tilegym_sdpa_impl,
)

tilegym_ex.register_implementation(
    ltorch.scaled_dot_product_attention,
    op=tilegym_sdpa,
    checker=_tilegym_sdpa_checker,
)


def _tilegym_rms_norm_checker(
    a: TensorLike,
    normalized_shape,
    weight: TensorLike | None = None,
    eps: float | None = None,
) -> bool:
    if not _is_cuda_tensor(a):
        return False

    if not _tilegym_device_cc_ok(a.device.index):
        return False

    if weight is None:
        # TileGym rms_norm requires affine weight
        return False
    if not _is_cuda_tensor(weight) or weight.device != a.device:
        return False
    if a.dtype not in (dtypes.float16, dtypes.bfloat16, dtypes.float32):
        return False
    if weight.dtype != a.dtype:
        return False
    # TileGym rms_norm doesn't implement backward yet.
    # We only enable this when the *activation* does not require grad
    # (typical inference usage).
    if a.requires_grad:
        return False
    # normalized_shape is validated by the underlying op; keep checker minimal.
    return True


def _tilegym_rms_norm_impl(
    a: torch.Tensor,
    normalized_shape,
    weight: torch.Tensor | None = None,
    eps: float | None = None,
) -> torch.Tensor:
    if eps is None:
        eps = torch.finfo(a.dtype).eps if a.dtype.is_floating_point else 0.0
    # Checker ensures weight is present.
    return tg_ops.rms_norm(a, normalized_shape, weight, eps)


TileGymTransform: Transform | None = None

if hasattr(ltorch, "rms_norm"):
    tilegym_rms_norm = tilegym_ex.register_operator(
        "tilegym_rms_norm",
        like=ltorch.rms_norm,
        fn=_tilegym_rms_norm_impl,
    )
    tilegym_ex.register_implementation(
        ltorch.rms_norm,
        op=tilegym_rms_norm,
        checker=_tilegym_rms_norm_checker,
    )
