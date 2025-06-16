from typing import Any
import torch
from thunder.torch.experimental.dtensor_utils import run_only_if_distributed_is_available

if torch.distributed.is_available():
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, DeviceMesh, TensorMeta
    from torch.distributed.tensor import DeviceMesh, Partial, Placement, Replicate, Shard


@run_only_if_distributed_is_available
def populate_object_ctx_for_dtensor_spec(x: Any, object_ctx: dict[str, Any]) -> bool:
    """
    Populate object context for DTensorSpec.

    ..note::
        This function will mutate the `object_ctx`

    Returns:
        bool: True if `x` is DTensorSpec (and also updates `object_ctx`) otherwise False.
    """
    if isinstance(x, DTensorSpec):
        object_ctx.update(
            {
                "DTensorSpec": DTensorSpec,
                "DeviceMesh": DeviceMesh,
                "Placement": Placement,
                "Replicate": Replicate,
                "Shard": Shard,
                "Partial": Partial,
                "TensorMeta": TensorMeta,
            }
        )
        return True
    return False


@run_only_if_distributed_is_available
def prettyprint_dtensor_spec(x):
    if isinstance(x, DTensorSpec):
        return x.__repr__()
    return ""
