from typing import Any
from torch.distributed.tensor._dtensor_spec import DTensorSpec, DeviceMesh, TensorMeta
from torch.distributed.tensor import DeviceMesh, Partial, Placement, Replicate, Shard


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


def prettyprint_dtensor_spec(x):
    if isinstance(x, DTensorSpec):
        return x.__repr__()
    return ""
