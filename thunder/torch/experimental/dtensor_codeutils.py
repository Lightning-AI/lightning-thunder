from torch.distributed.tensor._dtensor_spec import DTensorSpec, DeviceMesh, TensorMeta
from torch.distributed.tensor import DeviceMesh, DTensor, Partial, Placement, Replicate, Shard  # noqa: F401


def populate_object_ctx_for_dtensor_spec(x, object_ctx):
    if isinstance(x, DTensorSpec):
        object_ctx["DTensorSpec"] = DTensorSpec
        object_ctx["DeviceMesh"] = DeviceMesh
        object_ctx["Placement"] = Placement
        object_ctx["Replicate"] = Replicate
        object_ctx["Shard"] = Shard
        object_ctx["Partial"] = Partial
        object_ctx["TensorMeta"] = TensorMeta
        return True
    return False


def prettyprint_dtensor_spec(x):
    if isinstance(x, DTensorSpec):
        return x.__repr__()
    return ""
