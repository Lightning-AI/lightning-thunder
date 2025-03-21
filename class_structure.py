class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype


class Placement:
    def is_shard() -> bool:
        pass

    def is_replicate() -> bool:
        pass

    def is_partial() -> bool:
        pass


class DeviceMesh:
    device_type: str
    mesh: torch.Tensor
    mesh_dim_names: Optional[Tuple[str, ...]]


class DTensorSpec:
    mesh: DeviceMesh
    placements: Tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None


class DTensor:
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
