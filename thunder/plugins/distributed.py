import torch

from torch.distributed.device_mesh import DeviceMesh
from thunder import Plugin
import thunder.core.utils as utils
from thunder.distributed import FSDPType, FSDPBucketingStrategy, copy_default_process_group

from thunder.transforms import MaterializationTransform
from thunder.distributed.transforms.ddp_v2 import DDPTransform
from thunder.distributed.transforms.fsdp_v2 import FSDPTransform


class DDP(Plugin):
    def __init__(
        self,
        bucket_size_in_mb: float = 25.0,
        broadcast_from: int | None = None,
        process_group=None,
    ):
        self.bucket_size_in_mb = bucket_size_in_mb
        self.broadcast_from = broadcast_from
        self.process_group = copy_default_process_group() if process_group is None else process_group
        utils.check(
            self.process_group is not None, lambda: "No process group was defined and default process group is None"
        )

    def setup_transforms(self):
        ddp = DDPTransform(
            process_group=self.process_group,
            bucket_size_in_mb=self.bucket_size_in_mb,
            broadcast_from=self.broadcast_from,
        )
        return [ddp]


class FSDP(Plugin):
    def __init__(
        self,
        device: torch.device | None = None,
        broadcast_from: int | None = None,
        sharding_strategy: FSDPType = FSDPType.ZERO2,
        bucketing_strategy: FSDPBucketingStrategy = FSDPBucketingStrategy.NONE,
        move_state_dict_to_cpu: bool = False,
        ddp_bucket_size_in_mb: float = 25.0,
        process_group=None,
    ):
        self.device = device
        self.broadcast_from = broadcast_from
        self.sharding_strategy = sharding_strategy
        self.bucketing_strategy = bucketing_strategy
        self.move_state_dict_to_cpu = move_state_dict_to_cpu
        self.ddp_bucket_size_in_mb = ddp_bucket_size_in_mb
        self.process_group = copy_default_process_group() if process_group is None else process_group
        utils.check(
            self.process_group is not None, lambda: "No process group was defined and default process group is None"
        )

    def setup_transforms(self):
        pg = self.process_group
        transforms = []

        if isinstance(pg, DeviceMesh):

            dims = pg.mesh_dim_names

            if dims == ("ddp", "fsdp"):
                fsdp_pg = pg["fsdp"].get_group()
                ddp_pg = pg["ddp"].get_group()

                def get_local_rank_in_group(mesh, dim, global_rank):
                    group_ranks = mesh[dim].mesh.flatten().tolist()
                    if global_rank not in group_ranks:
                        raise ValueError(f"Global rank {global_rank} not in mesh_dim '{dim}' group: {group_ranks}")
                    return group_ranks.index(global_rank)

                global_broadcast_from = self.broadcast_from if self.broadcast_from is not None else 0
                fsdp_broadcast_from = get_local_rank_in_group(pg, "fsdp", global_broadcast_from)
                ddp_broadcast_from = get_local_rank_in_group(pg, "ddp", global_broadcast_from)

                fsdp = FSDPTransform(
                    device=self.device,
                    broadcast_from=fsdp_broadcast_from,
                    sharding_strategy=self.sharding_strategy,
                    bucketing_strategy=self.bucketing_strategy,
                    release_original_parameters=True,
                    move_state_dict_to_cpu=self.move_state_dict_to_cpu,
                    process_group=fsdp_pg,
                )
                ddp = DDPTransform(
                    process_group=ddp_pg,
                    bucket_size_in_mb=self.ddp_bucket_size_in_mb,
                    broadcast_from=ddp_broadcast_from,
                )
                transforms.extend([fsdp, ddp])

            elif dims == ("fsdp",):
                fsdp_pg = pg["fsdp"].get_group()

                fsdp = FSDPTransform(
                    device=self.device,
                    broadcast_from=self.broadcast_from,
                    sharding_strategy=self.sharding_strategy,
                    bucketing_strategy=self.bucketing_strategy,
                    release_original_parameters=True,
                    move_state_dict_to_cpu=self.move_state_dict_to_cpu,
                    process_group=fsdp_pg,
                )
                transforms.append(fsdp)

            else:
                raise ValueError(f"Unsupported mesh_dim_names: {dims}")

        else:
            # Plain ProcessGroup
            fsdp = FSDPTransform(
                device=self.device,
                broadcast_from=self.broadcast_from,
                sharding_strategy=self.sharding_strategy,
                bucketing_strategy=self.bucketing_strategy,
                release_original_parameters=True,
                move_state_dict_to_cpu=self.move_state_dict_to_cpu,
                process_group=pg,
            )
            transforms.append(fsdp)

        transforms.append(
            MaterializationTransform(fsdp.device, init=MaterializationTransform.init_from_original_module_init())
        )

        return transforms
