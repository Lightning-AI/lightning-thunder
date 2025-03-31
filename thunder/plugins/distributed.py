import torch

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
        process_group=None,
    ):
        self.device = device
        self.broadcast_from = broadcast_from
        self.sharding_strategy = sharding_strategy
        self.bucketing_strategy = bucketing_strategy
        self.move_state_dict_to_cpu = move_state_dict_to_cpu
        self.process_group = copy_default_process_group() if process_group is None else process_group
        utils.check(
            self.process_group is not None, lambda: "No process group was defined and default process group is None"
        )

    def setup_transforms(self):
        fsdp = FSDPTransform(
            device=self.device,
            broadcast_from=self.broadcast_from,
            sharding_strategy=self.sharding_strategy,
            bucketing_strategy=self.bucketing_strategy,
            release_original_parameters=True,
            move_state_dict_to_cpu=self.move_state_dict_to_cpu,
        )
        materialization = MaterializationTransform(
            fsdp.device, init=MaterializationTransform.init_from_original_module_init()
        )

        return [fsdp, materialization]
